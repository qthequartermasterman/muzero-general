import numpy as np
from gym.spaces import Discrete
import pandas as pd
import pandas_ta as ta
from symfit import parameters, variables, Fit
from sympy import cos, sin
from tensortrade.data.cdd import CryptoDataDownload
from tensortrade.env import default
from tensortrade.env.default.actions import TensorTradeActionScheme, ManagedRiskOrders
from tensortrade.env.default.renderers import PlotlyTradingChart
from tensortrade.env.default.rewards import TensorTradeRewardScheme, RiskAdjustedReturns
from tensortrade.feed.core import DataFeed, Stream
from tensortrade.oms.exchanges import Exchange
from tensortrade.oms.instruments import USD, BTC
from tensortrade.oms.orders.create import proportion_order
from tensortrade.oms.services.execution.simulated import execute_order
from tensortrade.oms.wallets import Portfolio, Wallet
from tensortrade.stochastic.processes import ornstein_uhlenbeck


def rsi(price: Stream[float], period: float) -> Stream[float]:
    r = price.diff()
    upside = r.clamp_min(0).abs()
    downside = r.clamp_max(0).abs()
    rs = upside.ewm(alpha=1 / period).mean() / downside.ewm(alpha=1 / period).mean()
    return 100 * (1 - (1 + rs) ** -1)


def macd(price: Stream[float], fast: float, slow: float, signal: float) -> Stream[float]:
    fm = price.ewm(span=fast, adjust=False).mean()
    sm = price.ewm(span=slow, adjust=False).mean()
    md = fm - sm
    signal = md - md.ewm(span=signal, adjust=False).mean()
    return signal


class PBR(TensorTradeRewardScheme):
    registered_name = "pbr"

    def __init__(self, price: 'Stream'):
        super().__init__()
        self.position = -1

        r = Stream.sensor(price, lambda p: p.value, dtype="float").diff()
        position = Stream.sensor(self, lambda rs: rs.position, dtype="float")

        reward = (r * position).fillna(0).rename("reward")

        self.feed = DataFeed([reward])
        self.feed.compile()

    def on_action(self, action: int):
        self.position = -1 if action == 0 else 1

    def get_reward(self, portfolio: 'Portfolio'):
        return self.feed.next()["reward"]

    def reset(self):
        self.position = -1
        self.feed.reset()


class BSH(TensorTradeActionScheme):
    registered_name = "bsh"

    def __init__(self, cash: 'Wallet', asset: 'Wallet'):
        super().__init__()
        self.cash = cash
        self.asset = asset

        self.listeners = []
        self.action = 0

    @property
    def action_space(self):
        return Discrete(2)

    def attach(self, listener):
        self.listeners += [listener]
        return self

    def get_orders(self, action: int, portfolio: 'Portfolio'):
        order = None

        if abs(action - self.action) > 0:
            src = self.cash if self.action == 0 else self.asset
            tgt = self.asset if self.action == 0 else self.cash
            order = proportion_order(portfolio, src, tgt, 1.0)
            self.action = action

        for listener in self.listeners:
            listener.on_action(action)

        return [order]

    def reset(self):
        super().reset()
        self.action = 0


def fourier_series(x, f, n=0):
    """Creates a symbolic fourier series of order `n`.

    Parameters
    ----------
    x : `symfit.Variable`
        The input variable for the function.
    f : `symfit.Parameter`
        Frequency of the fourier series
    n : int
        Order of the fourier series.
    """
    # Make the parameter objects for all the terms
    a0, *cos_a = parameters(','.join(['a{}'.format(i) for i in range(0, n + 1)]))
    sin_b = parameters(','.join(['b{}'.format(i) for i in range(1, n + 1)]))

    # Construct the series
    series = a0 + sum(ai * cos(i * f * x) + bi * sin(i * f * x)
                      for i, (ai, bi) in enumerate(zip(cos_a, sin_b), start=1))
    return series


def gbm(price: float,
        mu: float,
        sigma: float,
        dt: float,
        n: int) -> np.array:
    """Generates a geometric brownian motion path.

    Parameters
    ----------
    price : float
        The initial price of the series.
    mu : float
        The percentage drift.
    sigma : float
        The percentage volatility.
    dt : float
        The time step size.
    n : int
        The number of steps to be generated in the path.

    Returns
    -------
    `np.array`
        The generated path.
    """
    y = np.exp((mu - sigma ** 2 / 2) * dt + sigma * np.random.normal(0, np.sqrt(dt), size=n).T)
    y = price * y.cumprod(axis=0)
    return y


def fourier_gbm(price, mu, sigma, dt, n, order):
    x, y = variables('x, y')
    w, = parameters('w')
    model_dict = {y: fourier_series(x, f=w, n=order)}

    # Make step function data
    xdata = np.arange(-np.pi, np.pi, 2 * np.pi / n)
    ydata = np.log(gbm(price, mu, sigma, dt, n))

    # Define a Fit object for this model and data
    fit = Fit(model_dict, x=xdata, y=ydata)
    fit_result = fit.execute()

    return np.exp(fit.model(x=xdata, **fit_result.params).y)


def create_sine_env(config):
    x = np.arange(0, 2 * np.pi, 2 * np.pi / 1001)
    y = 50 * np.sin(2 * x) + 100
    data = price_date_to_ohlcv(y, x)
    return create_env(data, config)


def price_date_to_ohlcv(price, date):
    close_price = price
    open_price = np.roll(close_price, 1)
    high = np.maximum(open_price, close_price)
    low = np.minimum(open_price, close_price)
    volume = np.random.normal(50, 5)

    return pd.DataFrame(data={'date': date,
                              'open': open_price,
                              'high': high,
                              'low': low,
                              'close': close_price,
                              'volume': volume})


def create_fourier_gbm_env(config):
    # config = {**config, "y": fourier_gbm(price=100, mu=0.01, sigma=0.5, dt=0.01, n=1000, order=5)}
    y = gbm(price=100, mu=0.01, sigma=0.5, dt=0.01, n=1000)
    x = np.arange(0, 2 * np.pi, 2 * np.pi / 1000)

    data = price_date_to_ohlcv(y, x)

    return create_env(data, config)


def create_ornstein_uhlenbeck_env(config):
    data = ornstein_uhlenbeck.ornstein(time_frame='1H', times_to_generate=500)
    return create_env(data, config)


def create_recent_coinbase_env(config):
    cdd = CryptoDataDownload()
    data = cdd.fetch("gemini", "USD", "BTC", "1h")
    return create_env(data, config)


def create_env(data, config):
    features = []
    for c in data.columns[1:]:
        s = Stream.source(list(data[c]), dtype="float").rename(data[c].name)
        features += [s]

    cp = Stream.select(features, lambda s: s.name == "close")
    high_price = data['high']
    low_price = data['low']
    close_price = data['close']
    # print(data['volume'])

    try:
        data['date']
    except KeyError:
        data['date'] = data.index

    features = [
        cp.rename('USD/BTC'),
        cp.log().diff().rename("lr"),
        rsi(cp, period=14).rename("rsi"),
        macd(cp, fast=10, slow=50, signal=5).rename("macd"),
        Stream.source(ta.cci(high_price, low_price, close_price)).rename('cci')
        # cp.rolling(window=10).mean().rename("fast"),
        # cp.rolling(window=50).mean().rename("medium"),
        # cp.rolling(window=100).mean().rename("slow")
    ]

    feed = DataFeed(features)
    feed.compile()

    # for i in range(5):
    #     print(feed.next())

    coinbase = Exchange("coinbase", service=execute_order)(cp)

    cash = Wallet(coinbase, 100000 * USD)
    asset = Wallet(coinbase, 10 * BTC)

    portfolio = Portfolio(USD, [cash, asset])

    # reward_scheme = PBR(price=cp)
    reward_scheme = RiskAdjustedReturns()
    # action_scheme = BSH(cash=cash, asset=asset).attach(reward_scheme)
    action_scheme = ManagedRiskOrders()

    renderer_feed = DataFeed([
        Stream.source(list(data["date"])).rename("date"),
        Stream.source(list(data["open"]), dtype="float").rename("open"),
        Stream.source(list(data["high"]), dtype="float").rename("high"),
        Stream.source(list(data["low"]), dtype="float").rename("low"),
        Stream.source(list(data["close"]), dtype="float").rename("close"),
        Stream.source(list(data["volume"]), dtype="float").rename("volume"),
        #Stream.sensor(action_scheme, lambda s: s.action, dtype="float").rename("action")
    ])

    renderer = PlotlyTradingChart()

    environment = default.create(
        feed=feed,
        portfolio=portfolio,
        action_scheme=action_scheme,  # The DQN example uses action_scheme="managed-risk"
        reward_scheme=reward_scheme,  # The DQN uses reward_scheme="risk-adjusted"
        renderer_feed=renderer_feed,
        renderer=renderer,
        window_size=config["window_size"],
        max_allowed_loss=0.6
    )
    return environment
