from vnpy.app.cta_strategy import (
    CtaTemplate,
    StopOrder,
    TickData,
    BarData,
    TradeData,
    OrderData,
    BarGenerator,
    ArrayManager,
)
import numpy as np
import talib


class FutureStrategy(CtaTemplate):
    author = "用Python的交易员"

    test_day = 30

    parameters = ["test_day"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()
        self.param_array: np.ndarray = np.zeros(100)

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(30)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")
        self.put_event()

    def on_stop(self):
        """
        Callback when strategy is stopped.
        """
        self.write_log("策略停止")

        self.put_event()

    def on_tick(self, tick: TickData):
        """
        Callback of new tick data update.
        """
        self.bg.update_tick(tick)

    def on_bar(self, bar: BarData):
        """
        Callback of new bar data update.
        """

        am = self.am
        am.update_bar(bar)
        self.param_array[:-1] = self.param_array[1:]
        self.param_array[-1] = (bar.low_price + bar.high_price + bar.close_price * 2) / 4
        if not self.inited:
            return

        mac = talib.MA(self.param_array, 20)[-1]
        qg = max(am.open[-2], am.close[-2])
        qd = max(am.open[-2], am.close[-2])
        diff = am.ema(12, True) - am.ema(26, True)
        dea = talib.EMA(diff, 9)
        if self.pos > 0:
            if bar.close_price < mac and bar.close_price < qd:
                self.sell(bar.close_price, 1)
        if self.pos < 0:
            if bar.close_price > mac and bar.close_price > qg:
                self.cover(bar.close_price, 1)
        if self.pos == 0:
            if bar.close_price > mac and bar.close_price > qg and diff[-1] > dea[-1] and bar.close_price > bar.open_price:
                self.buy(bar.close_price, 1)
            if bar.close_price < mac and bar.close_price < qg and diff[-1] < dea[-1] and bar.close_price < bar.open_price:
                self.short(bar.close_price, 1)

        self.put_event()

    def on_order(self, order: OrderData):
        """
        Callback of new order data update.
        """
        pass

    def on_trade(self, trade: TradeData):
        """
        Callback of new trade data update.
        """
        self.put_event()

    def on_stop_order(self, stop_order: StopOrder):
        """
        Callback of stop order update.
        """
        pass
