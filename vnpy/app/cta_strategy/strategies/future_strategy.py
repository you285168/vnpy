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
    macd_param1 = 12
    macd_param2 = 6
    macd_param3 = 9

    parameters = ["test_day", "macd_param1", "macd_param2", "macd_param3"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(self.test_day)

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
        am.update_extra((bar.low_price + bar.high_price + bar.close_price * 2) / 4)
        if not self.inited:
            return

        mac = talib.MA(am.extra, 20)[-1]
        qg = max(am.open[-2], am.close[-2])
        qd = min(am.open[-2], am.close[-2]) #这个地方我改成了min

        macd, signal, hist = am.macd(self.macd_param1, self.macd_param2, self.macd_param3, True)

        if self.pos > 0:
            if bar.close_price < mac and bar.close_price < qd:
                self.sell(bar.close_price, 1)
        if self.pos < 0:
            if bar.close_price > mac and bar.close_price > qg:
                self.cover(bar.close_price, 1)
        if self.pos == 0:
            if bar.close_price > mac and bar.close_price > qg and macd[-1] > signal[-1] and bar.close_price > bar.open_price :
                self.buy(bar.close_price, 1)
            if bar.close_price < mac and bar.close_price < qg and macd[-1] < signal[-1] and bar.close_price < bar.open_price :
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
