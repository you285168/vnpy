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
from datetime import date, datetime, timedelta
import numpy as np
import talib
from vnpy.app.cta_strategy.future_params import Future_Params, get_symbol_flag


class MainFutureStrategy(CtaTemplate):
    author = "用Python的交易员"

    test_day = 30
    macd_param1 = 6
    macd_param2 = 13
    macd_param3 = 5
    mac_day = 15

    parameters = ["test_day", "macd_param1", "macd_param2", "macd_param3","mac_day"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = ArrayManager()
        self.active_symbol = None
        self.cache_bar = {}
        self.symbol_expire = {}     # 各个合约的结束日期
        self.change_flag = False

    def get_new_bar_price(self, bar):
        if self.active_symbol:
            temp = bar[self.active_symbol]
            return temp.close_price
        else:
            for temp in bar.values():
                return temp.close_price

    def get_new_bar_datatime(self, bar):
        for temp in bar.values():
            return temp.datetime

    def get_active_symbol(self):
        return self.active_symbol

    def get_like_symbol(self, symbol):
        """
        load data with like symbol
        """
        return symbol + '____'

    def init_history(self, history_data):
        """
        init history bar data
        """
        for data in history_data:
            for bar in data.values():
                self.symbol_expire[bar.symbol] = bar.datetime

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(self.test_day)

    def is_expire(self, symbol):
        if self.active_symbol:
            return self.symbol_expire[symbol] < self.symbol_expire[self.active_symbol]

    def change_active_symbol(self):
        self.change_flag = False
        best_bar = None
        for data in list(self.cache_bar.values()):
            bar = data[-1]
            if self.symbol_expire[bar.symbol] - bar.datetime <= timedelta(days=15):
                self.cache_bar.pop(bar.symbol)
                continue
            if not best_bar:
                best_bar = bar
            elif bar.volume > best_bar.volume:
                best_bar = bar
            elif bar.volume == best_bar.volume and self.symbol_expire[bar.symbol] < self.symbol_expire[best_bar.symbol]:
                best_bar = bar
        if not best_bar:
            print('not active symbol', self.active_symbol)
            self.active_symbol = None
            return
        self.active_symbol = best_bar.symbol
        flag = get_symbol_flag(best_bar.symbol)
        param = Future_Params[flag]
        param['pos'] = int(100000 / (best_bar.close_price * self.cta_engine.get_size() * 0.3))
        print('active symbol', best_bar.symbol)
        data = self.cache_bar[self.active_symbol]
        self.am = ArrayManager()
        for bar in data:
            am = self.am
            am.update_bar(bar)
            am.update_extra((bar.low_price + bar.high_price + bar.close_price * 2) / 4)
        for symbol in list(self.cache_bar.keys()):
            if self.is_expire(symbol):
                self.cache_bar.pop(symbol)

    def on_start(self):
        """
        Callback when strategy is started.
        """
        self.write_log("策略启动")
        self.put_event()
        self.change_active_symbol()

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

    def on_bar(self, data):
        """
        Callback of new bar data update.
        """
        if self.change_flag and self.pos == 0:
            self.change_active_symbol()
        best_bar = None
        for bar in data.values():
            if self.is_expire(bar.symbol):
                continue
            if bar.symbol not in self.cache_bar:
                self.cache_bar[bar.symbol] = []
            self.cache_bar[bar.symbol].append(bar)
            if not best_bar or bar.volume > best_bar.volume:
                best_bar = bar
        if not self.active_symbol:
            return

        # 交易合约是否切换
        expire = self.symbol_expire[self.active_symbol]
        bar = self.cache_bar[self.active_symbol][-1]
        am = self.am
        am.update_bar(bar)
        am.update_extra((bar.low_price + bar.high_price + bar.close_price * 2) / 4)
        self.cta_engine.set_order_bar(bar)
        change = False
        flag = get_symbol_flag(self.active_symbol)
        one_pos = Future_Params[flag]['pos']
        if expire - bar.datetime <= timedelta(days=7):
            # 小于7天平仓做主力合约
            change = True
        elif expire - bar.datetime <= timedelta(days=15):
            # 小于15天未开仓做主力合约
            if self.pos == 0:
                change = True
        elif best_bar.volume > 0:
            if not self.change_flag and bar.volume / best_bar.volume < 0.35:
                # 小于主力合约35% 下一次平仓做主力合约
                if self.pos == 0:
                    change = True
                else:
                    self.change_flag = True
            elif bar.volume / best_bar.volume < 0.1:
                # 小于主力合约10%平仓做主力合约
                change = True
        if change:
            if self.pos > 0:
                self.sell(bar.close_price, one_pos)
            elif self.pos < 0:
                self.cover(bar.close_price, one_pos)
            self.change_flag = True
            # self.change_active_symbol()

        elif len(self.cache_bar[self.active_symbol]) >= self.test_day:
            mac = talib.MA(am.extra, self.mac_day)[-1]
            qg = max(am.open[-2], am.close[-2])
            qd = min(am.open[-2], am.close[-2]) #这个地方我改成了min

            macd, signal, hist = am.macd(self.macd_param1, self.macd_param2, self.macd_param3, True)
            if self.pos > 0:
                if bar.close_price < mac and bar.close_price < qd:
                    self.sell(bar.close_price, one_pos)
            if self.pos < 0:
                if bar.close_price > mac and bar.close_price > qg:
                    self.cover(bar.close_price, one_pos)
            if self.pos == 0:
                if bar.close_price > mac and bar.close_price > qg and macd[-1] > signal[-1] and bar.close_price > bar.open_price :
                    self.buy(bar.close_price, one_pos)
                if bar.close_price < mac and bar.close_price < qd and macd[-1] < signal[-1] and bar.close_price < bar.open_price :
                    self.short(bar.close_price, one_pos)

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
