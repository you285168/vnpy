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
import re
import talib
from vnpy.app.cta_strategy.future_params import Future_Params, get_symbol_flag


class AllFutureStrategy(CtaTemplate):
    author = "用Python的交易员"

    test_day = 30
    macd_param1 = 12
    macd_param2 = 26
    macd_param3 = 9
    mac_day = 20

    parameters = ["test_day", "macd_param1", "macd_param2", "macd_param3","mac_day"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = {}
        self.active_symbol = {}
        self.cache_bar = {}
        self.symbol_expire = {}     # 各个合约的结束日期
        self.change_flag = set()
        self.symbol_pos = {}

    def get_new_bar_price(self, bar):
        close_price = {}
        for temp in bar.values():
            close_price[temp.symbol] = temp.close_price
        return close_price

    def get_new_bar_datatime(self, bar):
        for temp in bar.values():
            return temp.datetime

    def add_symbol_pos(self, symbol, pos_change):
        flag = get_symbol_flag(symbol)
        old = self.symbol_pos.get(flag, 0)
        self.symbol_pos[flag] = old + pos_change

    def get_symbol_pos(self, flag):
        return self.symbol_pos.get(flag, 0)

    def get_active_symbol(self, symbol):
        flag = get_symbol_flag(symbol)
        return self.active_symbol.get(flag, None)

    def get_like_symbol(self, symbol):
        """
        load data with like symbol
        """
        return '%'

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
        main_symbol = self.get_active_symbol(symbol)
        if main_symbol:
            return self.symbol_expire[symbol] < self.symbol_expire[main_symbol]

    def change_active_symbol(self, flag):
        best_bar = None
        for data in list(self.cache_bar.values()):
            bar = data[-1]
            if get_symbol_flag(bar.symbol) != flag:
                continue
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
            self.active_symbol[flag] = None
            print('change_active_symbol not best symbol', flag)
            return
        self.active_symbol[flag] = best_bar.symbol
        param = Future_Params[flag]
        param['pos'] = int(100000/ (best_bar.close_price * self.cta_engine.get_size() * 0.3))
        print('active symbol', best_bar.symbol)
        data = self.cache_bar[best_bar.symbol]
        self.am[flag] = ArrayManager()
        am = self.am[flag]
        for bar in data:
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
        for data in list(self.cache_bar.values()):
            bar = data[-1]
            flag = get_symbol_flag(bar.symbol)
            self.change_flag.add(flag)
        for flag in self.change_flag:
            self.change_active_symbol(flag)
        self.change_flag.clear()

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
        for flag in list(self.change_flag):
            if self.get_symbol_pos(flag) == 0:
                self.change_active_symbol(flag)
                self.change_flag.remove(flag)
        best_bardata = {}
        for bar in data.values():
            flag = get_symbol_flag(bar.symbol)
            if flag not in Future_Params:
                continue
            if self.is_expire(bar.symbol):
                continue
            if bar.symbol not in self.cache_bar:
                self.cache_bar[bar.symbol] = []
            self.cache_bar[bar.symbol].append(bar)
            best_bar = best_bardata.get(flag, None)
            if not best_bar or bar.volume > best_bar.volume:
                best_bardata[flag] = bar
        if not self.inited:
            return
        for flag, best_bar in best_bardata.items():
            # 交易合约是否切换
            active_symbol = self.active_symbol.get(flag, None)
            if not active_symbol:
                if len(self.cache_bar[best_bar.symbol]) >= self.test_day:
                    self.change_active_symbol(flag)
                    active_symbol = self.active_symbol.get(flag, None)
            if not active_symbol:
                continue
            expire = self.symbol_expire[active_symbol]
            bar = self.cache_bar[active_symbol][-1]
            am = self.am[flag]
            am.update_bar(bar)
            am.update_extra((bar.low_price + bar.high_price + bar.close_price * 2) / 4)
            self.cta_engine.set_order_bar(bar)
            change = False
            pos = self.get_symbol_pos(flag)
            one_pos = Future_Params[flag]['pos']
            if bar.datetime + timedelta(days=7) >= expire:
                # 小于7天平仓做主力合约
                print(bar.symbol, expire - bar.datetime)
                change = True
            elif bar.datetime + timedelta(days=15) >= expire:
                # 小于15天未开仓做主力合约
                if pos == 0:
                    change = True
                print(bar.symbol, expire - bar.datetime)
            elif best_bar.volume > 0:
                if flag not in self.change_flag and bar.volume / best_bar.volume < 0.01:
                    # 小于主力合约35% 下一次平仓做主力合约
                    if pos == 0:
                        change = True
                    else:
                        self.change_flag.add(flag)
                elif bar.volume / best_bar.volume < 0.001:
                    # 小于主力合约10%平仓做主力合约
                    change = True
            if change:
                if pos > 0:
                    self.sell(bar.close_price, one_pos)
                elif pos < 0:
                    self.cover(bar.close_price, one_pos)
                self.change_flag.add(flag)
            elif len(self.cache_bar[active_symbol]) >= self.test_day:
                mac = talib.MA(am.extra, self.mac_day)[-1]
                qg = max(am.open[-2], am.close[-2])
                qd = min(am.open[-2], am.close[-2]) #这个地方我改成了min

                macd, signal, hist = am.macd(self.macd_param1, self.macd_param2, self.macd_param3, True)
                if pos > 0:
                    if bar.close_price < mac and bar.close_price < qd:
                        self.sell(bar.close_price, one_pos)
                if pos < 0:
                    if bar.close_price > mac and bar.close_price > qg:
                        self.cover(bar.close_price, one_pos)
                if pos == 0:
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
