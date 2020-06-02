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


class HedgeStrategy(CtaTemplate):
    author = "用Python的交易员"

    N = 20 #最近幾天
    X = 5 #漲幅最大的5只
    X1 = 3 #閾值
    Y = 5#跌幅最大的5只
    Y1 = 3 #閾值
    Z = 7 #持有时间
    P = 5
    P1 = 5
    plan = 3

    parameters = ["N", "X", "X1", "Y", "Y1", "Z", "P", "P1", "plan"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

        self.bg = BarGenerator(self.on_bar)
        self.am = {}
        self.active_symbol = {}
        self.cache_bar = {}
        self.symbol_expire = {}     # 各个合约的结束日期
        self.day = 0

    def get_new_bar_price(self, bar):
        close_price = {}
        for temp in bar.values():
            close_price[temp.symbol] = temp.close_price
        return close_price

    def get_new_bar_datatime(self, bar):
        for temp in bar.values():
            return temp.datetime

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
        self.load_bar(self.N)

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

    def reset_symbol_pos(self, bar):
        flag = get_symbol_flag(bar.symbol)
        param = Future_Params[flag]
        pos = int(150000 / (bar.close_price * param['size'] * 0.3))
        return pos

    def select_symbol(self, best_bardata):
        """
        选取所有市场的主力合约，最近N（配置参数7）日上涨百分比最大的X（参数5）只产品，买入
        选取所有市场的主力合约，最近N（配置参数7）日下跌百分比最大的Y（参数5）只产品，卖出
        """
        rise = []
        fall = []
        for flag, best_bar in best_bardata.items():
            symbol = best_bar.symbol
            if len(self.cache_bar[symbol]) < self.N:
                pass
            else:
                maxbar = max(self.cache_bar[symbol][-self.N:-2], key=lambda x: x.close_price)
                minbar = min(self.cache_bar[symbol][-self.N:-2], key=lambda x: x.close_price)
                val = (best_bar.close_price / maxbar.close_price - 1) * 100
                val1 = (best_bar.close_price / minbar.close_price - 1) * 100
                if val1 >= self.X1:
                    rise.append({
                        'bar': best_bar,
                        'val': val,
                    })
                elif val <= -self.Y1:
                    fall.append({
                        'bar': best_bar,
                        'val': val,
                    })
        rise.sort(key=lambda x: x['val'], reverse=True)
        fall.sort(key=lambda x: abs(x['val']), reverse=True)
        for i in range(min(len(rise), self.X)):
            bar = rise[i]['bar']
            one_pos = self.reset_symbol_pos(bar)
            self.cta_engine.set_order_bar(bar)
            self.buy(bar.close_price, one_pos)
            self.active_symbol[bar.symbol] = {
                'price': bar.close_price,
                'pos': one_pos,
            }
            self.put_event()
        for i in range(min(len(fall), self.Y)):
            bar = fall[i]['bar']
            one_pos = self.reset_symbol_pos(bar)
            self.cta_engine.set_order_bar(bar)
            self.short(bar.close_price, one_pos)
            self.active_symbol[bar.symbol] = {
                'price': bar.close_price,
                'pos': -one_pos,
            }
            self.put_event()
        self.day = 0

    def is_expire(self, bar):
        expire = self.symbol_expire[bar.symbol]
        return bar.datetime + timedelta(days=7) >= expire

    def clear_all_symbol(self):
        for symbol, temp in self.active_symbol.items():
            pos = temp['pos']
            bar = self.cache_bar[symbol][-1]
            self.cta_engine.set_order_bar(bar)
            if pos > 0:
                self.sell(bar.close_price, pos)
            elif pos < 0:
                self.cover(bar.close_price, abs(pos))
            self.put_event()
        self.active_symbol.clear()

    def clear_symbol_pos(self, expire):
        for symbol in expire:
            temp = self.active_symbol[symbol]
            if not temp:
                assert False
            pos = temp['pos']
            bar = self.cache_bar[symbol][-1]
            self.cta_engine.set_order_bar(bar)
            if pos > 0:
                self.sell(bar.close_price, pos)
            elif pos < 0:
                self.cover(bar.close_price, abs(pos))
            self.put_event()

    def on_bar(self, data):
        """
        Callback of new bar data update.
        """
        best_bardata = {}
        for bar in data.values():
            flag = get_symbol_flag(bar.symbol)
            if flag not in Future_Params:
                continue
            if bar.symbol not in self.cache_bar:
                self.cache_bar[bar.symbol] = []
            self.cache_bar[bar.symbol].append(bar)
            if not self.is_expire(bar):     # 剩余天数 小于等于7天 排除
                best_bar = best_bardata.get(flag, None)
                if not best_bar or bar.volume > best_bar.volume:
                    best_bardata[flag] = bar
        self.day += 1
        if not self.inited:
            return
        if len(self.active_symbol) <= 0:
            # 选取对冲合约
            self.select_symbol(best_bardata)
            return
        if self.day >= self.Z:
            # 持续持有Z（参数）日后全部平仓 第二天重新开始选取对冲合约
            self.clear_all_symbol()
            return

        # 止损方案
        # if self.plan == 1:
        #     self.loss_plan_1()
        # elif self.plan == 2:
        #     self.loss_plan_2()
        # elif self.plan == 3:
        #     self.loss_plan_3()

    def loss_plan_1(self):
        # 若单品种亏损超过P%（10%），则平仓该品种，同时平仓对冲品种中亏损最大的一个品种
        expire = set()
        rise = []
        fall = []
        for symbol, temp in self.active_symbol.items():
            pos = temp['pos']
            price = temp['price']
            bar = self.cache_bar[symbol][-1]
            if pos > 0 and bar.close_price < price:
                loss = abs(bar.close_price - price) / price * 100
                rise.append({
                    'symbol': symbol,
                    'loss': loss,
                })
            elif pos < 0 and bar.close_price > price:
                loss = abs(bar.close_price - price) / price * 100
                fall.append({
                    'symbol': symbol,
                    'loss': loss,
                })
        rise.sort(key=lambda x: x['loss'], reverse=True)
        fall.sort(key=lambda x: x['loss'], reverse=True)
        for symbol, temp in self.active_symbol.items():
            if symbol in expire:
                continue
            pos = temp['pos']
            price = temp['price']
            bar = self.cache_bar[symbol][-1]
            loss_list = None
            if pos > 0 and bar.close_price < price:
                loss = abs(bar.close_price - price) / price * 100
                if loss >= self.P:
                    loss_list = fall
            elif pos < 0 and bar.close_price > price:
                loss = abs(bar.close_price - price) / price * 100
                if loss >= self.P:
                    loss_list = rise
            if loss_list is not None:
                expire.add(symbol)
                for val in loss_list:
                    if val['symbol'] not in expire:
                        expire.add(val['symbol'])
                        break
        self.clear_symbol_pos(expire)
        for symbol in expire:
            self.active_symbol.pop(symbol)

    def loss_plan_2(self):
        # 若整体亏损超过P1（2%），则平仓所有品种
        loss_price = 0
        max_price = 0
        for symbol, temp in self.active_symbol.items():
            pos = temp['pos']
            price = temp['price']
            bar = self.cache_bar[symbol][-1]
            flag = get_symbol_flag(bar.symbol)
            param = Future_Params[flag]
            if pos > 0:
                loss_price += (bar.close_price - price) * abs(pos) * param['size']
            elif pos < 0 and bar.close_price > price:
                loss_price += (price - bar.close_price) * abs(pos) * param['size']
            max_price += price * abs(pos) * param['size']
        if loss_price / max_price * 100 <= -self.P1:
            self.clear_all_symbol()

    def loss_plan_3(self):
        # 若单品种亏损超过P %（10 %），则平仓该品种
        expire = set()
        for symbol, temp in self.active_symbol.items():
            pos = temp['pos']
            price = temp['price']
            bar = self.cache_bar[symbol][-1]
            if pos > 0 and bar.close_price < price:
                loss = abs(bar.close_price - price) / price * 100
                if loss >= self.P:
                    expire.add(symbol)
            elif pos < 0 and bar.close_price > price:
                loss = abs(bar.close_price - price) / price * 100
                if loss >= self.P:
                    expire.add(symbol)
        self.clear_symbol_pos(expire)
        for symbol in expire:
            self.active_symbol.pop(symbol)

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