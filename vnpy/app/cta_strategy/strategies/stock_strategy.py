from vnpy.app.cta_strategy import (
    CtaTemplate,
)
from peewee import (
    CharField,
    FloatField,
    Model,
)
from vnpy.trader.object import BarData
from vnpy.trader.constant import Exchange
import datetime


class StockStrategy(CtaTemplate):
    stock_income = {}
    stock_daily = []
    valid_income = []
    Year = 3
    AvgDays = 60
    LastOverride = 2
    AvgOverride = 2
    database = None
    cursor = 0
    PEmin = 1
    PEmax = 50
    AvgPrice = 15
    buy_price = 0
    StopLoss = 0.96
    AvgSell = 15
    stock_price = {}
    last_stock_price = {}
    N = 3  # 计算N个有效股票
    cur_date = None
    end_date = None

    parameters = ["Year", "AvgDays", "LastOverride", "PEmin", "PEmax", "AvgPrice", "StopLoss", "AvgSell", "N"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

    def load_income(self, start, end):
        # 股票财报表
        class StockIncome(Model):
            ts_code: str = CharField()
            end_date: str = CharField()
            basic_eps: float = FloatField()
            operate_profit: float = FloatField()
            n_income: float = FloatField()

            class Meta:
                database = self.database
                table_name = 'stock_income'

        incomes = StockIncome.select().where((StockIncome.end_date % '%1231')
                                             & (StockIncome.end_date >= start.strftime("%Y%m%d"))
                                             & (StockIncome.end_date <= end.strftime("%Y%m%d"))
                                             ).order_by(StockIncome.end_date)
        temp = [cur.__data__ for cur in incomes]
        for data in temp:
            ts_code = data['ts_code']
            if ts_code not in self.stock_income:
                self.stock_income[ts_code] = []
            self.stock_income[ts_code].append(data)

    def load_daily(self, start, end, ts_code):
        class StockDailyBasic(Model):
            ts_code: str = CharField()
            trade_date: str = CharField()
            pe: float = FloatField()
            turnover_rate_f: float = FloatField()

            class Meta:
                database = self.database
                table_name = 'stock_daily_basic'
                indexes = ((("trade_date", "ts_code"), True),)
        basics = StockDailyBasic.select().where((StockDailyBasic.trade_date >= start.strftime("%Y%m%d"))
                                                & (StockDailyBasic.trade_date <= end.strftime("%Y%m%d"))
                                                & (StockDailyBasic.ts_code == ts_code)
                                                ).order_by(StockDailyBasic.trade_date)
        daily_basic = [cur.__data__ for cur in basics]
        if len(daily_basic) <= 0:
            print('not stock daily base', ts_code)
            return False
        # 股票日表
        class StockDaily(Model):
            ts_code: str = CharField()
            trade_date: str = CharField()
            open: float = FloatField()
            high: float = FloatField()
            low: float = FloatField()
            close: float = FloatField()
            pre_close: float = FloatField()
            change: float = FloatField()
            vol: float = FloatField()
            amount: float = FloatField()

            class Meta:
                database = self.database
                table_name = 'stock_daily'
                indexes = ((("trade_date", "ts_code"), True),)
        daily = StockDaily.select().where((StockDaily.trade_date >= start.strftime("%Y%m%d"))
                                                & (StockDaily.trade_date <= end.strftime("%Y%m%d"))
                                                & (StockDaily.ts_code == ts_code)
                                                ).order_by(StockDaily.trade_date)
        vt_symbol = ts_code.split(".")
        symbol = vt_symbol[0]
        exchange = Exchange(vt_symbol[1])

        temp = [cur.__data__ for cur in daily]
        old = len(self.stock_daily)
        for ix, data in enumerate(temp):
            day_data = daily_basic[ix]
            if day_data['trade_date'] != data['trade_date']:
                print('stock daily error', day_data['trade_date'], data['trade_date'])
            else:
                bar = BarData(
                    symbol=symbol,
                    exchange=exchange,
                    datetime=datetime.datetime.strptime(data['trade_date'], "%Y%m%d"),
                    interval="d",
                    volume=data['vol'],
                    open_price=data['open'],
                    high_price=data['high'],
                    open_interest=0,
                    low_price=data['low'],
                    close_price=data['close'],
                    gateway_name="DB",
                    turnover_rate_f=day_data['turnover_rate_f'],
                    pe=day_data['pe'],
                    cursor=ix + old,
                )
                self.stock_daily.append(bar)
        return True

    def load_data(self, db, start, end):
        self.database = db
        self.load_income(start, end)
        self.end_date = datetime.datetime.strptime(str(end), '%Y-%m-%d')

        # 连续N年利润大于0 加载每日数据
        i = 0
        for ts_code, incomes in self.stock_income.items():
            year = 0
            succ = False
            for data in incomes:
                basic_eps = data.get('basic_eps') or 0
                n_income = data.get('n_income') or 0
                operate_profit = data.get('operate_profit') or 0
                if operate_profit > 0 and n_income > 0 and basic_eps > 0:
                    year += 1
                else:
                    year = 0
                if year >= self.Year:
                    self.valid_income.append(data)
                    succ = True
            if succ:
                i += 1
            if i >= self.N:
                break
        return True

    def is_over(self):
        return self.cursor >= len(self.valid_income)

    def get_history_bar(self):
        if self.pos > 0:
            # 还未平仓 下一条数据不是接下来的一年则继续加载下一年数据
            cur = self.valid_income[self.cursor - 1]
            is_nex = False
            cur_date = datetime.datetime.strptime(cur['end_date'], "%Y%m%d")
            if not self.is_over():
                nex = self.valid_income[self.cursor]
                if nex['ts_code'] == cur['ts_code']:
                    nex_date = datetime.datetime.strptime(nex['end_date'], "%Y%m%d")
                    if nex_date.replace(year=nex_date.year - 1) == cur_date:
                        is_nex = True
            if not is_nex:
                old = len(self.stock_daily)
                start = cur_date.replace(year=cur_date.year + 1) + datetime.timedelta(days=1)
                end = cur_date.replace(year=cur_date.year + 2)
                self.cur_date = start
                if self.load_daily(start, end, cur['ts_code']):
                    return self.stock_daily[old:]
                else:
                    print("not daily data", str(start), str(end))
                    # 没有数据了
        self.stock_daily.clear()
        while not self.is_over():
            data = self.valid_income[self.cursor]
            dd = datetime.datetime.strptime(data['end_date'], "%Y%m%d")
            start = dd - datetime.timedelta(days=self.AvgDays * 2)
            end = dd.replace(year=dd.year + 1)
            self.cursor += 1
            self.cur_date = dd
            if self.load_daily(start, end, data['ts_code']):
                break
        if len(self.stock_daily) <= self.AvgDays:
            return None
        return self.stock_daily[self.AvgDays:]

    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")

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

    def on_bar(self, bar):
        """
        Callback of new bar data update.
        """
        if bar.datetime <= self.cur_date:
            return
        if bar.datetime >= self.end_date:
            if self.pos > 0:
                # 最后一天直接平仓
                self.sell(bar.close_price, self.pos)
            return
        last = self.stock_daily[bar.cursor - 1]
        if bar.symbol not in self.stock_price:
            self.stock_price[bar.symbol] = {}
            self.last_stock_price[bar.symbol] = {}
        self.last_stock_price[bar.symbol][bar.datetime.date()] = last.close_price
        self.stock_price[bar.symbol][bar.datetime.date()] = bar.close_price
        self.cta_engine.set_order_bar(bar)
        if self.pos == 0:
            # 当天换手率除以昨日换手率大于等于2（参数可调
            if bar.turnover_rate_f / last.turnover_rate_f < self.LastOverride:
                return
            # 当天换手率除以前60天平均换手率大于等于2
            avg = sum(bar.turnover_rate_f for bar in self.stock_daily[bar.cursor - self.AvgDays: bar.cursor - 1]) / self.AvgDays
            if bar.turnover_rate_f / avg < self.LastOverride:
                return
            # closeprice>openprice 且今日closeprice>昨日closeprice
            if bar.close_price <= bar.open_price or bar.close_price <= last.close_price:
                return
            # 市盈率 pe 指标大于1且pe指标小于50
            if bar.pe <= self.PEmin or bar.pe >= self.PEmax:
                return
            # 收盘价格没有涨停（今日closeprice/昨日closeprice<1.09999）
            if bar.close_price / last.close_price >= 1.09999:
                return
            # closeprice>15日均线时
            price = sum(bar.close_price for bar in self.stock_daily[bar.cursor - self.AvgPrice: bar.cursor - 1]) / self.AvgPrice
            if bar.close_price <= price:
                return
            # 条件全部满足  买入

            pos = int(50000 / bar.close_price / 100) * 100
            self.buy(bar.close_price, pos)
            self.buy_price = bar.close_price
        else:
            if bar.close_price / self.buy_price < self.StopLoss:
                # 当日closeprice / 买入价格 < 0.96 强制以买入价格的0.96倍平仓，强制成交
                price = self.buy_price * self.StopLoss
                self.sell(price, self.pos)
            else:
                # 当日closeprice < 15日均线时卖出平仓 以收盘价平仓设置为必定成交
                price = sum(bar.close_price for bar in self.stock_daily[bar.cursor - self.AvgSell: bar.cursor - 1]) / self.AvgSell
                if bar.close_price < price:
                    self.sell(bar.close_price, self.pos)
        self.put_event()

    def get_stock_price(self, symbol, date):
        return self.stock_price[symbol].get(date, 0)

    def get_last_price(self, symbol, date):
        return self.last_stock_price[symbol].get(date, 0)
