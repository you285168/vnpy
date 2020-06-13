from vnpy.app.cta_strategy import (
    CtaTemplate,
)
from peewee import (
    AutoField,
    CharField,
    FloatField,
    Model,
)
import datetime


class StockStrategy(CtaTemplate):
    stock_income = {}
    stock_daily = {}
    Year = 3
    Days = 60

    parameters = ["Year", "Days"]

    def __init__(self, cta_engine, strategy_name, vt_symbol, setting):
        """"""
        super().__init__(cta_engine, strategy_name, vt_symbol, setting)

    def load_income(self, db, start, end):
        # 股票财报表
        class StockIncome(Model):
            ts_code: str = CharField()
            end_date: str = CharField()
            basic_eps: float = FloatField()
            operate_profit: float = FloatField()
            n_income: float = FloatField()

            class Meta:
                database = db
                table_name = 'stock_income'

        incomes = StockIncome.select().where((StockIncome.end_date % '%1231')
                                             & (StockIncome.end_date >= start.strftime("%Y%m%d"))
                                             & (StockIncome.end_date <= end.strftime("%Y%m%d"))
                                             ).order_by(StockIncome.end_date)
        temp = [cur.__data__ for cur in incomes]
        for data in temp:
            ts_code = data['ts_code']
            if  ts_code not in self.stock_income:
                self.stock_income[ts_code] = []
            self.stock_income[ts_code].append(data)

    def load_daily_basic(self, db, start, end, ts_code):
        class StockDailyBasic(Model):
            ts_code: str = CharField()
            trade_date: str = CharField()
            pe: float = FloatField()
            turnover_rate_f: float = FloatField()

            class Meta:
                database = db
                table_name = 'stock_daily_basic'
                indexes = ((("trade_date", "ts_code"), True),)
        basics = StockDailyBasic.select().where((StockDailyBasic.trade_date >= start.strftime("%Y%m%d"))
                                                & (StockDailyBasic.trade_date <= end.strftime("%Y%m%d"))
                                                & (StockDailyBasic.ts_code == ts_code)
                                                ).order_by(StockDailyBasic.trade_date)
        self.stock_daily[ts_code] = [cur.__data__ for cur in basics]

    def load_daily(self, db, start, end, ts_code):
        if ts_code not in self.stock_daily or len(self.stock_daily) <=0:
            print('not stock daily base', ts_code)
            return
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
                database = db
                table_name = 'stock_daily'

        daily = StockDaily.select().where((StockDaily.trade_date >= start.strftime("%Y%m%d"))
                                                & (StockDaily.trade_date <= end.strftime("%Y%m%d"))
                                                & (StockDaily.ts_code == ts_code)
                                                ).order_by(StockDaily.trade_date)
        temp = [cur.__data__ for cur in daily]
        day_info = self.stock_daily[ts_code]
        for ix, data in enumerate(temp):
            day_data = day_info[ix]
            if day_data['trade_date'] != data['trade_data']:
                print('stock daily error', day_data['trade_date'], data['trade_data'])
            else:
                day_data.update(data)
        print(self.stock_daily)



    def load_data(self, db, start, end):
        self.load_income(db, start, end)

        # 连续N年利润大于0 加载每日数据
        for ts_code, incomes in self.stock_income.items():
            year = 0
            for data in incomes:
                if getattr(data, 'basic_eps', 0) > 0 and getattr(data, 'n_income', 0) > 0 \
                    and getattr(data, 'operate_profit', 0) > 0:
                    year += 1
                else:
                    year = 0
                if year >= self.Year:
                    dd = datetime.datetime.strptime(data['end_date'], "%Y%m%d")
                    s = dd - datetime.timedelta(days=self.Days)
                    e = dd + datetime.timedelta(days=365)
                    self.load_daily_basic(db, s, e, data['ts_code'])
                    self.load_daily(db, s, e, data['ts_code'])
                    pass


    def on_init(self):
        """
        Callback when strategy is inited.
        """
        self.write_log("策略初始化")
        self.load_bar(self.test_day)
