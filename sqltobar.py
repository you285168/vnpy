from rqdatac.share.errors import QuotaExceeded
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float
import pandas as pd

''' 主合约

'''
Future_List = [
    {'symbol': 'A', 'exchange': 'DCE', },
    {'symbol': 'AG', 'exchange': 'SHFE', },
    {'symbol': 'AL', 'exchange': 'SHFE', },
    {'symbol': 'AP', 'exchange': 'CZCE', },
    {'symbol': 'AU', 'exchange': 'SHFE', },
    {'symbol': 'B', 'exchange': 'DCE', },
    {'symbol': 'BB', 'exchange': 'DCE', },
    {'symbol': 'BU', 'exchange': 'SHFE', },
    {'symbol': 'C', 'exchange': 'DCE', },
    {'symbol': 'CF', 'exchange': 'CZCE', },
    {'symbol': 'CJ', 'exchange': 'CZCE', },
    {'symbol': 'CS', 'exchange': 'DCE', },
    {'symbol': 'CU', 'exchange': 'SHFE', },
    {'symbol': 'CY', 'exchange': 'CZCE', },
    {'symbol': 'EB', 'exchange': 'DCE', },
    {'symbol': 'EG', 'exchange': 'DCE', },
    {'symbol': 'FB', 'exchange': 'DCE', },
    {'symbol': 'FG', 'exchange': 'CZCE', },
    {'symbol': 'FU', 'exchange': 'SHFE', },
    {'symbol': 'HC', 'exchange': 'SHFE', },
    {'symbol': 'I', 'exchange': 'DCE', },
    {'symbol': 'IC', 'exchange': 'CFFEX', },
    {'symbol': 'ICA', 'exchange': 'CFFEX', },
    {'symbol': 'IF', 'exchange': 'CFFEX', },
    {'symbol': 'IFA', 'exchange': 'CFFEX', },
    {'symbol': 'IH', 'exchange': 'CFFEX', },
    {'symbol': 'IHA', 'exchange': 'CFFEX', },
    {'symbol': 'J', 'exchange': 'DCE', },
    {'symbol': 'JD', 'exchange': 'DCE', },
    {'symbol': 'JM', 'exchange': 'DCE', },
    {'symbol': 'JR', 'exchange': 'CZCE', },
    {'symbol': 'L', 'exchange': 'DCE', },
    {'symbol': 'LR', 'exchange': 'CZCE', },
    {'symbol': 'M', 'exchange': 'DCE', },
    {'symbol': 'MA', 'exchange': 'CZCE', },
    {'symbol': 'NI', 'exchange': 'SHFE', },
    {'symbol': 'NR', 'exchange': 'INE', },
    {'symbol': 'OI', 'exchange': 'CZCE', },
    {'symbol': 'P', 'exchange': 'DCE', },
    {'symbol': 'PB', 'exchange': 'SHFE', },
    {'symbol': 'PG', 'exchange': 'DCE', },
    {'symbol': 'PM', 'exchange': 'CZCE', },
    {'symbol': 'PP', 'exchange': 'DCE', },
    {'symbol': 'RB', 'exchange': 'SHFE', },
    {'symbol': 'RI', 'exchange': 'CZCE', },
    {'symbol': 'RM', 'exchange': 'CZCE', },
    {'symbol': 'RR', 'exchange': 'DCE', },
    {'symbol': 'RS', 'exchange': 'CZCE', },
    {'symbol': 'RU', 'exchange': 'SHFE', },
    {'symbol': 'SA', 'exchange': 'CZCE', },
    {'symbol': 'SC', 'exchange': 'INE', },
    {'symbol': 'SF', 'exchange': 'CZCE', },
    {'symbol': 'SM', 'exchange': 'CZCE', },
    {'symbol': 'SN', 'exchange': 'SHFE', },
    {'symbol': 'SP', 'exchange': 'SHFE', },
    {'symbol': 'SR', 'exchange': 'CZCE', },
    {'symbol': 'SS', 'exchange': 'SHFE', },
    {'symbol': 'T', 'exchange': 'CFFEX', },
    {'symbol': 'TA', 'exchange': 'CZCE', },
    {'symbol': 'TF', 'exchange': 'CFFEX', },
    {'symbol': 'TS', 'exchange': 'CFFEX', },
    {'symbol': 'UR', 'exchange': 'CZCE', },
    {'symbol': 'V', 'exchange': 'DCE', },
    {'symbol': 'WH', 'exchange': 'CZCE', },
    {'symbol': 'WR', 'exchange': 'SHFE', },
    {'symbol': 'Y', 'exchange': 'DCE', },
    {'symbol': 'ZC', 'exchange': 'CZCE', },
    {'symbol': 'ZN', 'exchange': 'SHFE', },
]


# 下载股票数据
def future_to_bar():
    engine = create_engine('mysql+pymysql://root:123456@127.0.0.1/vnpy?charset=utf8')

    # 3. 创建表模型
    Base = declarative_base()
    tbname = 'dbbardata'
    class BarData(Base):
        __tablename__ = tbname
        id = Column(Integer, primary_key=True, autoincrement=True)
        symbol = Column(String(32))
        exchange = Column(String(32))
        datetime = Column(DateTime)
        open_price = Column(Float)
        low_price = Column(Float)
        high_price = Column(Float)
        close_price = Column(Float)
        volume = Column(Float)
        total_turnover = Column(Float)
        interval = Column(String(32))
        open_interest = Column(Float)

        def __repr__(self):
            return self.name
    Base.metadata.create_all(engine)

    for data in Future_List:
        print('load data {0}'.format(data['symbol']))
        sql = 'select * from futuredaydata where order_book_id like "{0}____" order by datetime asc'.format(data['symbol'])
        df = pd.read_sql_query(sql, engine, index_col='datetime')
        print('load data {0} success number {1}'.format(data['symbol'], len(df)))
        df.rename(columns={
            'order_book_id': 'symbol',
            'open': 'open_price',
            'close': 'close_price',
            'high': 'high_price',
            'low': 'low_price',
        }, inplace=True)
        df['exchange'] = df.apply(lambda x: data['exchange'], axis=1)
        df['open_interest'] = df.apply(lambda x: 0, axis=1)
        del df['id']
        df.to_sql(tbname, engine, if_exists='append')
        print('save data {0} success'.format(data['symbol']))


if __name__ == "__main__":
    # future_to_bar()
    pass
