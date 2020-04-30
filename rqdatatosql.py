from rqdatac.client import init as rqdata_init
import pandas as pd
from rqdatac.services.future import get_dominant as rqdata_get_dominant
from rqdatac.services.get_price import get_price as rqdata_get_price
from rqdatac.services.basic import all_instruments as rqdata_all_instruments
from rqdatac.share.errors import QuotaExceeded
from sqlalchemy import create_engine
import sqlalchemy
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float
from multiprocessing import Pool, RLock
import multiprocessing
import sys
import os

RQData_User = '15980235329'
RQData_Password = 'icesola1'


def connect_rqdata():
    rqdata_init(
        RQData_User,
        RQData_Password,
        ('rqdatad-pro.ricequant.com', 16011),
        use_pool=False,
        max_pool_size=8
    )


def download_stock(order_book_id, engine, tbname):
    print('download ' + order_book_id)
    interval = '30m'
    df = rqdata_get_price(order_book_id, start_date='2010-04-15', end_date='2020-04-15', frequency=interval,
                          fields=['open', 'close', 'high', 'low', 'volume', 'total_turnover'])
    df['interval'] = df.apply(lambda x: interval, axis=1)
    df['order_book_id'] = df.apply(lambda x: order_book_id, axis=1)
    df.to_sql(tbname, engine, if_exists='append')
    print('download ' + order_book_id + ' success')


# 下载股票数据
def rqdata_download_cs():
    connect_rqdata()
    cslist = rqdata_all_instruments('CS')
    engine = create_engine('mysql+pymysql://root:123@192.168.1.12/vnpy?charset=utf8')
    # 550
    begin, over = 1800, len(cslist)
    print('rqdata', begin, over)
    with engine.connect() as con:
        cursor = begin
        length = len(cslist)

        tbname = 'xshedata'
        Base = declarative_base()
        # insert into xshedata(order_book_id, datatime, open, close, high, low, volume, total_turnover, interval) values(select order_book_id, datatime, open, close, high, low, volume, total_turnover, interval from ``)
        # 3. 创建表模型
        class User(Base):
            __tablename__ = tbname
            id = Column(Integer, primary_key=True, autoincrement=True)
            order_book_id = Column(String(32))
            datetime = Column(DateTime)
            open = Column(Float)
            close = Column(Float)
            high = Column(Float)
            low = Column(Float)
            volume = Column(Float)
            total_turnover = Column(Float)
            interval = Column(String(32))

            def __repr__(self):
                return self.name

        Base.metadata.create_all(engine)

        while cursor < over and cursor < length:
            row = cslist.iloc[cursor]
            try:
                download_stock(row['order_book_id'], engine, tbname)
            except QuotaExceeded:
                print('pause', cursor)
                raise QuotaExceeded
            except Exception as e:
                print(e)
            cursor += 1

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
def rqdata_download_future():
    connect_rqdata()
    engine = create_engine('mysql+pymysql://root:123@192.168.1.12/vnpy?charset=utf8')
    with engine.connect() as con:
        tbname = 'futuredaydata'
        Base = declarative_base()

        # 3. 创建表模型
        class User(Base):
            __tablename__ = tbname
            id = Column(Integer, primary_key=True, autoincrement=True)
            order_book_id = Column(String(32))
            date = Column(DateTime)
            open = Column(Float)
            close = Column(Float)
            high = Column(Float)
            low = Column(Float)
            volume = Column(Float)
            total_turnover = Column(Float)
            interval = Column(String(32))

            def __repr__(self):
                return self.name
        Base.metadata.create_all(engine)

        for v in Future_List:
            succlist = []
            try:
                print('rqdata future ', v['symbol'])
                df = rqdata_get_dominant(v['symbol'], start_date='2010-04-15', end_date='2020-04-15')
                for val in set(df.values):
                    print('rqdata future ', val)
                    try:
                        download_stock(val, engine, tbname)
                    except QuotaExceeded:
                        raise QuotaExceeded
                    except Exception as e:
                        print(e)
                    succlist.append(val)
            except QuotaExceeded:
                print('success', succlist)
                break
            except Exception as e:
                print(e)


if __name__ == '__main__':
    rqdata_download_cs()