import tushare as ts
from sqlalchemy import create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy import Column, Integer, String, DateTime, Float
import pandas as pd
import time

START_DATE = '20000601'
END_DATA = '20200601'

ts.set_token('5f9bdb72fd4dff979a64102a2f33f067c0118fc6fb4ce3001a27d5ae')
pro = ts.pro_api()
engine = create_engine('mysql+pymysql://root:123@192.168.1.12/vnpy?charset=utf8')


def _reset_primary_key(tbname):
    with engine.connect() as con:
        con.execute('alter table {} add column `id` bigint(20) NOT NULL PRIMARY KEY AUTO_INCREMENT'.format(tbname))


def save_stock_daily_basic(data):
    tbname = 'stock_daily_basic'
    for index, row in data.iterrows():
        df = pro.daily_basic(ts_code=row['ts_code'], start_date=START_DATE, end_date=END_DATA)
        df.to_sql(tbname, engine, if_exists='append', index=False)
        print("download {0}:{1} success".format(tbname, row['ts_code']))
        time.sleep(1)
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)


def save_stock_income(data):
    tbname = 'stock_income'
    for index, row in data.iterrows():
        df = pro.income(ts_code=row['ts_code'], start_date=START_DATE, end_date=END_DATA)
        df.to_sql(tbname, engine, if_exists='append', index=False)
        time.sleep(1)
        print("download {0}:{1} success".format(tbname, row['ts_code']))
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)


def save_stock_balancesheet(data):
    tbname = 'stock_balancesheet'
    for index, row in data.iterrows():
        df = pro.balancesheet(ts_code=row['ts_code'], start_date=START_DATE, end_date=END_DATA)
        df.to_sql(tbname, engine, if_exists='append', index=False)
        time.sleep(1)
        print("download {0}:{1} success".format(tbname, row['ts_code']))
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)


def save_stock_cashflow(data):
    tbname = 'stock_cashflow'
    for index, row in data.iterrows():
        df = pro.cashflow(ts_code=row['ts_code'], start_date=START_DATE, end_date=END_DATA)
        df.to_sql(tbname, engine, if_exists='append', index=False)
        time.sleep(1)
        print("download {0}:{1} success".format(tbname, row['ts_code']))
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)


def save_stock_daily(data):
    tbname = 'stock_daily'
    for index, row in data.iterrows():
        df = pro.daily(ts_code=row['ts_code'], start_date=START_DATE, end_date=END_DATA)
        df.to_sql(tbname, engine, if_exists='append', index=False)
        print("download {0}:{1} success".format(tbname, row['ts_code']))
        time.sleep(1)
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)


def save_stock_adj_factor(data):
    tbname = 'stock_adj_factor'
    for index, row in data.iterrows():
        df = pro.adj_factor(ts_code=row['ts_code'], start_date=START_DATE, end_date=END_DATA)
        df.to_sql(tbname, engine, if_exists='append', index=False)
        print("download {0}:{1} success".format(tbname, row['ts_code']))
        time.sleep(1)
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)


def save_stock_basic():
    # 3. 创建表模型
    tbname = 'stock_basic'
    data = pro.stock_basic(exchange='', list_status='L')
    data.to_sql(tbname, engine, if_exists='replace', index=False)
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)
    return data


if __name__ == "__main__":
    data = save_stock_basic()
    save_stock_cashflow(data)
    save_stock_balancesheet(data)
    save_stock_income(data)
    save_stock_adj_factor(data)
    save_stock_daily_basic(data)
    save_stock_daily(data)

