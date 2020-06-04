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
con = engine.connect()


def _reset_primary_key(tbname):
    con.execute('alter table {} add column `id` bigint(20) NOT NULL PRIMARY KEY AUTO_INCREMENT'.format(tbname))
    con.execute('alter table {} add ADD INDEX tscode (`ts_code`);'.format(tbname))


def _select_ts_code(tbname, tscode):
    result = con.execute('select * from {0} where ts_code="{1}" limit 1;'.format(tbname, tscode))
    cur = result.fetchone()
    return cur


def _save_stock_data(data, tbname, func, *, dtype, check=False, begin=None, over=None, sleep_time=1.2):
    for index, row in data.iterrows():
        ts_code = row['ts_code']
        if begin and index <= begin:
            continue
        if over and index > over:
            break
        if not check or not _select_ts_code(tbname, ts_code):
            df = pro.query(func, ts_code=ts_code, start_date=START_DATE, end_date=END_DATA)
            print(ts_code, len(df))
            df.to_sql(tbname, engine, if_exists='append', index=False, dtype=dtype)
            time.sleep(sleep_time)
            print("download {0}:{1} success".format(tbname, ts_code))
    print("download {} success".format(tbname))
    # _reset_primary_key(tbname)


def save_stock_basic():
    # 3. 创建表模型
    tbname = 'stock_basic'
    data = pro.stock_basic(exchange='', list_status='L')
    data.to_sql(tbname, engine, if_exists='replace', index=False, dtype={
        'ts_code': String(64),
        'trade_date': String(64),
    })
    print("download {} success".format(tbname))
    _reset_primary_key(tbname)
    return data


if __name__ == "__main__":
    data = save_stock_basic()
    _save_stock_data(data, 'stock_daily_basic', 'daily_basic', sleep_time=0.1, dtype={
        'ts_code': String(64),
        'trade_date': String(64),
    })
    _save_stock_data(data, 'stock_income', 'income', sleep_time=1.2, dtype={
        'ts_code': String(64),
        'ann_date': String(64),
        'f_ann_date': String(64),
        'end_date': String(64),
        'report_type': String(64),
        'comp_type': String(64),
        'prem_earned': String(64),
        'prem_income': String(64),
        'out_prem': String(64),
        'une_prem_reser': String(64),
        'reins_income': String(64),
        'n_sec_tb_income': String(64),
        'n_sec_uw_income': String(64),
        'n_asset_mg_income': String(64),
    })
    _save_stock_data(data, 'stock_balancesheet', 'balancesheet', sleep_time=1.2, dtype={
        'ts_code': String(64),
        'ann_date': String(64),
        'f_ann_date': String(64),
        'end_date': String(64),
        'report_type': String(64),
        'comp_type': String(64),
    })
    _save_stock_data(data, 'stock_cashflow', 'cashflow', sleep_time=1.2, dtype={
        'ts_code': String(64),
        'ann_date': String(64),
        'f_ann_date': String(64),
        'end_date': String(64),
        'report_type': String(64),
        'comp_type': String(64),
    })
    
    _save_stock_data(data, 'stock_adj_factor', 'adj_factor', sleep_time=0.1, dtype={
        'ts_code': String(64),
        'trade_date': String(64),
    })
    _save_stock_data(data, 'stock_daily', 'daily', sleep_time=0.1, dtype={
        'ts_code': String(64),
        'trade_date': String(64),
    })
    con.close()

