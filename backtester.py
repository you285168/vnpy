import datetime
from backtester.engine import BacktesterEngine

def main():
    class_name = 'DoubleMaStrategy'
    vt_symbol = 'IF88.CFFEX'
    interval = '1m'
    start = datetime.date(2017, 3, 1)
    end = datetime.date.today()
    rate = 2.5
    slippage = 0.2
    size = 300
    pricetick = 0.2
    capital = 1000000
    inverse = True #正向
    backtesting_setting = {
        "class_name": class_name,
        "vt_symbol": vt_symbol,
        "interval": interval,
        "rate": rate,
        "slippage": slippage,
        "size": size,
        "pricetick": pricetick,
        "capital": capital,
        "inverse": inverse,
    }
    new_setting = {
        'fast_window': 10,
        'slow_window': 20,
    }

    engine = BacktesterEngine()
    engine.init_engine()
    result = engine.start_backtesting(
        class_name,
        vt_symbol,
        interval,
        start,
        end,
        rate,
        slippage,
        size,
        pricetick,
        capital,
        inverse,
        new_setting
    )


if __name__ == "__main__":
    main()