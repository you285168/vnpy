from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Callable
from itertools import product
from functools import lru_cache
from time import time
import multiprocessing
import random
import traceback

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pandas import DataFrame
from deap import creator, base, tools, algorithms

from vnpy.trader.constant import (Direction, Offset, Exchange,
                                  Interval, Status)
from vnpy.trader.database import database_manager
from vnpy.trader.object import OrderData, TradeData, BarData, TickData
from vnpy.trader.utility import round_to
from vnpy.app.cta_strategy.future_params import Future_Params, get_symbol_flag

from .base import (
    BacktestingMode,
    EngineType,
    STOPORDER_PREFIX,
    StopOrder,
    StopOrderStatus,
    INTERVAL_DELTA_MAP
)
from .template import CtaTemplate

# Set seaborn style
sns.set_style("whitegrid")

# Set deap algo
creator.create("FitnessMax", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMax)


class OptimizationSetting:
    """
    Setting for runnning optimization.
    """

    def __init__(self):
        """"""
        self.params = {}
        self.target_name = ""

    def add_parameter(
        self, name: str, start: float, end: float = None, step: float = None
    ):
        """"""
        if not end and not step:
            self.params[name] = [start]
            return

        if start >= end:
            print("参数优化起始点必须小于终止点")
            return

        if step <= 0:
            print("参数优化步进必须大于0")
            return

        value = start
        value_list = []

        while value <= end:
            value_list.append(value)
            value += step

        self.params[name] = value_list

    def set_target(self, target_name: str):
        """"""
        self.target_name = target_name

    def generate_setting(self):
        """"""
        keys = self.params.keys()
        values = self.params.values()
        products = list(product(*values))

        settings = []
        for p in products:
            setting = dict(zip(keys, p))
            settings.append(setting)

        return settings

    def generate_setting_ga(self):
        """"""
        settings_ga = []
        settings = self.generate_setting()
        for d in settings:
            param = [tuple(i) for i in d.items()]
            settings_ga.append(param)
        return settings_ga


class BacktestingEngine:
    """"""

    engine_type = EngineType.BACKTESTING
    gateway_name = "BACKTESTING"

    def __init__(self):
        """"""
        self.vt_symbol = ""
        self.symbol = ""
        self.exchange = None
        self.start = None
        self.end = None
        self.rate = 0
        self.slippage = 0
        self.size = 1
        self.pricetick = 0
        self.capital = 1_000_000
        self.mode = BacktestingMode.BAR
        self.inverse = False

        self.strategy_class = None
        self.strategy = None
        self.tick: TickData
        self.bar = None
        self.order_bar: BarData
        self.datetime = None

        self.interval = None
        self.days = 0
        self.callback = None
        self.history_data = []

        self.stop_order_count = 0
        self.stop_orders = {}
        self.active_stop_orders = {}

        self.limit_order_count = 0
        self.limit_orders = {}
        self.active_limit_orders = {}

        self.trade_count = 0
        self.trades = {}

        self.logs = []

        self.daily_results = {}
        self.daily_df = None

    def get_size(self):
        return self.size

    def clear_data(self):
        """
        Clear all data of last backtesting.
        """
        self.strategy = None
        self.tick = None
        self.bar = None
        self.order_bar = None
        self.datetime = None

        self.stop_order_count = 0
        self.stop_orders.clear()
        self.active_stop_orders.clear()

        self.limit_order_count = 0
        self.limit_orders.clear()
        self.active_limit_orders.clear()

        self.trade_count = 0
        self.trades.clear()

        self.logs.clear()
        self.daily_results.clear()

    def set_parameters(
        self,
        vt_symbol: str,
        interval: Interval,
        start: datetime,
        rate: float,
        slippage: float,
        size: float,
        pricetick: float,
        capital: int = 0,
        end: datetime = None,
        mode: BacktestingMode = BacktestingMode.BAR,
        inverse: bool = False
    ):
        """"""
        self.mode = mode
        self.vt_symbol = vt_symbol
        self.interval = Interval(interval)
        self.rate = rate
        self.slippage = slippage
        self.size = size
        self.pricetick = pricetick
        self.start = start

        temp = self.vt_symbol.split(".")
        self.symbol = temp[0]
        if len(temp) > 1:
            exchange_str = temp[1]
            self.exchange = Exchange(exchange_str)

        self.capital = capital
        self.end = end
        self.mode = mode
        self.inverse = inverse

    def add_strategy(self, strategy_class: type, setting: dict):
        """"""
        self.strategy_class = strategy_class
        self.strategy = strategy_class(
            self, strategy_class.__name__, self.vt_symbol, setting
        )

    def load_data(self):
        """"""
        self.output("开始加载历史数据")

        if not self.end:
            self.end = datetime.now()

        if self.start >= self.end:
            self.output("起始日期必须小于结束日期")
            return

        self.history_data.clear()       # Clear previously loaded history data

        # Load 30 days of data each time and allow for progress update

        total_delta = self.end - self.start
        interval_delta = INTERVAL_DELTA_MAP[self.interval]
        progress_delta = timedelta(days=(30 * interval_delta.total_seconds() / 60))

        start = self.start
        end = self.start + progress_delta
        progress = 0

        while start < self.end:
            end = min(end, self.end)  # Make sure end time stays within set range

            if self.mode == BacktestingMode.BAR:
                like = self.strategy.get_like_symbol(self.symbol)

                data = load_bar_data(
                    self.symbol,
                    self.exchange,
                    self.interval,
                    start,
                    end,
                    like
                )

                if like:
                    temp_data = data[:]
                    data = []
                    old_time = None
                    temp = {}
                    for bar in temp_data:
                        if bar.datetime != old_time:
                            old_time = bar.datetime
                            if len(temp) > 0:
                                data.append(temp)
                                temp = {}
                        temp[bar.symbol] = bar
                    if len(temp) > 0:
                        data.append(temp)
            else:
                data = load_tick_data(
                    self.symbol,
                    self.exchange,
                    start,
                    end
                )

            self.history_data.extend(data)

            progress += progress_delta / total_delta
            progress = min(progress, 1)
            progress_bar = "#" * int(progress * 10)
            self.output(f"加载进度：{progress_bar} [{progress:.0%}]")

            start = end + interval_delta
            end += (progress_delta + interval_delta)

        self.output(f"历史数据加载完成，数据量：{len(self.history_data)}")

    def run_backtesting(self):
        """"""
        if self.mode == BacktestingMode.BAR:
            func = self.new_bar
        else:
            func = self.new_tick

        self.strategy.on_init()
        self.strategy.init_history(self.history_data)

        # Use the first [days] of history data for initializing strategy
        day_count = 0
        ix = 0

        for ix, data in enumerate(self.history_data):
            bardate = self.strategy.get_new_bar_datatime(data)
            if self.datetime and bardate.day != self.datetime.day:
                day_count += 1
                if day_count >= self.days:
                    break

            self.datetime = bardate

            try:
                self.callback(data)
            except Exception:
                self.output("触发异常，回测终止")
                self.output(traceback.format_exc())
                return

        self.strategy.inited = True
        self.output("策略初始化完成")

        self.strategy.on_start()
        self.strategy.trading = True
        self.output("开始回放历史数据")

        # Use the rest of history data for running backtesting
        for data in self.history_data[ix:]:
            try:
                func(data)
            except Exception:
                self.output("触发异常，回测终止")
                self.output(traceback.format_exc())
                return

        self.output("历史数据回放结束")

    def calculate_result(self):
        """"""
        self.output("开始计算逐日盯市盈亏")

        if not self.trades:
            self.output("成交记录为空，无法计算")
            return

        # Add trade data into daily reuslt.
        for trade in self.trades.values():
            d = trade.datetime.date()
            daily_result = self.daily_results[d]
            daily_result.add_trade(trade)

        # Calculate daily result by iteration.
        pre_close = None
        start_pos = None

        size = self.size
        flag = get_symbol_flag(self.symbol)
        if flag in Future_Params:
            param = Future_Params[flag]
            size = param.get('size', size)

        for daily_result in self.daily_results.values():
            daily_result.calculate_pnl(
                pre_close,
                start_pos,
                size,
                self.rate,
                self.slippage,
                self.inverse
            )

            pre_close = daily_result.close_price
            start_pos = daily_result.end_pos

        # Generate dataframe
        results = defaultdict(list)

        for daily_result in self.daily_results.values():
            for key, value in daily_result.__dict__.items():
                results[key].append(value)

        self.daily_df = DataFrame.from_dict(results).set_index("date")

        self.output("逐日盯市盈亏计算完成")
        return self.daily_df

    def calculate_statistics(self, df: DataFrame = None, output=True):
        """"""
        self.output("开始计算策略统计指标")

        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            # Set all statistics to 0 if no trade.
            start_date = ""
            end_date = ""
            total_days = 0
            profit_days = 0
            loss_days = 0
            end_balance = 0
            max_drawdown = 0
            max_ddpercent = 0
            max_drawdown_duration = 0
            total_net_pnl = 0
            daily_net_pnl = 0
            total_commission = 0
            daily_commission = 0
            total_slippage = 0
            daily_slippage = 0
            total_turnover = 0
            daily_turnover = 0
            total_trade_count = 0
            daily_trade_count = 0
            total_return = 0
            annual_return = 0
            daily_return = 0
            return_std = 0
            sharpe_ratio = 0
            return_drawdown_ratio = 0
        else:
            # Calculate balance related time series data
            df["balance"] = df["net_pnl"].cumsum() + self.capital
            df["return"] = np.log(df["balance"] / df["balance"].shift(1)).fillna(0)
            df["highlevel"] = (
                df["balance"].rolling(
                    min_periods=1, window=len(df), center=False).max()
            )
            df["drawdown"] = df["balance"] - df["highlevel"]
            df["ddpercent"] = df["drawdown"] / df["highlevel"] * 100

            # Calculate statistics value
            start_date = df.index[0]
            end_date = df.index[-1]

            total_days = len(df)
            profit_days = len(df[df["net_pnl"] > 0])
            loss_days = len(df[df["net_pnl"] < 0])

            end_balance = df["balance"].iloc[-1]
            max_drawdown = df["drawdown"].min()
            max_ddpercent = df["ddpercent"].min()
            if max_ddpercent == 0:
                max_ddpercent = 1
            max_drawdown_end = df["drawdown"].idxmin()

            if isinstance(max_drawdown_end, date):
                max_drawdown_start = df["balance"][:max_drawdown_end].idxmax()
                max_drawdown_duration = (max_drawdown_end - max_drawdown_start).days
            else:
                max_drawdown_duration = 0

            total_net_pnl = df["net_pnl"].sum()
            daily_net_pnl = total_net_pnl / total_days

            total_commission = df["commission"].sum()
            daily_commission = total_commission / total_days

            total_slippage = df["slippage"].sum()
            daily_slippage = total_slippage / total_days

            total_turnover = df["turnover"].sum()
            daily_turnover = total_turnover / total_days

            total_trade_count = df["trade_count"].sum()
            daily_trade_count = total_trade_count / total_days

            total_return = (end_balance / self.capital - 1) * 100
            annual_return = total_return / total_days * 240
            daily_return = df["return"].mean() * 100
            return_std = df["return"].std() * 100

            if return_std:
                sharpe_ratio = daily_return / return_std * np.sqrt(240)
            else:
                sharpe_ratio = 0

            return_drawdown_ratio = -total_return / max_ddpercent

        # Output
        if output:
            self.output("-" * 30)
            self.output(f"首个交易日：\t{start_date}")
            self.output(f"最后交易日：\t{end_date}")

            self.output(f"总交易日：\t{total_days}")
            self.output(f"盈利交易日：\t{profit_days}")
            self.output(f"亏损交易日：\t{loss_days}")

            self.output(f"起始资金：\t{self.capital:,.2f}")
            self.output(f"结束资金：\t{end_balance:,.2f}")

            self.output(f"总收益率：\t{total_return:,.2f}%")
            self.output(f"年化收益：\t{annual_return:,.2f}%")
            self.output(f"最大回撤: \t{max_drawdown:,.2f}")
            self.output(f"百分比最大回撤: {max_ddpercent:,.2f}%")
            self.output(f"最长回撤天数: \t{max_drawdown_duration}")

            self.output(f"总盈亏：\t{total_net_pnl:,.2f}")
            self.output(f"总手续费：\t{total_commission:,.2f}")
            self.output(f"总滑点：\t{total_slippage:,.2f}")
            self.output(f"总成交金额：\t{total_turnover:,.2f}")
            self.output(f"总成交笔数：\t{total_trade_count}")

            self.output(f"日均盈亏：\t{daily_net_pnl:,.2f}")
            self.output(f"日均手续费：\t{daily_commission:,.2f}")
            self.output(f"日均滑点：\t{daily_slippage:,.2f}")
            self.output(f"日均成交金额：\t{daily_turnover:,.2f}")
            self.output(f"日均成交笔数：\t{daily_trade_count}")

            self.output(f"日均收益率：\t{daily_return:,.2f}%")
            self.output(f"收益标准差：\t{return_std:,.2f}%")
            self.output(f"Sharpe Ratio：\t{sharpe_ratio:,.2f}")
            self.output(f"收益回撤比：\t{return_drawdown_ratio:,.2f}")

        statistics = {
            "start_date": start_date,
            "end_date": end_date,
            "total_days": total_days,
            "profit_days": profit_days,
            "loss_days": loss_days,
            "capital": self.capital,
            "end_balance": end_balance,
            "max_drawdown": max_drawdown,
            "max_ddpercent": max_ddpercent,
            "max_drawdown_duration": max_drawdown_duration,
            "total_net_pnl": total_net_pnl,
            "daily_net_pnl": daily_net_pnl,
            "total_commission": total_commission,
            "daily_commission": daily_commission,
            "total_slippage": total_slippage,
            "daily_slippage": daily_slippage,
            "total_turnover": total_turnover,
            "daily_turnover": daily_turnover,
            "total_trade_count": total_trade_count,
            "daily_trade_count": daily_trade_count,
            "total_return": total_return,
            "annual_return": annual_return,
            "daily_return": daily_return,
            "return_std": return_std,
            "sharpe_ratio": sharpe_ratio,
            "return_drawdown_ratio": return_drawdown_ratio,
        }

        # Filter potential error infinite value
        for key, value in statistics.items():
            if value in (np.inf, -np.inf):
                value = 0
            statistics[key] = np.nan_to_num(value)

        self.output("策略统计指标计算完成")
        return statistics

    def show_chart(self, df: DataFrame = None):
        """"""
        # Check DataFrame input exterior
        if df is None:
            df = self.daily_df

        # Check for init DataFrame
        if df is None:
            return

        plt.figure(figsize=(10, 16))

        balance_plot = plt.subplot(4, 1, 1)
        balance_plot.set_title("Balance")
        df["balance"].plot(legend=True)

        drawdown_plot = plt.subplot(4, 1, 2)
        drawdown_plot.set_title("Drawdown")
        drawdown_plot.fill_between(range(len(df)), df["drawdown"].values)

        pnl_plot = plt.subplot(4, 1, 3)
        pnl_plot.set_title("Daily Pnl")
        df["net_pnl"].plot(kind="bar", legend=False, grid=False, xticks=[])

        distribution_plot = plt.subplot(4, 1, 4)
        distribution_plot.set_title("Daily Pnl Distribution")
        df["net_pnl"].hist(bins=50)

        plt.show()

    def run_optimization(self, optimization_setting: OptimizationSetting, output=True):
        """"""
        # Get optimization setting and target
        settings = optimization_setting.generate_setting()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("优化参数组合为空，请检查")
            return

        if not target_name:
            self.output("优化目标未设置，请检查")
            return

        # Use multiprocessing pool for running backtesting with different setting
        # Force to use spawn method to create new process (instead of fork on Linux)
        ctx = multiprocessing.get_context("spawn")
        pool = ctx.Pool(multiprocessing.cpu_count())

        results = []
        for setting in settings:
            result = (pool.apply_async(optimize, (
                target_name,
                self.strategy_class,
                setting,
                self.vt_symbol,
                self.interval,
                self.start,
                self.rate,
                self.slippage,
                self.size,
                self.pricetick,
                self.capital,
                self.end,
                self.mode,
                self.inverse
            )))
            results.append(result)

        pool.close()
        pool.join()

        # Sort results and output
        result_values = [result.get() for result in results]
        result_values.sort(reverse=True, key=lambda result: result[1])

        if output:
            for value in result_values:
                msg = f"参数：{value[0]}, 目标：{value[1]}"
                self.output(msg)

        return result_values

    def run_ga_optimization(self, optimization_setting: OptimizationSetting, population_size=100, ngen_size=30, output=True):
        """"""
        # Get optimization setting and target
        settings = optimization_setting.generate_setting_ga()
        target_name = optimization_setting.target_name

        if not settings:
            self.output("优化参数组合为空，请检查")
            return

        if not target_name:
            self.output("优化目标未设置，请检查")
            return

        # Define parameter generation function
        def generate_parameter():
            """"""
            return random.choice(settings)

        def mutate_individual(individual, indpb):
            """"""
            size = len(individual)
            paramlist = generate_parameter()
            for i in range(size):
                if random.random() < indpb:
                    individual[i] = paramlist[i]
            return individual,

        # Create ga object function
        global ga_target_name
        global ga_strategy_class
        global ga_setting
        global ga_vt_symbol
        global ga_interval
        global ga_start
        global ga_rate
        global ga_slippage
        global ga_size
        global ga_pricetick
        global ga_capital
        global ga_end
        global ga_mode
        global ga_inverse

        ga_target_name = target_name
        ga_strategy_class = self.strategy_class
        ga_setting = settings[0]
        ga_vt_symbol = self.vt_symbol
        ga_interval = self.interval
        ga_start = self.start
        ga_rate = self.rate
        ga_slippage = self.slippage
        ga_size = self.size
        ga_pricetick = self.pricetick
        ga_capital = self.capital
        ga_end = self.end
        ga_mode = self.mode
        ga_inverse = self.inverse

        # Set up genetic algorithem
        toolbox = base.Toolbox()
        toolbox.register("individual", tools.initIterate, creator.Individual, generate_parameter)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", mutate_individual, indpb=1)
        toolbox.register("evaluate", ga_optimize)
        toolbox.register("select", tools.selNSGA2)

        total_size = len(settings)
        pop_size = population_size                      # number of individuals in each generation
        lambda_ = pop_size                              # number of children to produce at each generation
        mu = int(pop_size * 0.8)                        # number of individuals to select for the next generation

        cxpb = 0.95         # probability that an offspring is produced by crossover
        mutpb = 1 - cxpb    # probability that an offspring is produced by mutation
        ngen = ngen_size    # number of generation

        pop = toolbox.population(pop_size)
        hof = tools.ParetoFront()               # end result of pareto front

        stats = tools.Statistics(lambda ind: ind.fitness.values)
        np.set_printoptions(suppress=True)
        stats.register("mean", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        # Multiprocessing is not supported yet.
        # pool = multiprocessing.Pool(multiprocessing.cpu_count())
        # toolbox.register("map", pool.map)

        # Run ga optimization
        self.output(f"参数优化空间：{total_size}")
        self.output(f"每代族群总数：{pop_size}")
        self.output(f"优良筛选个数：{mu}")
        self.output(f"迭代次数：{ngen}")
        self.output(f"交叉概率：{cxpb:.0%}")
        self.output(f"突变概率：{mutpb:.0%}")

        start = time()

        algorithms.eaMuPlusLambda(
            pop,
            toolbox,
            mu,
            lambda_,
            cxpb,
            mutpb,
            ngen,
            stats,
            halloffame=hof
        )

        end = time()
        cost = int((end - start))

        self.output(f"遗传算法优化完成，耗时{cost}秒")

        # Return result list
        results = []

        for parameter_values in hof:
            setting = dict(parameter_values)
            target_value = ga_optimize(parameter_values)[0]
            results.append((setting, target_value, {}))

        return results

    def update_daily_close(self, price):
        """"""
        d = self.datetime.date()
        if self.strategy.get_name() == 'AllFutureStrategy' or self.strategy.get_name() == 'HedgeStrategy':
            daily_result = self.daily_results.get(d, None)
            if not daily_result:
                self.daily_results[d] = DayDailyResult(d, price)
        else:
            daily_result = self.daily_results.get(d, None)
            if daily_result:
                daily_result.close_price = price
            else:
                self.daily_results[d] = DailyResult(d, price)

    def get_current_bar(self, symbol):
        if type(self.bar) is dict:
            return self.bar.get(symbol, None)
        else:
            return self.bar

    def set_order_bar(self, bar):
        self.order_bar = bar

    def new_bar(self, data):
        """"""
        self.bar = data
        self.datetime = self.strategy.get_new_bar_datatime(data)

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_bar(data)

        self.update_daily_close(self.strategy.get_new_bar_price(data))

    def new_tick(self, tick: TickData):
        """"""
        self.tick = tick
        self.datetime = tick.datetime

        self.cross_limit_order()
        self.cross_stop_order()
        self.strategy.on_tick(tick)

        self.update_daily_close(tick.last_price)

    def cross_limit_order(self):
        """
        Cross limit order with last bar/tick data.
        """
        for order in list(self.active_limit_orders.values()):
            if self.mode == BacktestingMode.BAR:
                bar = self.get_current_bar(order.symbol)
                if not bar:
                    continue
                long_cross_price = bar.low_price
                short_cross_price = bar.high_price
                long_best_price = bar.open_price
                short_best_price = bar.open_price
            else:
                long_cross_price = self.tick.ask_price_1
                short_cross_price = self.tick.bid_price_1
                long_best_price = long_cross_price
                short_best_price = short_cross_price

            # Push order update with status "not traded" (pending).
            if order.status == Status.SUBMITTING:
                order.status = Status.NOTTRADED
                self.strategy.on_order(order)

            if order.direction == Direction.LONG:
                order.price = long_best_price
            elif order.direction == Direction.SHORT:
                order.price = short_best_price

            # Check whether limit orders can be filled.
            long_cross = (
                order.direction == Direction.LONG
                and order.price >= long_cross_price
                and long_cross_price > 0
            )

            short_cross = (
                order.direction == Direction.SHORT
                and order.price <= short_cross_price
                and short_cross_price > 0
            )

            if not long_cross and not short_cross:
                continue

            # Push order udpate with status "all traded" (filled).
            order.traded = order.volume
            order.status = Status.ALLTRADED
            self.strategy.on_order(order)

            self.active_limit_orders.pop(order.vt_orderid)

            # Push trade update
            self.trade_count += 1

            if long_cross:
                trade_price = min(order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = max(order.price, short_best_price)
                pos_change = -order.volume

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                time=self.datetime.strftime("%H:%M:%S"),
                gateway_name=self.gateway_name,
            )
            trade.datetime = self.datetime

            self.strategy.pos += pos_change
            if self.strategy.get_name() == 'AllFutureStrategy':
                self.strategy.add_symbol_pos(order.symbol, pos_change)
            self.strategy.on_trade(trade)

            self.trades[trade.vt_tradeid] = trade

    def cross_stop_order(self):
        """
        Cross stop order with last bar/tick data.
        """

        for stop_order in list(self.active_stop_orders.values()):
            bar = self.get_current_bar(stop_order.symbol)
            if self.mode == BacktestingMode.BAR:
                long_cross_price = bar.high_price
                short_cross_price = bar.low_price
                long_best_price = bar.open_price
                short_best_price = bar.open_price
            else:
                long_cross_price = self.tick.last_price
                short_cross_price = self.tick.last_price
                long_best_price = long_cross_price
                short_best_price = short_cross_price

            # Check whether stop order can be triggered.
            long_cross = (
                stop_order.direction == Direction.LONG
                and stop_order.price <= long_cross_price
            )

            short_cross = (
                stop_order.direction == Direction.SHORT
                and stop_order.price >= short_cross_price
            )

            if not long_cross and not short_cross:
                continue

            # Create order data.
            self.limit_order_count += 1

            order = OrderData(
                symbol=bar.symbol,
                exchange=bar.exchange,
                orderid=str(self.limit_order_count),
                direction=stop_order.direction,
                offset=stop_order.offset,
                price=stop_order.price,
                volume=stop_order.volume,
                status=Status.ALLTRADED,
                gateway_name=self.gateway_name,
            )
            order.datetime = self.datetime

            self.limit_orders[order.vt_orderid] = order

            # Create trade data.
            if long_cross:
                trade_price = max(stop_order.price, long_best_price)
                pos_change = order.volume
            else:
                trade_price = min(stop_order.price, short_best_price)
                pos_change = -order.volume

            self.trade_count += 1

            trade = TradeData(
                symbol=order.symbol,
                exchange=order.exchange,
                orderid=order.orderid,
                tradeid=str(self.trade_count),
                direction=order.direction,
                offset=order.offset,
                price=trade_price,
                volume=order.volume,
                time=self.datetime.strftime("%H:%M:%S"),
                gateway_name=self.gateway_name,
            )
            trade.datetime = self.datetime

            self.trades[trade.vt_tradeid] = trade

            # Update stop order.
            stop_order.vt_orderids.append(order.vt_orderid)
            stop_order.status = StopOrderStatus.TRIGGERED

            if stop_order.stop_orderid in self.active_stop_orders:
                self.active_stop_orders.pop(stop_order.stop_orderid)

            # Push update to strategy.
            self.strategy.on_stop_order(stop_order)
            self.strategy.on_order(order)

            self.strategy.pos += pos_change
            self.strategy.on_trade(trade)

    def load_bar(
        self,
        vt_symbol: str,
        days: int,
        interval: Interval,
        callback: Callable,
        use_database: bool
    ):
        """"""
        self.days = days
        self.callback = callback

    def load_tick(self, vt_symbol: str, days: int, callback: Callable):
        """"""
        self.days = days
        self.callback = callback

    def send_order(
        self,
        strategy: CtaTemplate,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float,
        stop: bool,
        lock: bool
    ):
        """"""
        price = round_to(price, self.pricetick)
        if stop:
            vt_orderid = self.send_stop_order(direction, offset, price, volume)
        else:
            vt_orderid = self.send_limit_order(direction, offset, price, volume)
        return [vt_orderid]

    def send_stop_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ):
        """"""
        self.stop_order_count += 1

        stop_order = StopOrder(
            vt_symbol=self.vt_symbol,
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            stop_orderid=f"{STOPORDER_PREFIX}.{self.stop_order_count}",
            strategy_name=self.strategy.strategy_name,
        )

        self.active_stop_orders[stop_order.stop_orderid] = stop_order
        self.stop_orders[stop_order.stop_orderid] = stop_order

        return stop_order.stop_orderid

    def send_limit_order(
        self,
        direction: Direction,
        offset: Offset,
        price: float,
        volume: float
    ):
        """"""
        self.limit_order_count += 1

        order = OrderData(
            symbol=self.order_bar.symbol,
            exchange=self.order_bar.exchange,
            orderid=str(self.limit_order_count),
            direction=direction,
            offset=offset,
            price=price,
            volume=volume,
            status=Status.SUBMITTING,
            gateway_name=self.gateway_name,
        )
        order.datetime = self.order_bar.datetime

        self.active_limit_orders[order.vt_orderid] = order
        self.limit_orders[order.vt_orderid] = order

        return order.vt_orderid

    def cancel_order(self, strategy: CtaTemplate, vt_orderid: str):
        """
        Cancel order by vt_orderid.
        """
        if vt_orderid.startswith(STOPORDER_PREFIX):
            self.cancel_stop_order(strategy, vt_orderid)
        else:
            self.cancel_limit_order(strategy, vt_orderid)

    def cancel_stop_order(self, strategy: CtaTemplate, vt_orderid: str):
        """"""
        if vt_orderid not in self.active_stop_orders:
            return
        stop_order = self.active_stop_orders.pop(vt_orderid)

        stop_order.status = StopOrderStatus.CANCELLED
        self.strategy.on_stop_order(stop_order)

    def cancel_limit_order(self, strategy: CtaTemplate, vt_orderid: str):
        """"""
        if vt_orderid not in self.active_limit_orders:
            return
        order = self.active_limit_orders.pop(vt_orderid)

        order.status = Status.CANCELLED
        self.strategy.on_order(order)

    def cancel_all(self, strategy: CtaTemplate):
        """
        Cancel all orders, both limit and stop.
        """
        vt_orderids = list(self.active_limit_orders.keys())
        for vt_orderid in vt_orderids:
            self.cancel_limit_order(strategy, vt_orderid)

        stop_orderids = list(self.active_stop_orders.keys())
        for vt_orderid in stop_orderids:
            self.cancel_stop_order(strategy, vt_orderid)

    def write_log(self, msg: str, strategy: CtaTemplate = None):
        """
        Write log message.
        """
        msg = f"{self.datetime}\t{msg}"
        self.logs.append(msg)

    def send_email(self, msg: str, strategy: CtaTemplate = None):
        """
        Send email to default receiver.
        """
        pass

    def sync_strategy_data(self, strategy: CtaTemplate):
        """
        Sync strategy data into json file.
        """
        pass

    def get_engine_type(self):
        """
        Return engine type.
        """
        return self.engine_type

    def put_strategy_event(self, strategy: CtaTemplate):
        """
        Put an event to update strategy status.
        """
        pass

    def output(self, msg):
        """
        Output message of backtesting engine.
        """
        print(f"{datetime.now()}\t{msg}")

    def get_all_trades(self):
        """
        Return all trade data of current backtesting result.
        """
        return list(self.trades.values())

    def get_all_orders(self):
        """
        Return all limit order data of current backtesting result.
        """
        return list(self.limit_orders.values())

    def get_all_daily_results(self):
        """
        Return all daily result data.
        """
        return list(self.daily_results.values())


class DayDailyResult:
    """"""

    def __init__(self, date: date, price: dict):
        """"""
        self.close_price = price
        self.date = date

        self.trades = []
        self.trade_count = 0

        self.start_pos = {}
        self.end_pos = {}

        self.turnover = 0
        self.commission = 0
        self.slippage = 0

        self.trading_pnl = 0
        self.holding_pnl = 0
        self.total_pnl = 0
        self.net_pnl = 0

    def add_trade(self, trade: TradeData):
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: dict,
        start_pos: dict,
        size: int,
        rate: float,
        slippage: float,
        inverse: bool
    ):
        """"""
        if pre_close is None:
            pre_close = {}
        if start_pos is None:
            start_pos = {}
        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        # Holding pnl is the pnl from holding position at day start
        for symbol, pos in start_pos.items():
            flag = get_symbol_flag(symbol)
            param = Future_Params[flag]
            # If no pre_close provided on the first day,
            # use value 1 to avoid zero division error
            pre = pre_close.get(symbol, 1)
            close = self.close_price.get(symbol, 1)
            size = param.get('size', size)
            if not inverse:  # For normal contract
                self.holding_pnl += pos * (close - pre) * size
            else:  # For crypto currency inverse contract
                self.holding_pnl += pos * (1 / pre - 1 / close) * size

        for trade in self.trades:
            flag = get_symbol_flag(trade.symbol)
            param = Future_Params[flag]
            close = self.close_price.get(trade.symbol, 1)
            size = param.get('size', size)
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos[trade.symbol] = self.end_pos.get(trade.symbol, 0) + pos_change

            # For normal contract
            if not inverse:
                turnover = trade.volume * size * trade.price
                self.trading_pnl += pos_change * (close - trade.price) * size
                self.slippage += trade.volume * size * param['slip'] * param['slip_loss']
            # For crypto currency inverse contract
            else:
                turnover = trade.volume * size / trade.price
                self.trading_pnl += pos_change * (1 / trade.price - 1 / close) * size
                self.slippage += trade.volume * size * param['slip'] * param['slip_loss'] / (trade.price ** 2)

            self.turnover += turnover
            if 'rate' in param:
                self.commission += turnover * param['rate'] * 1.05
            elif 'rate_point' in param:
                self.commission += pos_change * param['rate_point'] * 1.05


        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage


class DailyResult:
    """"""

    def __init__(self, date: date, close_price):
        """"""
        self.date = date
        self.close_price = close_price
        self.pre_close = 0

        self.trades = []
        self.trade_count = 0

        self.start_pos = 0
        self.end_pos = 0

        self.turnover = 0
        self.commission = 0
        self.slippage = 0

        self.trading_pnl = 0
        self.holding_pnl = 0
        self.total_pnl = 0
        self.net_pnl = 0

    def add_trade(self, trade: TradeData):
        """"""
        self.trades.append(trade)

    def calculate_pnl(
        self,
        pre_close: float,
        start_pos: float,
        size: int,
        rate: float,
        slippage: float,
        inverse: bool
    ):
        """"""
        # If no pre_close provided on the first day,
        # use value 1 to avoid zero division error
        if pre_close is None:
            pre_close = 0
        if start_pos is None:
            start_pos = 0

        if pre_close:
            self.pre_close = pre_close
        else:
            self.pre_close = 1

        # Holding pnl is the pnl from holding position at day start
        self.start_pos = start_pos
        self.end_pos = start_pos

        if not inverse:     # For normal contract
            self.holding_pnl = self.start_pos * \
                (self.close_price - self.pre_close) * size
        else:               # For crypto currency inverse contract
            self.holding_pnl = self.start_pos * \
                (1 / self.pre_close - 1 / self.close_price) * size

        # Trading pnl is the pnl from new trade during the day
        self.trade_count = len(self.trades)

        for trade in self.trades:
            if trade.direction == Direction.LONG:
                pos_change = trade.volume
            else:
                pos_change = -trade.volume

            self.end_pos += pos_change

            flag = get_symbol_flag(trade.symbol)
            param = Future_Params.get(flag, None)
            # For normal contract
            if not inverse:
                turnover = trade.volume * size * trade.price
                self.trading_pnl += pos_change * \
                    (self.close_price - trade.price) * size
                if param:
                    self.slippage += trade.volume * size * param['slip'] * param['slip_loss']
                else:
                    self.slippage += trade.volume * size * slippage
            # For crypto currency inverse contract
            else:
                turnover = trade.volume * size / trade.price
                self.trading_pnl += pos_change * \
                    (1 / trade.price - 1 / self.close_price) * size
                if param:
                    self.slippage += trade.volume * size * param['slip'] * param['slip_loss']
                else:
                    self.slippage += trade.volume * size * slippage / (trade.price ** 2)

            self.turnover += turnover
            if param:
                if 'rate' in param:
                    self.commission += turnover * param['rate'] * 1.05
                elif 'rate_point' in param:
                    self.commission += pos_change * param['rate_point'] * 1.05
            else:
                self.commission += turnover * rate

        # Net pnl takes account of commission and slippage cost
        self.total_pnl = self.trading_pnl + self.holding_pnl
        self.net_pnl = self.total_pnl - self.commission - self.slippage


def optimize(
    target_name: str,
    strategy_class: CtaTemplate,
    setting: dict,
    vt_symbol: str,
    interval: Interval,
    start: datetime,
    rate: float,
    slippage: float,
    size: float,
    pricetick: float,
    capital: int,
    end: datetime,
    mode: BacktestingMode,
    inverse: bool
):
    """
    Function for running in multiprocessing.pool
    """
    engine = BacktestingEngine()

    engine.set_parameters(
        vt_symbol=vt_symbol,
        interval=interval,
        start=start,
        rate=rate,
        slippage=slippage,
        size=size,
        pricetick=pricetick,
        capital=capital,
        end=end,
        mode=mode,
        inverse=inverse
    )

    engine.add_strategy(strategy_class, setting)
    engine.load_data()
    engine.run_backtesting()
    engine.calculate_result()
    statistics = engine.calculate_statistics(output=False)

    target_value = statistics[target_name]
    return (str(setting), target_value, statistics)


@lru_cache(maxsize=1000000)
def _ga_optimize(parameter_values: tuple):
    """"""
    setting = dict(parameter_values)

    result = optimize(
        ga_target_name,
        ga_strategy_class,
        setting,
        ga_vt_symbol,
        ga_interval,
        ga_start,
        ga_rate,
        ga_slippage,
        ga_size,
        ga_pricetick,
        ga_capital,
        ga_end,
        ga_mode,
        ga_inverse
    )
    return (result[1],)


def ga_optimize(parameter_values: list):
    """"""
    return _ga_optimize(tuple(parameter_values))


@lru_cache(maxsize=999)
def load_bar_data(
    symbol: str,
    exchange: Exchange,
    interval: Interval,
    start: datetime,
    end: datetime,
    like_symbol=None
):
    """"""
    return database_manager.load_bar_data(
        symbol, exchange, interval, start, end, like_symbol
    )


@lru_cache(maxsize=999)
def load_tick_data(
    symbol: str,
    exchange: Exchange,
    start: datetime,
    end: datetime
):
    """"""
    return database_manager.load_tick_data(
        symbol, exchange, start, end
    )


# GA related global value
ga_end = None
ga_mode = None
ga_target_name = None
ga_strategy_class = None
ga_setting = None
ga_vt_symbol = None
ga_interval = None
ga_start = None
ga_rate = None
ga_slippage = None
ga_size = None
ga_pricetick = None
ga_capital = None
