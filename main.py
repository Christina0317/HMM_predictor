from preprocess import DataProcess
from trading_strategy import TradingStrategy
import plotting
import numpy as np
import pandas as pd
import pickle
from tqdm import tqdm
from hmm_model import viterbi_method, comparison_method
from config import *
import random
random.seed(42)
np.random.seed(42)


def implement_hmm(data, n_states, n_symbols, n_iter, n_window, em_threshold, predict_method, feature_name):
    """
    hmm 模型进行预测
    :param data: 价格 / 观测 数据
    :param n_states: 隐藏态数量 -> int
    :param n_symbols: 观测态数量 -> int
    :param n_iter: 训练次数 -> int
    :param n_window: 用过去多少天的数据训练 -> int
    :param em_threshold: 收敛阈值 -> float
    :return: df_obs 预测的观测概率 -> dataframe (len(dates), n_symbols)
    :return: df_price 预测的股票价格 -> dataframe (len(dates), )
    """
    dates = data.index.values

    if predict_method == 'viterbi':
        predicted_seq = viterbi_method(data, n_states, n_symbols, n_iter, n_window, em_threshold, feature_name)

    if predict_method == 'comparison':
        predicted_seq = comparison_method(data, n_states, n_symbols, n_iter, n_window, em_threshold, feature_name)

    df_predicted = pd.DataFrame(predicted_seq, index=dates)
    # df[date] 所对应的是过去 n_window 天模型训练得到的 date 这一天观测概率的预测结果

    return df_predicted


def implement_random(dates, n_symbols):
    from model import RandomModel
    random_model = RandomModel(n_symbols, dates)
    obs_prob_predicted = random_model.generate_observation_state()
    df_obs = pd.DataFrame(obs_prob_predicted, columns=[f'Symbol_{j}' for j in range(n_symbols)], index=dates)
    return df_obs


def implement_trading(predicted_value, stock_data, initial_cash, n_window, feature_name):
    """
    根据预测观测态的概率进行交易
    :param predicted_value: 观测态概率 or 预测的价格/收益率, dataframe
    :param stock_data: dataframe, index -> date, columns -> [price, observation]
    :param initial_cash: int
    :param n_window: int
    :return:
    """
    # 初始化交易模型
    strategy = TradingStrategy(stock_data, initial_cash, n_window, predicted_value.index.values)
    probs = strategy.value_to_prob(predicted_value.values, feature_name)
    signals = strategy.generate_signals(probs)
    trading_record = strategy.trading_return()

    daily_return = trading_record['value'] / trading_record['value'].shift(1) - 1

    # 模型准确率
    accuracy_rate = strategy.calculate_accuracy_rate(trading_record)

    # 年化收益率
    annulized_return = strategy.calculate_annualized_return(trading_record['value'])

    # 最大回撤
    maximum_drawdown = strategy.calculate_max_drawdown(daily_return)

    # 夏普比率
    sharpe_ratio = strategy.calculate_sharp_ratio(daily_return)

    # 胜率
    win_ratio = strategy.calculate_win_ratio(daily_return)

    strategy_index = {'accuracy_rate': accuracy_rate, 'annulized_return': annulized_return, 'maximum_drawdown': maximum_drawdown,
                      'sharpe_ratio': sharpe_ratio, 'win_ratio': win_ratio}
    strategy_index = pd.DataFrame(strategy_index, index=[0])
    print(strategy_index)

    return trading_record, strategy_index


if __name__ == '__main__':
    with open(path, 'rb') as f:
        data = pickle.load(f)

    dates = [date for date in data.keys() if start_date <= date <= end_date]

    # 观测序列 -> shape (len(dates), )
    processor = DataProcess(dates, stock_code, 'close')
    # 获取 feature 这一列的数据
    stock_data = processor.data_process(data)
    stock_data = processor.transform_observation(stock_data)
    stock_data = processor.cal_daily_return(stock_data)

    # stock_data: columns -> [price, observation, return]

    if error_check():
        # 用hmm模型进行训练和预测
        df_predicted = implement_hmm(stock_data[feature_name], n_hidden_states, n_symbols, n_iter, n_window, em_threshold, predict_method, feature_name)

        # 随机给定预测的观测序列
        df_obs_random = implement_random(stock_data.index.values, n_symbols)

        trading_record, strategy_index = implement_trading(df_predicted, stock_data, initial_cash, n_window, feature_name)
        trading_record_1, strategy_index_1 = implement_trading(df_obs_random, stock_data, initial_cash, n_window, feature_name)

        plotting.plot_trading(trading_record, initial_cash)
        plotting.plot_trading(trading_record_1, initial_cash)
