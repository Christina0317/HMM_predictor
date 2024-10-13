import numpy as np
from tqdm import tqdm
from config import data_type
from multiprocessing import Pool

if data_type == 'discrete':
    from model import DiscreteHMM
if data_type == 'continuous':
    from model import GaussianHMM


def continuous_predicted_value(next_states):
    if isinstance(next_states, np.ndarray):
        next_value = next_states
    else:
        next_value = next_states
        # if feature_name == 'price':
        #     if predicted[-1] > predicted[-2]:
        #         obs_prob_predicted.append([1, 0, 0])
        #     elif predicted[-1] < predicted[-2]:
        #         obs_prob_predicted.append([0, 0, 1])
        #     else:
        #         obs_prob_predicted.append([0, 1, 0])
        # elif feature_name == 'return':
        #     if predicted[-1] > 0:
        #         obs_prob_predicted.append([1, 0, 0])
        #     elif predicted[-1] < 0:
        #         obs_prob_predicted.append([0, 0, 1])
        #     else:
        #         obs_prob_predicted.append([0, 1, 0])
        # else:
        #     print(f'feature_name {feature_name} is not supported')
    return next_value


def viterbi_method(data, n_states, n_symbols, n_iter, n_window, em_threshold, feature_name):
    """
    用viterbi算法预测, 可以预测离散数据(DiscreteHMM)或者连续数据(GaussianHMM)
    :param data: series, index -> dates, 股票价格或者是涨跌幅都可以
    :return: obs_prob_predicted -> list, shape (len(data), n_symbols)
    :return: price_predicted -> list, shape (len(data), )
    """
    # 添加外层进度条
    pbar = tqdm(range(n_window, len(data.index.values)), desc="Processing windows")

    # 初始化hmm模型
    if data_type == 'discrete':
        hmm_model = DiscreteHMM(n_states, n_symbols, n_iter, n_window, em_threshold)
        # 预测的观测序列 -> shape (len(dates), n_symbols)
        next_value_seq = [[np.nan] * n_symbols] * (n_window)
        for i in pbar:
            X = data.iloc[i - n_window: i].values
            X = X[~np.isnan(X)]
            if len(X) == 0:
                next_value_seq.append([0, 1, 0])
            else:
                hmm_model.reset_parameters(X)
                transmat, mean, var = hmm_model.em_algorithm(X)
                next_states_prob = hmm_model.viterbi_predict(X)
                next_value_seq.append(next_states_prob)

    else:
        hmm_model = GaussianHMM(n_states, n_symbols, n_iter, n_window, em_threshold)
        # 价格预测序列 -> shape (len(dates), )
        next_value_seq = [np.nan] * (n_window)
        for i in pbar:
            X = data.iloc[i - n_window: i].values
            X = X[~np.isnan(X)]
            if len(X) == 0:
                next_value_seq.append(np.nan)
            else:
                hmm_model.reset_parameters(X)
                transmat, mean, var = hmm_model.em_algorithm(X)
                next_states_prob = hmm_model.viterbi_predict(X)
                next_value_seq.append(next_states_prob)

    return next_value_seq


def process_single_window(args):
    index, window_data, hmm_model, n_window, data = args
    # window_data = window_data[~np.isnan(window_data)]
    if index < n_window + 20:
        hmm_model.reset_parameters(window_data)
        transmat, mean, var = hmm_model.em_algorithm(window_data)
        return hmm_model.viterbi_predict(window_data)
    else:
        hmm_model.reset_parameters(window_data)
        transmat, mean, var = hmm_model.em_algorithm(window_data)
        return hmm_model.comparison_predict(window_data, data[max(0, index - 300):index - 1].values, 1e-3)


def comparison_method(data, n_states, n_symbols, n_iter, n_window, em_threshold, feature_name):
    """
    用comparison算法预测, 只可以预测连续数据
    :param data: series, index -> dates, 股票价格的涨跌幅
    :param feature_name: str, 'return' or 'price', 预测的股票价格还是涨跌幅
    :return: obs_prob_predicted -> list, shape (len(data), n_symbols)
    :return: price_predicted -> list, shape (len(data), )
    :return:
    """
    # 初始化hmm模型
    hmm_model = GaussianHMM(n_states, n_symbols, n_iter, n_window, em_threshold)

    # next_value_seq = [np.nan] * (n_window)
    #
    # # 添加外层进度条
    # pbar = tqdm(range(n_window, len(data.index.values)), desc="Processing windows")
    #
    # for i in pbar:
    #     X = data.iloc[i - n_window: i].values
    #     if i < n_window + 20:
    #         hmm_model.reset_parameters(X)
    #         transmat, mean, var = hmm_model.em_algorithm(X)
    #         next_value = hmm_model.viterbi_predict(X)
    #         next_value_seq.append(next_value)
    #     else:
    #         hmm_model.reset_parameters(X)
    #         transmat, mean, var = hmm_model.em_algorithm(X)
    #         o_t = hmm_model.comparison_predict(X, data.iloc[max(0, i-300):i - 1].values, 1e-3)
    #         next_value_seq.append(o_t)

    tasks = [(i, data.iloc[i - n_window: i].values, hmm_model, n_window, data) for i in range(n_window, len(data.index.values))]

    # Using a pool of workers to process data windows in parallel
    with Pool(processes=4) as pool:  # You can adjust the number of processes based on your CPU
        results = list(tqdm(pool.imap(process_single_window, tasks), total=len(tasks)))

    # Initialize results list with NaNs for the initial window
    next_value_seq = [np.nan] * n_window
    next_value_seq.extend(results)

    return next_value_seq
