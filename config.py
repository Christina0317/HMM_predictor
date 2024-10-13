path = '/Volumes/E/quant_internship_sgd/data/db_jq_candle_daily.pkl'
start_date = '20230101'
end_date = '20230512'
stock_code = '600877.XSHG'
initial_cash = 10000
n_hidden_states = 6
n_symbols = 3
n_iter = 1000
n_window = 20
em_threshold = 1e-3
predict_method = 'viterbi'  # 'viterbi' or 'comparison'
feature_name = 'observation'   # 'observation', 'price', 'return'
data_type = 'discrete'   # 'continuous' or 'discrete'


def error_check():
    if data_type == 'continuous' and feature_name == 'observation':
        print('The continuous data cannot use the observation column')
        return False
    if feature_name == 'observation' and predict_method == 'comparison':
        print('The observation column cannot use the comparison method')
        return False
    if data_type == 'discrete' and predict_method == 'comparison':
        print('The discrete data cannot use the comparison method')
        return False
    if data_type == 'discrete' and feature_name != 'observation':
        print('The discrete data can only use the observation column')
        return False
    return True
