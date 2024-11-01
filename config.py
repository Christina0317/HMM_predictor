path = '/Volumes/E/quant_data/db_jq_candle_daily.pkl'
start_date = '20220101'
end_date = '20230512'
stock_code = '600877.XSHG'
initial_cash = 10000
n_hidden_states = 8
n_symbols = 3
n_iter = 1000
n_window = 30
em_threshold = 1e-3
predict_method = 'comparison'  # 'viterbi' or 'comparison'
feature_name = 'return'   # 'observation', 'price', 'return'
data_type = 'continuous'   # 'continuous' or 'discrete'


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
