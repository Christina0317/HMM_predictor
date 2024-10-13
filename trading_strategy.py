import pandas as pd
import numpy as np


class TradingStrategy:
    """
    Trading strategy based on HMM predictions
    : param probs: dataframe, index -> date, columns -> states
    : param prices: dataframe, index -> date, columns -> [price, diff, observation]
    : param signals: dict, key -> date, value -> signal
    """
    def __init__(self, prices, initial_cash, n_window, dates):
        self.prices = prices   # Price sequences per unit of the asset
        self.dates = dates
        self.signals = None

        self.initial_cash = initial_cash  # Initial amount of cash available for trading
        self.n_window = n_window  # Number of days to look back for trading

    def value_to_prob(self, value_seq, feature_name):
        if value_seq.shape[1] == 1:
            if feature_name == 'price':
                # 检查数组是否至少有两行
                if value_seq.shape[0] < 2:
                    raise ValueError("Array must have at least two rows for comparison.")

                result = np.zeros((value_seq.shape[0], 3), dtype=int)
                for i in range(1, value_seq.shape[0]):
                    if np.all(value_seq[i] > value_seq[i - 1]):
                        result[i] = (1, 0, 0)
                    elif np.all(value_seq[i] < value_seq[i - 1]):
                        result[i] = (0, 0, 1)
                    else:
                        result[i] = (0, 1, 0)
                return pd.DataFrame(result, index=self.dates)

            elif feature_name == 'return':
                result = np.zeros((value_seq.shape[0], 3), dtype=int)
                for i in range(value_seq.shape[0]):
                    if value_seq[i] > 0:
                        result[i] = (1, 0, 0)
                    elif value_seq[i] < 0:
                        result[i] = (0, 0, 1)
                    else:
                        result[i] = (0, 1, 0)
                return pd.DataFrame(result, index=self.dates)
            else:
                print(f'feature_name {feature_name} is not supported')

        else:
            return pd.DataFrame(value_seq, index=self.dates)

    def daily_signal(self, prob):
        """
        Generate a buying signal based on the probability of the next state tomorrow
        :param prob: array, shape (n_symbols, )
        :return: int, 1 for buying, -1 for selling, 0 for no change.
        """
        if prob[0] > 0.6:
            return 1
        elif prob[2] > 0.6:
            return -1
        else:
            return 0

    def generate_signals(self, probs):
        """
        Generate signals via the observations predicted by the HMM
        :return: list, a sequence of signals -> shape (len(probs), )
        """
        self.prices['signal'] = np.nan * len(self.prices)
        for i in range(len(self.dates) - 1):
            today = self.dates[i]
            next_day = self.dates[i + 1]
            prob = probs.loc[next_day].values
            # 生成每日交易信号
            signal = self.daily_signal(prob)
            self.prices.loc[today, 'signal'] = signal
        self.signals = self.prices['signal']
        return self.prices[['signal']]

    def trading_return(self):
        """
        Simulate trading based on trading signals.
        :return: Change of cash, units, total value -> shape (len(probs), )
        """
        cash = self.initial_cash
        units = 0
        last_action = 0  # Track the last action to prevent consecutive buys without sells

        trading_record = pd.DataFrame(data=None, index=self.dates, columns=['price', 'signal', 'cash', 'units', 'value'])
        trading_record['price'] = self.prices['price']
        trading_record['signal'] = self.signals

        for i in range(len(self.dates)):
            date = self.dates[i]
            signal = self.signals.loc[date]
            price = self.prices.loc[date, 'price']
            if signal == 1 and last_action != 1:
                max_units_can_buy = int(cash / price)  # Buy if signal is +1 and last action was not a buy
                if max_units_can_buy > 0:
                    units += max_units_can_buy
                    cash -= max_units_can_buy * price
                    last_action = 1
                # else:
                    # print("Not enough cash to buy more units.")
            elif signal == -1:  # Sell if signal is -1
                if units > 0:
                    cash += price * units
                    units = 0
                    last_action = -1
                # else:
                    # print("No units to sell.")

            else:
                last_action = 0  # Hold or no action

            # Calculate total value
            total_value = cash + units * price
            # Record the state after each transaction
            trading_record.loc[date, 'cash'] = cash
            trading_record.loc[date, 'units'] = units
            trading_record.loc[date, 'value'] = total_value

        return trading_record

    def calculate_accuracy_rate(self, trading_record):
        """
        计算预测上涨、下跌的准确率
        :param trading_record: dataframe -> 交易记录
        :return:
        """
        predicted_signal = trading_record['signal']
        actual_signal = trading_record['price'].shift(1) - trading_record['price']

        def cal_actual_signal(row):
            if row > 0:
                return 1
            elif row == 0:
                return 0
            else:
                return -1

        actual_signal = actual_signal.apply(cal_actual_signal)

        predicted_signal = predicted_signal.values
        actual_signal = actual_signal.values
        predicted_signal = predicted_signal[self.n_window:]
        actual_signal = actual_signal[self.n_window:]

        matching_elements = np.sum(predicted_signal == actual_signal)
        total_elements = predicted_signal.size

        return matching_elements / total_elements

    def calculate_annualized_return(self, total_value):
        """ 计算年化收益率 """
        total_value = total_value.iloc[self.n_window:]
        dates = pd.to_datetime(total_value.index)
        days = (dates[-1] - dates[0]).days
        years = days / 365.25
        total_return = total_value.iloc[-1] / total_value.iloc[0] - 1
        annualized_return = (1 + total_return) ** (1 / years) - 1
        return annualized_return

    def calculate_max_drawdown(self, daily_return):
        """ 计算最大回测 """
        cumulative_return = (1 + daily_return).cumprod()

        # 计算滚动最大值和最大回撤
        rolling_max = cumulative_return.cummax()
        daily_drawdown = cumulative_return / rolling_max - 1
        max_drawdown = daily_drawdown.min()
        return -max_drawdown

    def calculate_sharp_ratio(self, daily_return):
        # 假设无风险日利率
        risk_free_rate = 0.03 / 252  # 将年化利率转换为日利率
        excess_returns = daily_return - risk_free_rate

        # 计算夏普比率
        sharpe_ratio = np.sqrt(252) * excess_returns.mean() / excess_returns.std()
        return sharpe_ratio

    def calculate_win_ratio(self, daily_return):
        # 计算胜利的日子
        win_days = daily_return > 0
        win_rate = win_days.mean()
        return win_rate

