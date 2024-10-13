import numpy as np
import pandas as pd


class DataProcess:
    def __init__(self, dates, code, feature):
        self.dates = dates
        self.code = code
        self.feature = feature

    def data_process(self, data):
        values = []
        for date in self.dates:
            if self.code in data[date].index:
                values.append(data[date].loc[self.code, self.feature])
            else:
                values.append(np.nan)
        return pd.DataFrame(values, index=self.dates, columns=['price'])

    def transform_observation(self, df):
        """
        将价格数据, 转换为离散观测序列, 即价格上涨为0, 下跌为2, 持平为1
        :param df: dataframe
        :return: arr, diff -> shape (len(dates) - 1, )
        """
        # 去除第一个值, 因为第一个值是nan
        df['diff'] = df.price.shift(1) - df.price
        conditions = [df['diff'] > 0, df['diff'] == 0, df['diff'] < 0]

        choices = [0, 1, 2]  # 上涨为0，持平为1，下跌为2

        df['observation'] = np.select(conditions, choices, default=np.nan)

        # 删掉有nan的一行
        df = df.dropna()
        # 转换为int
        df['observation'] = df['observation'].astype(int)

        return df[['price', 'observation']]

    def cal_daily_return(self, df):
        df['return'] = np.log(df['price'] / df['price'].shift(1))
        return df