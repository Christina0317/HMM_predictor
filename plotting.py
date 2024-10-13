import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_trading(df, initial_cash):
    """
    画出交易过程的个股价格走势、资产变化趋势、买卖点
    :param df: dataframe -> index: datetime, columns: ['price', 'signal', 'cash', 'units', 'value']
    :return:
    """
    df.index = pd.to_datetime(df.index)

    fig, ax1 = plt.subplots(figsize=(14, 7))

    # 价格走势
    color = 'tab:blue'
    ax1.set_xlabel('date')
    ax1.set_ylabel('price', color=color)
    ax1.plot(df.index, initial_cash / df['price'].iloc[0] * df['price'], label='stock price', color=color)
    ax1.tick_params(axis='y', labelcolor=color)

    # 标记买卖点
    buys = df[df['signal'] == 1]
    sells = df[df['signal'] == -1]
    ax1.plot(buys.index, initial_cash / df['price'].iloc[0] * df['price'][buys.index], '^', markersize=10, color='g', label='Buy Signal', alpha=0.75)
    ax1.plot(sells.index, initial_cash / df['price'].iloc[0] * df['price'][sells.index], 'v', markersize=10, color='r', label='Sell Signal', alpha=0.75)

    # 资产变化趋势
    ax2 = ax1.twinx()  # 实例化一个新的轴，共享相同的x轴
    color = 'tab:red'
    ax2.set_ylabel('Total Value', color=color)
    ax2.plot(df.index, df['value'], label='Total Asset Value', color=color)
    ax2.tick_params(axis='y', labelcolor=color)

    points = (initial_cash / df['price'].iloc[0] * df['price']).values
    points = np.append(points, df['value'].values)
    ymin, ymax = points.min(), points.max()
    ax1.set_ylim(ymin, ymax)
    ax2.set_ylim(ymin, ymax)

    # 图例
    fig.tight_layout()  # 调整布局
    ax1.legend(loc='upper left')
    ax2.legend(loc='upper right')
    plt.title('Trading Process Overview')
    plt.show()

    return



