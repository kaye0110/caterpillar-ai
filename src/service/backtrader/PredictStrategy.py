from datetime import datetime

import backtrader as bt
import numpy as np
import pandas as pd
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression

from src.service.OverseeService import OverseeService
from src.service.StockService import StockService
from src.service.processor.ProcessorBuilder import ProcessorBuilder


class PredictStrategy(bt.Strategy):
    params = (
        ('atr_period', 14),
        ('atr_multiplier', 1.5),
        ('short_period', 10),
        ('long_period', 30),
        ('rsi_period', 14),
        ('rsi_upper', 70),
        ('rsi_lower', 30),
        ('stop_loss', 0.02),  # 2% stop loss
        ('take_profit', 0.05),  # 5% take profit
    )

    def __init__(self):
        self.dataclose = self.datas[0].close
        self.order = None
        self.atr = bt.indicators.ATR(self.datas[0], period=self.params.atr_period)

        # Track the order
        self.order = None

        self.buy_signals = []
        self.sell_signals = []
        self.predicted_lines = []
        self.batch_code = "20250207001_600010.SH"
        self.processor = ProcessorBuilder.build_by_batch_code(batch_code=self.batch_code)

        self.oversee = OverseeService(batch_code=self.batch_code)
        self.oversee.set_single_stock_by_code("600010.SH")
        self.data_with_label = self.oversee.load_data_with_label(start_date='20240601', end_date='20250207')
        self.data_with_label['trade_date_copy'] = pd.to_datetime(self.data_with_label['trade_date'], format='%Y%m%d')

        # Initialize moving averages
        self.sma_short = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.short_period)
        self.sma_long = bt.indicators.SimpleMovingAverage(self.data.close, period=self.params.long_period)

        # Initialize RSI
        self.rsi = bt.indicators.RSI(self.data.close, period=self.params.rsi_period)

    def _get_before_data(self):
        current_date = self.data.datetime.date(0).strftime("%Y%m%d")
        # 找到与当前日期匹配的行，并获取其index
        matching_index = self.data_with_label.index[self.data_with_label['trade_date'] == current_date]
        if matching_index.empty:
            return None

        # 获取匹配行的索引值
        idx = matching_index[0]

        # 确保我们不会尝试访问超出DataFrame范围的行
        if idx < self.processor.config.n_timestep:
            return None

        # 使用 iloc 方法从前40行中选取数据
        return self.data_with_label.iloc[idx - self.processor.config.n_timestep:idx]

    def next(self):
        if self.data.datetime.date(0) < datetime(2025, 2, 4).date():
            return

        if len(self.dataclose) < self.processor.config.n_timestep:
            return

        if self.order:
            return

        current_price = self.dataclose[0]

        # 获取过去40天的收盘价
        close_array = self._get_before_data()
        if close_array is None:
            return
        predicted_prices = self.processor.predict(close_array).future_predict_value[0]

        self.predicted_lines.append((self.data.datetime.date(0), predicted_prices))

        # Calculate the trend using linear regression
        X = np.arange(len(predicted_prices)).reshape(-1, 1)
        y = np.array(predicted_prices).reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        predicted_trend_slope = model.coef_[0][0]

        # Calculate volatility as the standard deviation of predicted prices
        predicted_volatility = np.std(predicted_prices)

        # Calculate future returns
        future_returns = [predicted_prices[i] / predicted_prices[0] - 1 for i in [1, 2, 3, 4, 5, 7, 8]]

        # Check if we are in the market
        if not self.position:

            if future_returns[0] > 0.05 and future_returns[1] > 0.05:
                self.order = self.buy()
                self.buy_signals.append((self.data.datetime.date(0), current_price))
                self.predicted_lines.append((self.data.datetime.date(0), predicted_prices))
        else:
            if future_returns[0] < 0 or future_returns[1] < 0:
                self.order = self.sell()
                self.sell_signals.append((self.data.datetime.date(0), self.dataclose[0]))

            if self.data.close[0] < self.position.price - self.atr[0] * self.params.atr_multiplier:
                self.order = self.sell()
                self.sell_signals.append((self.data.datetime.date(0), self.dataclose[0]))

    def notify_order(self, order):
        if order.status in [order.Completed, order.Canceled, order.Margin]:
            self.order = None

    def notify_trade(self, trade):
        if trade.isclosed:
            print(f'Operation Profit, Gross {trade.pnl}, Net {trade.pnlcomm}')


if __name__ == '__main__':
    price_data = StockService().get_stock_price_data("600010.SH", "20240601", "20250207")
    price_data['trade_date'] = pd.to_datetime(price_data['trade_date'], format='%Y%m%d')
    price_data.set_index('trade_date', inplace=False)
    data = bt.feeds.PandasData(dataname=price_data,
                               datetime='trade_date',
                               open='open',
                               high='high',
                               low='low',
                               close='close',
                               volume='vol')

    # 创建回测引擎
    cerebro = bt.Cerebro()
    cerebro.addstrategy(PredictStrategy)

    # 加载数据
    # data = bt.feeds.YahooFinanceData(dataname='AAPL', fromdate=datetime(2020, 1, 1), todate=datetime(2021, 1, 1))
    cerebro.adddata(data)

    # 设置初始资金
    cerebro.broker.setcash(10000.0)

    # 设置佣金
    cerebro.broker.setcommission(commission=0.001)

    # 运行回测
    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    strategies = cerebro.run()
    print('Ending Portfolio Value: %.2f' % cerebro.broker.getvalue())

    # 获取策略实例
    strategy = strategies[0]

    # 准备数据用于绘图
    dates = []
    for i in range(len(data)):
        idx = -(len(data) - 1 - i)
        dates.append(data.datetime.date(idx))

    open_prices = np.array(data.open.array)
    high_prices = np.array(data.high.array)
    low_prices = np.array(data.low.array)
    close_prices = np.array(data.close.array)
    atr_values = np.array(strategy.atr.array)

    # 创建 Plotly 图表
    fig = go.Figure()

    # 添加 K 线图
    # fig.add_trace(go.Candlestick(x=dates,
    #                              open=open_prices,
    #                              high=high_prices,
    #                              low=low_prices,
    #                              close=close_prices,
    #                              name='Candlestick',
    #                              # 涨的日期为红色
    #                              increasing_line_color='red',
    #                              # 跌的日期为绿色
    #                              decreasing_line_color='green'))
    # 添加收盘价线
    fig.add_trace(go.Scatter(x=dates, y=close_prices, mode='lines', name='Close Price'))

    # 添加 ATR 线
    fig.add_trace(go.Scatter(x=dates, y=atr_values, mode='lines', name='ATR', line=dict(dash='dash')))

    # 添加买入点
    for buy_date, buy_price in strategy.buy_signals:
        fig.add_trace(go.Scatter(x=[buy_date], y=[buy_price], mode='markers', name='Buy Signal', marker=dict(symbol='arrow-up', color='blue', size=8)))

    # 添加卖出点
    for sell_date, sell_price in strategy.sell_signals:
        fig.add_trace(go.Scatter(x=[sell_date], y=[sell_price], mode='markers', name='Sell Signal', marker=dict(symbol='arrow-down', color='orange', size=8)))

    # 添加预测价格线
    for pred_date, pred_prices in strategy.predicted_lines:
        pred_dates = [pred_date + pd.Timedelta(days=i) for i in range(len(pred_prices))]
        fig.add_trace(go.Scatter(x=pred_dates, y=pred_prices, mode='lines', name=f'Predicted Prices {pred_date}', line=dict(dash='dot')))

    # 更新布局
    fig.update_layout(title='Trading Strategy Report', xaxis_title='Date', yaxis_title='Price')

    # 保存为 HTML
    fig.write_html('report.html')

    print("Report saved to report.html")
