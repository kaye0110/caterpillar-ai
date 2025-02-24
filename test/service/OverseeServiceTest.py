import logging
import unittest

import pandas as pd
import plotly.graph_objects as go

from src.service.OverseeService import OverseeService
from src.service.processor.LSTMProcessor2 import LSTMProcess
from src.service.processor.Processor import Processor


class OverseeServiceTest(unittest.TestCase):
    logger = logging.getLogger(__name__)

    def test_stock_labels(self):
        oversee = OverseeService()
        oversee.set_stock_array_by_stock_code(["000001.SZ", "000002.SZ"])
        self.logger.info(oversee.stock_array)

    def test_model_with_all(self):
        batch_code = "20250166"

        oversee = OverseeService()
        df = oversee.load_data_with_label(batch_code=batch_code, start_date='20230101', end_date='20251230')
        df['trade_date_copy'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        # 定义筛选条件
        start_date = '2023-06-01'
        end_date = '2025-12-31'

        # 使用布尔索引进行筛选
        mask = (df['trade_date_copy'] >= start_date) & (df['trade_date_copy'] <= end_date)
        filtered_df = df.loc[mask]

        processor = LSTMProcess(name="with_all_data", data=filtered_df, batch_code=batch_code).run_simple()

        print(processor.model.summary())

        self.test_predict(df=df, processor=processor, ts_code="000001.SZ", start_date='2023-01-01', end_date='2023-06-01')

    def test_predict(self, df: pd.DataFrame, processor: Processor, ts_code: str, start_date: str, end_date: str, limit: int = 30):

        mask = (df['trade_date_copy'] >= start_date) & (df['trade_date_copy'] <= end_date) & (df['ts_code'] == ts_code)
        filtered_df = df.loc[mask]

        # 1. 取最后 60 条记录（如果数据本身少于60条，则全部取）
        df_last_60 = filtered_df.tail(limit)

        # 2. 取最后 50 条的 close 和 trade_date 放入 last_close_array 和 last_trade_date_array
        last_close_array = df_last_60['close'].tail(limit - 10).values
        last_trade_date_array = df_last_60['trade_date'].tail(limit - 10).values

        # 初始化 period_date_array 和 period_close_array
        period_date_array = []
        period_close_array = []

        predict_data = []

        # 3 和 4. 按照 1 行为步长分别获取 60-50、59-49...10-0 行的数据，并处理第10行的数据
        for start in range(limit, 9, -1):  # 从60到10（不包括10），倒序循环

            if start == 0:
                a_period_date = df_last_60.iloc[start - 10:]
            else:
                a_period_date = df_last_60.iloc[start - 10:start]  # 获取当前窗口的10行数据

            a_predict_data = processor.model_predict(a_period_date)
            predict_data.append(a_predict_data[0][0])

            # 获取第10行（即索引为start-1）的数据，并找到其下一行的 trade_date 和 close
            if start < len(df_last_60):
                next_idx = start
                period_date_array.append(int(df_last_60.iloc[next_idx]['trade_date']))
                period_close_array.append(df_last_60.iloc[next_idx]['close'])
            else:
                # 如果已经是最后一行，没有下一行可供添加，则跳过
                continue

        print(last_trade_date_array)
        print(period_date_array)
        print(last_close_array)
        print(period_close_array)
        print(predict_data)

        # 使用 Plotly 绘制曲线图
        fig = go.Figure(data=go.Scatter(x=last_trade_date_array, y=predict_data, mode='lines', name='Predict Price'))

        # 添加 close 曲线
        fig.add_trace(go.Scatter(x=last_trade_date_array, y=last_close_array, mode='lines', name='Close Price'))

        # 添加标题和标签
        fig.update_layout(title='Random Data Line Chart',
                          xaxis_title='Index',
                          yaxis_title='Value')

        # 显示图表
        fig.show()

    def test(self):
        oversee = OverseeService()
        processor: Processor = oversee.run('002594.SZ', '20210101', '20240930')

        df = oversee._load_data('002594.SZ', '20240101', '20251231')

        # 1. 取最后 60 条记录（如果数据本身少于60条，则全部取）
        df_last_60 = df.tail(160)

        # 2. 取最后 50 条的 close 和 trade_date 放入 last_close_array 和 last_trade_date_array
        last_close_array = df_last_60['close'].tail(150).values
        last_trade_date_array = df_last_60['trade_date'].tail(150).values

        # 初始化 period_date_array 和 period_close_array
        period_date_array = []
        period_close_array = []

        predict_data = []

        # 3 和 4. 按照 1 行为步长分别获取 60-50、59-49...10-0 行的数据，并处理第10行的数据
        for start in range(160, 9, -1):  # 从60到10（不包括10），倒序循环

            if start == 0:
                a_period_date = df_last_60.iloc[start - 10:]
            else:
                a_period_date = df_last_60.iloc[start - 10:start]  # 获取当前窗口的10行数据

            a_predict_data = processor.model_predict(a_period_date)
            predict_data.append(a_predict_data[0][0])

            # 获取第10行（即索引为start-1）的数据，并找到其下一行的 trade_date 和 close
            if start < len(df_last_60):
                next_idx = start
                period_date_array.append(int(df_last_60.iloc[next_idx]['trade_date']))
                period_close_array.append(df_last_60.iloc[next_idx]['close'])
            else:
                # 如果已经是最后一行，没有下一行可供添加，则跳过
                continue

        print(last_trade_date_array)
        print(period_date_array)
        print(last_close_array)
        print(period_close_array)
        print(predict_data)

        # 使用 Plotly 绘制曲线图
        fig = go.Figure(data=go.Scatter(x=last_trade_date_array, y=predict_data, mode='lines', name='Predict Price'))

        # 添加 close 曲线
        fig.add_trace(go.Scatter(x=last_trade_date_array, y=last_close_array, mode='lines', name='Close Price'))

        # 添加标题和标签
        fig.update_layout(title='Random Data Line Chart',
                          xaxis_title='Index',
                          yaxis_title='Value')

        # 显示图表
        fig.show()

        # self.logger.info(predict_data)
