import datetime
import logging
import os
import unittest

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from src.config.AppConfig import AppConfig
from src.service.OverseeService import OverseeService
from src.service.processor.ProcessorBuilder import ProcessorBuilder
from src.service.processor.v3.Config import Configuration
from src.service.processor.v3.LSTMProcessor import LSTMProcess
from src.service.processor.v3.Processor import Processor


class OverseeServiceTest(unittest.TestCase):
    logger = logging.getLogger(__name__)
    app_config = AppConfig()
    batch_code = "20250204_高股息精选"
    labels = ['高股息精选']

    def test_model_with_all(self):
        oversee = OverseeService(batch_code=self.batch_code)
        oversee.set_stock_array_by_label(labels=self.labels)
        df = oversee.load_data_with_label(start_date='20230101', end_date='20250120')
        df['trade_date_copy'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        # 定义筛选条件
        start_date = '2023-06-01'
        end_date = '2024-12-31'

        # 使用布尔索引进行筛选
        mask = (df['trade_date_copy'] >= start_date) & (df['trade_date_copy'] <= end_date)
        filtered_df = df.loc[mask]

        config = Configuration().set_model_type("LSTM").set_batch_code(self.batch_code)

        processor = LSTMProcess(config).prepare(data=filtered_df).train().test().report()

        self.predict(df=df, processor=processor, ts_code="000001.SZ", start_date='2024-11-01', end_date='2025-01-10')

    def test_load_model_and_predict(self):
        oversee = OverseeService(batch_code=self.batch_code)
        oversee.set_single_stock_by_code("002049.SZ")
        df = oversee.load_data_with_label(start_date='20230301', end_date='20250120')
        df['trade_date_copy'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

        processor = ProcessorBuilder.build_by_batch_code(batch_code=self.batch_code)

        self.predict(df=df, processor=processor, ts_code="002049.SZ", start_date='2024-06-01', end_date='2025-01-20')

    def predict(self, df: pd.DataFrame, processor: Processor, ts_code: str, start_date: str, end_date: str):
        mask = (df['trade_date_copy'] >= start_date) & (df['trade_date_copy'] <= end_date) & (df['stock_code'] == ts_code)
        filtered_df = df.loc[mask]

        df_last = filtered_df.tail(processor.config.n_timestep + processor.config.n_predict * 2)

        last_close_array = df_last[processor.config.n_timestep:]['close'].values
        last_trade_date_array = df_last[processor.config.n_timestep:]['trade_date'].values

        # 初始化 period_date_array 和 period_close_array
        period_date_array = []
        period_close_array = []

        predict_data = []

        predict_data_line = []
        start = 0
        end = (len(df_last) - processor.config.n_predict)
        for start in range(start, end, 1):
            a_period_data = df_last.iloc[start: start + processor.config.n_timestep]
            if len(a_period_data) < processor.config.n_timestep:
                break
            a_predict_data = processor.predict(a_period_data).future_predict_value

            len_of_append = len(last_close_array) - len(a_predict_data[0]) - start

            tmp = None
            if start == 0:
                tmp = np.append(a_predict_data[0], np.full(len_of_append, np.nan))
            elif len_of_append <= 0:
                tmp = np.append(np.full(start, np.nan), a_predict_data[0])
            else:
                tmp = np.append(np.full(start, np.nan), a_predict_data[0])
                tmp = np.append(tmp, np.full(len_of_append, np.nan))

            predict_data_line.append(tmp)

            predict_data.append(a_predict_data[0][0])
            next_day_data = df_last.iloc[start + processor.config.n_timestep:start + processor.config.n_timestep + 1]
            if next_day_data is not None and len(next_day_data) > 0:
                period_date_array.append(next_day_data['trade_date'].tolist()[0])
                period_close_array.append(next_day_data['close'].tolist()[0])

        print(last_trade_date_array)
        print(period_date_array)
        print(last_close_array)
        print(period_close_array)
        print(predict_data)

        # 使用 Plotly 绘制曲线图
        fig = go.Figure(data=go.Scatter(x=last_trade_date_array, y=predict_data, mode='lines', name='Predict Price'))

        # 添加 close 曲线
        fig.add_trace(go.Scatter(x=last_trade_date_array, y=last_close_array, mode='lines', name='Close Price'))

        # 添加 close 曲线
        fig.add_trace(go.Scatter(x=last_trade_date_array, y=period_close_array, mode='lines', name='Next Day Close Price'))

        for i in range(len(predict_data_line)):
            fig.add_trace(go.Scatter(x=last_trade_date_array, y=predict_data_line[i], mode='lines', name=f'Predict Price Line {i}'))

        # 添加标题和标签
        fig.update_layout(title=f'Predict Data Line Chart {ts_code} from:{start_date} to {end_date}',
                          xaxis_title='Index',
                          yaxis_title='Value')

        fig_path = os.path.join(processor.store.get_predict_result_path(),
                                processor.store.get_plotly_name(f"{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{ts_code}_{start_date}_{end_date}.html"))
        # 保存为 HTML 文件
        fig.write_html(fig_path)

        # 显示图表
        # fig.show()

    def test_choose_top10(self):
        oversee = OverseeService(batch_code=self.batch_code)
        oversee.set_stock_array_by_label(self.labels)
        df = oversee.load_data_with_label(start_date='20230401', end_date='20250120')
        df['trade_date_copy'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')

        processor = ProcessorBuilder.build_by_batch_code(batch_code=self.batch_code)

        stock_code_array = df["ts_code"].unique()

        for stock_code in stock_code_array:
            stock_df = df[(df["ts_code"] == stock_code)].tail(processor.config.n_timestep * 5)
            last_predict_price = self.predict_last(stock_df, processor, stock_code)
            self.logger.info(f"last_predict_price: {last_predict_price}")

    def predict_last(self, stock_df, processor, ts_code):
        start_date = stock_df.iloc[0]['trade_date']
        end_date = stock_df.iloc[-1]['trade_date']

        last_close_array = np.append(stock_df[processor.config.n_timestep:]['close'].values, np.full(processor.config.n_timestep, np.nan))
        dynamic_days = np.array([f'day_{i + 1}' for i in range(processor.config.n_timestep)])
        last_trade_date_array = np.append(stock_df[processor.config.n_timestep:]['trade_date'].values, dynamic_days)

        # 初始化 period_date_array 和 period_close_array
        period_date_array = []
        period_close_array = []

        predict_data = []

        predict_data_line = []
        start = 0
        end = len(stock_df)
        for start in range(start, end, 1):
            period_end = start + processor.config.n_timestep
            if period_end > end:
                break

            a_period_data = stock_df.iloc[start: period_end]
            a_predict_data = processor.predict(a_period_data).future_predict_value

            len_of_append = len(last_close_array) - len(a_predict_data[0]) - start

            tmp = None
            if start == 0:
                tmp = np.append(a_predict_data[0], np.full(len_of_append, np.nan))
            elif len_of_append <= 0:
                tmp = np.append(np.full(start, np.nan), a_predict_data[0])
            else:
                tmp = np.append(np.full(start, np.nan), a_predict_data[0])
                tmp = np.append(tmp, np.full(len_of_append, np.nan))

            predict_data_line.append(tmp)

            predict_data.append(a_predict_data[0][0])
            next_day_data = stock_df.iloc[period_end:period_end + 1]
            if next_day_data is not None and len(next_day_data) > 0:
                period_date_array.append(next_day_data['trade_date'].tolist()[0])
                period_close_array.append(next_day_data['close'].tolist()[0])

        print(last_trade_date_array)
        print(period_date_array)
        print(last_close_array)
        print(period_close_array)
        print(predict_data)

        # 使用 Plotly 绘制曲线图
        fig = go.Figure(data=go.Scatter(x=last_trade_date_array, y=predict_data, mode='lines', name='Predict Price'))

        # 添加 close 曲线
        fig.add_trace(go.Scatter(x=last_trade_date_array, y=last_close_array, mode='lines', name='Close Price'))

        # 添加 close 曲线
        fig.add_trace(go.Scatter(x=last_trade_date_array, y=period_close_array, mode='lines', name='Next Day Close Price'))

        for i in range(len(predict_data_line)):
            fig.add_trace(go.Scatter(x=last_trade_date_array,
                                     y=predict_data_line[i],
                                     mode='lines',
                                     name=f'Predict Price Line {i}',
                                     visible=True if i == len(predict_data_line) - 1 else 'legendonly'))

        # 添加标题和标签
        fig.update_layout(title=f'Predict Data Line Chart {ts_code} from:{start_date} to {end_date}',
                          xaxis_title='Index',
                          yaxis_title='Value',
                          xaxis=dict(
                              showspikes=True,  # 启用 x 轴参考线
                              spikemode='across',  # 参考线贯穿整个图表
                              spikesnap='cursor',  # 参考线跟随光标
                              spikethickness=1,  # 参考线粗细
                              spikecolor='gray'  # 参考线颜色
                          ),
                          yaxis=dict(
                              showspikes=True,  # 启用 y 轴参考线
                              spikemode='across',  # 参考线贯穿整个图表
                              spikesnap='cursor',  # 参考线跟随光标
                              spikethickness=1,  # 参考线粗细
                              spikecolor='gray'  # 参考线颜色
                          ),
                          hovermode='x unified',  # 悬停模式（'x'、'y' 或 'closest'）
                          spikedistance=-1,  # 显示参考线的距离（-1 表示总是显示）
                          showlegend=True)

        fig_path = os.path.join(processor.store.get_predict_result_path(),
                                processor.store.get_plotly_name(f"{datetime.datetime.now().strftime("%Y%m%d%H%M%S")}_{ts_code}_{start_date}_{end_date}.html"))
        # 保存为 HTML 文件
        fig.write_html(fig_path)

        return period_close_array[len(period_close_array) - 1]
