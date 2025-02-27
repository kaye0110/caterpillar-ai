import datetime
import itertools
import logging
import os
import random
import string
import traceback

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
from sklearn.preprocessing import MinMaxScaler

from src.config.AppConfig import AppConfig
from src.infra.repo.DataStore import DataStore
from src.infra.repo.StockPriceProvider import StockPriceProvider
from src.service.processor.ProcessorBuilder import ProcessorBuilder
from src.service.processor.torch.HybridModelProcess import HybridModelProcess
from src.service.processor.torch.XLSTMProcessor import XLSTMProcess
from src.service.processor.v3.Config import Configuration
from src.service.processor.v3.Processor import Processor
from src.tools.pycaterpillar_wrapper import time_counter


class OverseeService:
    params = {
        "file_name_stock_label": "stock_label",
        "file_name_stock_price": "stock_price",
        "file_name_stock_basic": "stock_basic",
        "file_name_stock_indicator": "stock_indicator",
        "file_name_stock_with_label": "stock_with_label",
        "file_type": "parquet"
    }

    def __init__(self, batch_code: str = None):
        self.logger = logging.getLogger(__name__)
        self.current_time = datetime.datetime.now().strftime("%Y%m%d%H")
        self.random_num_char = ''.join(random.choices(string.ascii_letters + string.digits, k=2))
        self.stock_price_provider = StockPriceProvider()
        if batch_code is None:
            self.batch_code = self.current_time + self.random_num_char
        else:
            self.batch_code = batch_code
        self.datastore = DataStore(self.batch_code)

        self.logger = logging.getLogger(__name__)
        self.app_config = AppConfig()
        self.stock_code_array = None
        self.start_date = None
        self.end_date = None
        self.train_start_date = None
        self.train_end_date = None
        self.predict_start_date = None
        self.predict_end_date = None
        self.polt_end_date = None
        self.model_type = 'XLSTM'

        self.predict_trade_list = {}

    def train_and_test_single(self):
        df = self.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)
        for stock_code in self.stock_code_array:
            try:
                self._train_and_test(df, stock_code)
            except Exception as e:
                self.logger.error(f"Error occurred while training and testing {stock_code}: {str(e)}")
                traceback.print_exc()

    @time_counter(logger_name=__name__)
    def train_and_test_all(self):
        df = self.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)
        self._train_and_test(df)

    @time_counter(logger_name=__name__)
    def performance_tuning_all(self):
        df = self.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)
        params = {'all': self._performance_tuning(df)}

    def performance_tuning_single(self):
        df = self.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)
        params = {}
        for stock_code in self.stock_code_array:
            try:
                params = self._performance_tuning(df, stock_code)
                if params is None or len(params) == 0:
                    continue

                params[stock_code] = params
            except Exception as e:
                self.logger.error("Error in performance tuning for stock code: %s, error: %s", stock_code, str(e))

    def predict_single(self):
        df = self.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)
        last_day_datas = []
        for stock_code in self.stock_code_array:
            try:
                file_name = self.predict(df, stock_code)
                last_day_datas.append(file_name)
            except Exception as e:
                self.logger.error("Error in predict for stock code: %s, error: %s", stock_code, str(e))
                traceback.print_exc()

        return

    @time_counter(logger_name=__name__)
    def _train_and_test(self, df: pd.DataFrame, ts_code: str = 'all'):
        if ts_code is None or ts_code == "" or ts_code not in df['ts_code'].values:
            df_single = df.copy()
        else:
            df_single = df[df['ts_code'] == ts_code].copy()

        df_single['trade_date_copy'] = pd.to_datetime(df_single['trade_date'], format='%Y%m%d')

        # 使用布尔索引进行筛选
        mask = (df_single['trade_date_copy'] >= self.train_start_date) & (df_single['trade_date_copy'] <= self.train_end_date)
        filtered_df = df_single.loc[mask]

        config = Configuration().set_model_type(self.model_type).set_batch_code(self.batch_code)

        config.batch_size = 64
        config.hidden_size = 128
        config.num_layers = 1
        config.dropout = 0.1
        config.learning_rate = 0.0001
        config.model_idx = 9999


        processor = None
        if "XLSTM" == self.model_type:
            processor = XLSTMProcess(config)
        elif "hybrid" == self.model_type:
            processor = HybridModelProcess(config)

        if ts_code is not None and ts_code != "" and ts_code in df['ts_code'].values:
            processor.stock_code = ts_code

        processor.prepare(data=filtered_df).build_model().train().test().report()



    @time_counter(logger_name=__name__)
    def _performance_tuning(self, df: pd.DataFrame, ts_code: str = 'all') -> {}:
        if ts_code is None or ts_code == "" or ts_code == "all":
            df_single = df.copy()
        elif ts_code in df['ts_code'].values:
            df_single = df[df['ts_code'] == ts_code].copy()
        else:
            self.logger.warning("No Data found for stock: {}", ts_code)

            return {}

        df_single['trade_date_copy'] = pd.to_datetime(df_single['trade_date'], format='%Y%m%d')

        # 使用布尔索引进行筛选
        mask = (df_single['trade_date_copy'] >= self.train_start_date) & (df_single['trade_date_copy'] <= self.train_end_date)
        filtered_df = df_single.loc[mask]

        config = Configuration().set_model_type(self.model_type).set_batch_code(self.batch_code)

        # 定义参数网格
        param_grid = {
            'batch_size': [32, 64, 128],
            'hidden_size': [64, 128, 256],
            'num_layers': [1, 2, 3],
            'dropout': [0.05, 0.1, 0.2],
            'learning_rate': [0.0005, 0.0001, 0.00005],
            'max_depth': [100],  # [50, 100, 150],
            'n_estimators': [100],  # [50, 100, 200],
        }

        # 生成所有参数组合
        param_combinations = list(itertools.product(
            param_grid['batch_size'],
            param_grid['hidden_size'],
            param_grid['num_layers'],
            param_grid['dropout'],
            param_grid['learning_rate'],
            param_grid['max_depth'],
            param_grid['n_estimators'],

        ))

        # ${ts_code}_${idx} = {model.params & score}
        metrics = {}
        processor = None
        # 循环遍历所有参数组合
        for idx in range(len(param_combinations)):
            params = param_combinations[idx]
            if params is None or len(params) != 7:
                continue

            try:
                config = Configuration().set_model_type(self.model_type).set_batch_code(self.batch_code)
                config.batch_size = params[0]
                config.hidden_size = params[1]
                config.num_layers = params[2]
                config.dropout = params[3]
                config.learning_rate = params[4]
                config.max_depth = params[5]
                config.n_estimators = params[6]
                config.model_idx = idx + 1

                processor = None
                if "XLSTM" == self.model_type:
                    processor = XLSTMProcess(config)
                elif "hybrid" == self.model_type:
                    processor = HybridModelProcess(config)

                if ts_code is not None and ts_code != "" and ts_code in df['ts_code'].values:
                    processor.stock_code = ts_code

                processor.prepare(data=filtered_df).build_model().train().test().report()

                # 获取测试结果
                mse = processor.result_mse
                mae = processor.result_mae
                r2 = processor.result_r2
                rmse = processor.result_rmse
                mape = processor.result_mape
                smape = processor.result_smape
                explained_variance = processor.result_explained_variance
                adj_r2 = processor.result_adjusted_r2
                composite_score = processor.composite_score

                # 计算加权得分
                metrics[f'{ts_code}_{idx}'] = {
                    'batch_size': config.batch_size,
                    'hidden_size': config.hidden_size,
                    'num_layers': config.num_layers,
                    'dropout': config.dropout,
                    'learning_rate': config.learning_rate,
                    'max_depth': config.max_depth,
                    'n_estimators': config.n_estimators,
                    'model_idx': config.model_idx,
                    'mse': mse,
                    'mae': mae,
                    'rmse': rmse,
                    'mape': mape,
                    'smape': smape,
                    'explained_variance': explained_variance,
                    'adj_r2': adj_r2,
                    'composite_score': composite_score
                }
                self.logger.info(f"Stock:{ts_code} Model:{idx}/{len(param_combinations)} finished with composite_score:{composite_score}, mse:{mse}, mae:{mae}, r2:{r2} "
                                 f"rmse:{rmse} mape:{mape} smape:{smape} explained_variance:{explained_variance} adj_r2:{adj_r2}")
            except Exception as e:
                self.logger.error(f"An error occurred: {e} {config.batch_size} {config.hidden_size} {config.num_layers} {config.dropout} {config.learning_rate} ")
                traceback.print_exc()

        # 提取所有记录并按 composite_score 降序排序
        sorted_records = sorted(
            metrics.values(),
            key=lambda x: x['composite_score'],
            reverse=True
        )

        # 获取前10条记录
        top_records = sorted_records[:config.model_top_size]
        top_dataframe = pd.DataFrame(top_records)
        self.logger.info(f"Stock:{ts_code} Top 10 Records: {top_records}")

        file_name = f"performance_tuning_result_{processor.store.batch_code}_{ts_code}"
        processor.store.save_model_data(data=top_dataframe, file_name=file_name)

        return top_dataframe

    def predict(self, df: pd.DataFrame, ts_code: str) -> str:
        if ts_code is None or ts_code == "" or ts_code not in df['ts_code'].values:
            df_single = df.copy()
        else:
            df_single = df[df['ts_code'] == ts_code].copy()

        df_single['trade_date_copy'] = pd.to_datetime(df_single['trade_date'], format='%Y%m%d')

        # 使用布尔索引进行筛选
        mask = (df_single['trade_date_copy'] >= self.predict_start_date) & (df_single['trade_date_copy'] <= self.predict_end_date)
        filtered_df = df_single.loc[mask]

        file_name = f"performance_tuning_result_{self.batch_code}_{ts_code}"
        top_params = self.datastore.load_model_data(file_name=file_name)

        if top_params is None or len(top_params) <= 0:
            processor = ProcessorBuilder.build_by_batch_code(batch_code=self.batch_code, stock_code=ts_code, model_idx="9999")
            df_last = filtered_df.tail(processor.config.n_timestep + processor.config.n_predict + 30)
            return self.predict_with_export(df=df_last, processor=processor, ts_code=ts_code, start_date=self.predict_start_date, end_date=self.predict_end_date, plotly_end_date=self.polt_end_date)

        else:
            for idx, record in top_params.iterrows():
                self.logger.info(f"Record {record['model_idx']}, composite_score: {record['composite_score']}  mse: {record['mse']}  mae: {record['mae']}  rmse: {record['rmse']}  mape: {record['mape']}  smape: {record['smape']}  explained_variance: {record['explained_variance']}  adj_r2: {record['adj_r2']}   ")
                self.logger.info(f"Record {record['model_idx']}, batch_size: {record['batch_size']} hidden_size: {record['hidden_size']} num_layers: {record['num_layers']} dropout: {record['dropout']} learning_rate: {record['learning_rate']} max_depth: {record['max_depth']} n_estimators: {record['n_estimators']}  ")

                processor = ProcessorBuilder.build_by_batch_code(batch_code=self.batch_code, stock_code=ts_code, model_idx=str(record['model_idx']))
                df_last = filtered_df.tail(processor.config.n_timestep + processor.config.n_predict + 30)
                self.predict_with_export(df=df_last, processor=processor, ts_code=ts_code, start_date=self.predict_start_date, end_date=self.predict_end_date, plotly_end_date=self.polt_end_date)

    def predict_with_export(self, df: pd.DataFrame, processor: Processor, ts_code: str, start_date: str, end_date: str, plotly_end_date: str) -> str:

        # 获取最后一条记录的 trade_date
        last_trade_date = pd.to_datetime(df['trade_date'].iloc[-1], format='%Y%m%d')

        # 生成 n_predict 天的日期
        future_dates = pd.date_range(start=last_trade_date + pd.Timedelta(days=1), periods=processor.config.n_predict)

        # 创建占位符数据，并填充 future_dates 到 trade_date 列
        placeholder_data = pd.DataFrame(
            np.nan,
            index=range(processor.config.n_predict),
            columns=df.columns
        )
        placeholder_data['trade_date'] = future_dates.strftime('%Y%m%d')  # 将日期格式化为字符串

        df_last = pd.concat([df, placeholder_data], ignore_index=True)

        # 获取真实的 close 价格和交易日期
        last_close_array = df_last['close'].values
        last_trade_date_array = df_last['trade_date'].values

        # 确保日期格式正确
        last_trade_date_array = pd.to_datetime(last_trade_date_array, format='%Y%m%d', errors='coerce')

        # 直接将 df_last 丢给模型进行预测
        predict_data_array = processor.predict(df_last)
        if predict_data_array is None:
            raise ValueError("预测失败，返回结果为 None。")

        # 处理预测数据
        predict_data_line = []

        for i in range(len(predict_data_array)):
            a_predict_data = predict_data_array[i, :]
            tmp = np.append(np.full(processor.config.n_timestep + i, np.nan), a_predict_data)
            if len(tmp) < len(last_close_array):
                tmp = np.append(tmp, np.full(len(last_close_array) - len(tmp), np.nan))
            predict_data_line.append(tmp)

        # 扩展 x 轴到 plotly_end_date
        plotly_end_date = pd.to_datetime(plotly_end_date, format='%Y%m%d')
        if not pd.isna(last_trade_date_array[-1]) and plotly_end_date > last_trade_date_array[-1]:
            extended_dates = pd.date_range(start=last_trade_date_array[-1] + pd.Timedelta(days=1), end=plotly_end_date)
            last_trade_date_array = np.append(last_trade_date_array, extended_dates)

            # 如果 extended_dates 在 df 的范围内，补充对应的 close 值
            extended_close_values = np.full(len(extended_dates), np.nan)
            for i, date in enumerate(extended_dates):
                if date in df['trade_date_copy'].values:
                    extended_close_values[i] = df.loc[df['trade_date_copy'] == date, 'close'].values[0]

            last_close_array = np.append(last_close_array, extended_close_values)

        # 使用 Plotly 绘制曲线图
        fig = go.Figure()

        # 添加真实的 close 价格曲线
        fig.add_trace(go.Scatter(x=last_trade_date_array, y=last_close_array, mode='lines', name='Close Price'))

        # 添加预测价格曲线
        for i, predict_line in enumerate(predict_data_line):
            fig.add_trace(go.Scatter(
                x=last_trade_date_array,
                y=predict_line,
                mode='lines',
                name=f'Predict Price Line {i + 1}',
                visible='legendonly'
            ))

        # 添加标题和标签
        fig.update_layout(
            title=f'Predict Data Line Chart {ts_code} from {start_date} to {end_date}',
            xaxis_title='Date',
            yaxis_title='Price',
            hovermode='x unified'
        )

        # 保存为 HTML 文件
        fig_path = os.path.join(processor.store.get_predict_result_path(),
                                processor.store.get_plotly_name(processor.config.batch_code, ts_code, f"{processor.config.model_idx}_{start_date}_{end_date}"))
        fig.write_html(fig_path)

        # 获取最后一行的预测价格
        last_predict_prices = predict_data_array[-1, :]

        segments = self.segment_data(last_predict_prices)
        # 输出结果
        for segment in segments:
            start_idx, end_idx, length, change_rate = segment
            self.logger.info(f"StockCode:{ts_code} Start: {start_idx}, End: {end_idx}, Length: {length}, Change Rate: {change_rate:.2f}")

        last_date = df['trade_date_copy'].iloc[-1]

        # 计算价格变化
        price_changes = {
            '1_day_pct_chg': (last_predict_prices[1] - last_predict_prices[0]) / last_predict_prices[0],
            '2_day_pct_chg': (last_predict_prices[2] - last_predict_prices[0]) / last_predict_prices[0],
            '3_day_pct_chg': (last_predict_prices[3] - last_predict_prices[0]) / last_predict_prices[0],
            '5_day_pct_chg': (last_predict_prices[5] - last_predict_prices[0]) / last_predict_prices[0],
            '8_day_pct_chg': (last_predict_prices[8] - last_predict_prices[0]) / last_predict_prices[0]
        }
        # 创建 DataFrame 保存预测价格和变化¬
        predict_price_df = pd.DataFrame({
            'ts_code': [ts_code],
            'last_predict_price': [last_predict_prices[-1]],
            **price_changes
        })

        # 保存到本地文件
        file_name = f"predict_data_{processor.config.batch_code}_{ts_code}_{processor.config.model_idx}_{start_date}_{end_date}"
        self.datastore.save_predict_result(predict_price_df, file_name)

        for day, pct_chg in price_changes.items():
            date_key = (last_date + datetime.timedelta(days=int(day.split('_')[0]))).strftime('%Y%m%d')
            if date_key not in self.predict_trade_list:
                self.predict_trade_list[date_key] = {'buy_list': [], 'sell_list': []}

                if pct_chg > 0.05:  # 假设涨幅超过5%为买入信号
                    self.predict_trade_list[date_key]['buy_list'].append(ts_code)
                elif pct_chg < -0.05:  # 假设跌幅超过5%为卖出信号
                    self.predict_trade_list[date_key]['sell_list'].append(ts_code)

        return file_name

    @staticmethod
    def segment_data(x):
        segments = []
        start_idx = 0
        current_trend = None  # None, 'increasing', or 'decreasing'

        for i in range(1, len(x)):
            if x[i] > x[i - 1]:
                trend = 'increasing'
            elif x[i] < x[i - 1]:
                trend = 'decreasing'
            else:
                trend = 'constant'

            if current_trend is None:
                current_trend = trend

            if trend != current_trend or trend == 'constant':
                if current_trend != 'constant':
                    end_idx = i - 1
                    segment = x[start_idx:end_idx + 1]
                    length = end_idx - start_idx + 1
                    change_rate = (x[end_idx] - x[start_idx]) / length
                    segments.append((start_idx, end_idx, length, change_rate))
                start_idx = i
                current_trend = trend

        # Add the last segment
        if current_trend != 'constant':
            end_idx = len(x) - 1
            segment = x[start_idx:end_idx + 1]
            length = end_idx - start_idx + 1
            change_rate = (x[end_idx] - x[start_idx]) / length
            segments.append((start_idx, end_idx, length, change_rate))

        return segments

    @time_counter(logger_name=__name__)
    def load_data(self, stock_codes: [str], start_date: str, end_date: str) -> pd.DataFrame:

        file_name = f"indicator_data_{start_date}_{end_date}"
        data = self.datastore.load_original_data(file_name=file_name)
        if data is not None and len(data) > 0:
            return data

        file_path, file_name = self.stock_price_provider.calculate_indicator(batch_code=self.batch_code, start_date=start_date, end_date=end_date, stock_codes=stock_codes)
        export_path = self.datastore.generate_original_data_path(file_name=file_name)
        downloaded = self.stock_price_provider.download_indicator(export_path=export_path, batch_code=self.batch_code, start_date=start_date, end_date=end_date, stock_codes=stock_codes)
        if not downloaded:
            raise "Download failed."

        return self.datastore.load_original_data(file_name=file_name)
