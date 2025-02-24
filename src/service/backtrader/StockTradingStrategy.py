import traceback

import numpy as np
import pandas as pd

from src.service.processor.ProcessorBuilder import ProcessorBuilder


class StockTradingStrategy:
    def __init__(self, batch_code, stock_code_array, start_date, end_date, predict_start_date, predict_end_date, datastore):
        # 初始化股票交易策略类
        self.batch_code = batch_code
        self.stock_code_array = stock_code_array
        self.start_date = start_date
        self.end_date = end_date
        self.predict_start_date = predict_start_date
        self.predict_end_date = predict_end_date
        self.datastore = datastore

    def predict_and_select_stocks(self):
        # 预测并选择股票
        df = self.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)
        buy_list = []
        sell_list = []

        for stock_code in self.stock_code_array:
            try:
                predict_result = self.predict(df, stock_code)
                if predict_result is not None:
                    buy, sell = self.select_stocks(predict_result, stock_code)
                    buy_list.extend(buy)
                    sell_list.extend(sell)
            except Exception as e:
                print(f"Error in predict for stock code: {stock_code}, error: {str(e)}")
                traceback.print_exc()

        return buy_list, sell_list

    def predict(self, df: pd.DataFrame, ts_code: str):
        # 预测股票价格
        if ts_code is None or ts_code == "" or ts_code not in df['ts_code'].values:
            df_single = df.copy()
        else:
            df_single = df[df['ts_code'] == ts_code].copy()

        df_single['trade_date_copy'] = pd.to_datetime(df_single['trade_date'], format='%Y%m%d')

        # 使用布尔索引进行筛选
        mask = (df_single['trade_date_copy'] >= self.predict_start_date) & (df_single['trade_date_copy'] <= self.predict_end_date)
        filtered_df = df_single.loc[mask]

        processor = ProcessorBuilder.build_by_batch_code(batch_code=self.batch_code, stock_code=ts_code)
        df_last = filtered_df.tail(processor.config.n_timestep + processor.config.n_predict + 30)

        return self.predict_with_export(df=df_last, processor=processor, ts_code=ts_code)

    def predict_with_export(self, df: pd.DataFrame, processor, ts_code: str):
        # 预测并导出结果
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

        # 直接将 df_last 丢给模型进行预测
        a_predict_data = processor.predict(df_last)
        if a_predict_data is None:
            raise ValueError("预测失败，返回结果为 None。")

        return a_predict_data

    def select_stocks(self, predict_data, ts_code):
        # 选择股票
        buy_list = []
        sell_list = []

        # 获取最后一行的预测价格
        last_predict_prices = predict_data[-1, :]

        # 计算价格变化
        price_changes = {
            '1_day_pct_chg': (last_predict_prices[1] - last_predict_prices[0]) / last_predict_prices[0],
            '3_day_pct_chg': (last_predict_prices[3] - last_predict_prices[0]) / last_predict_prices[0],
            '5_day_pct_chg': (last_predict_prices[5] - last_predict_prices[0]) / last_predict_prices[0],
            '8_day_pct_chg': (last_predict_prices[8] - last_predict_prices[0]) / last_predict_prices[0]
        }

        # 根据涨跌幅情况挑选股票
        if price_changes['1_day_pct_chg'] > 0.02 or price_changes['3_day_pct_chg'] > 0.05:
            buy_list.append(ts_code)
        if price_changes['5_day_pct_chg'] < -0.02 or price_changes['8_day_pct_chg'] < -0.05:
            sell_list.append(ts_code)

        return buy_list, sell_list
