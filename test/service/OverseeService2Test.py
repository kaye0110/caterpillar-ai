import logging
import unittest

import pandas as pd

from src.service.OverseeServiceV2 import OverseeService
from src.service.processor.v3.Config import Configuration
from src.service.processor.v3.LSTMProcessor import LSTMProcess


class OverseeServiceTest(unittest.TestCase):
    logger = logging.getLogger(__name__)
    batch_code = '20250211003'

    def test_stock_labels(self):
        oversee = OverseeService(self.batch_code)
        df = oversee.load_data(stock_codes=['000001.SZ'], start_date='20230101', end_date='20251230')

        df['trade_date_copy'] = pd.to_datetime(df['trade_date'], format='%Y%m%d')
        # 定义筛选条件
        start_date = '2020-01-01'
        end_date = '2024-06-30'

        # 使用布尔索引进行筛选
        mask = (df['trade_date_copy'] >= start_date) & (df['trade_date_copy'] <= end_date)
        filtered_df = df.loc[mask]

        config = Configuration().set_model_type("LSTM").set_batch_code(self.batch_code)

        processor = LSTMProcess(config).prepare(data=filtered_df).train().test().report()

        print(processor.model.summary())

        print(df)
