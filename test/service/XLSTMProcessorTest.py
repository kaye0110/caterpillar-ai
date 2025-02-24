import logging
import unittest

from src.config.AppConfig import AppConfig
from src.model import StockCodeInternet
from src.service.OverseeServiceV2 import OverseeService
from test.service import StockCode


class OverseeServiceTest(unittest.TestCase):
    logger = logging.getLogger(__name__)
    app_config = AppConfig()
    batch_code = "20250201202_300_001"
    stock_code_array = StockCode.ts_code_300
    start_date = '2020101'
    end_date = '20251230'
    train_start_date = '2020-01-01'
    train_end_date = '2024-06-30'
    predict_start_date = '2024-07-01'
    predict_end_date = '2025-02-20'

    poltly_end_date = '20250228'

    def test_load_all_stock_data(self):
        oversee = OverseeService(self.batch_code)
        oversee.stock_code_array = self.stock_code_array
        oversee.start_date = self.start_date
        oversee.end_date = self.end_date
        oversee.train_start_date = self.train_start_date
        oversee.train_end_date = self.train_end_date
        oversee.predict_start_date = self.predict_start_date
        oversee.predict_end_date = self.predict_end_date
        oversee.poltly_end_date = self.poltly_end_date

        oversee.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)

    def test_performance_tuning(self):
        oversee = OverseeService(self.batch_code)
        oversee.stock_code_array = StockCodeInternet.stock_code_internet
        oversee.start_date = self.start_date
        oversee.end_date = self.end_date
        oversee.train_start_date = self.train_start_date
        oversee.train_end_date = self.train_end_date
        oversee.predict_start_date = self.predict_start_date
        oversee.predict_end_date = self.predict_end_date
        oversee.poltly_end_date = self.poltly_end_date

        oversee.performance_tuning_single()
