import logging
import unittest

from src.config.AppConfig import AppConfig
from src.model.StockCodeTech50ETF import stock_code_tech_50_etf
from src.service.OverseeServiceV2 import OverseeService


class OverseeServiceTest:
    logger = logging.getLogger(__name__)
    app_config = AppConfig()
    model_type = "hybrid"
    batch_code = "20250226_001"
    stock_code_array = stock_code_tech_50_etf
    start_date = '2020101'
    end_date = '20250225'
    train_start_date = '2020-01-01'
    train_end_date = '2025-01-30'
    predict_start_date = '2024-11-01'
    predict_end_date = '2025-02-25'

    poltly_end_date = '20250325'

    def test_load_all_stock_data(self):
        oversee = OverseeService(self.batch_code)
        oversee.model_type = self.model_type
        oversee.stock_code_array = self.stock_code_array
        oversee.start_date = self.start_date
        oversee.end_date = self.end_date
        oversee.train_start_date = self.train_start_date
        oversee.train_end_date = self.train_end_date
        oversee.predict_start_date = self.predict_start_date
        oversee.predict_end_date = self.predict_end_date
        oversee.polt_end_date = self.poltly_end_date

        oversee.load_data(stock_codes=self.stock_code_array, start_date=self.start_date, end_date=self.end_date)

    def test_performance_tuning(self):
        oversee = OverseeService(self.batch_code)
        oversee.model_type = self.model_type
        oversee.stock_code_array = self.stock_code_array
        oversee.start_date = self.start_date
        oversee.end_date = self.end_date
        oversee.train_start_date = self.train_start_date
        oversee.train_end_date = self.train_end_date
        oversee.predict_start_date = self.predict_start_date
        oversee.predict_end_date = self.predict_end_date
        oversee.polt_end_date = self.poltly_end_date

        oversee.performance_tuning_single()
        # oversee.train_and_test_single()

    def test_predict(self):
        oversee = OverseeService(self.batch_code)
        oversee.model_type = self.model_type
        oversee.stock_code_array = self.stock_code_array
        oversee.start_date = self.start_date
        oversee.end_date = self.end_date
        oversee.train_start_date = self.train_start_date
        oversee.train_end_date = self.train_end_date
        oversee.predict_start_date = self.predict_start_date
        oversee.predict_end_date = self.predict_end_date
        oversee.polt_end_date = self.poltly_end_date

        oversee.predict_single()

        self.logger.info("Test predict finished")
        self.logger.info(oversee.predict_trade_list)


if __name__ == '__main__':
    test = OverseeServiceTest()
    test.test_load_all_stock_data()
    test.test_performance_tuning()
    test.test_predict()
