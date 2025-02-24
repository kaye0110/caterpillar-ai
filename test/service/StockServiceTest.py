import unittest
from unittest.mock import patch

import pandas as pd

from src.service.DataClearService import DataClearService
from src.service.IndicatorService import IndicatorService
from src.service.OverseeService import OverseeService
from src.service.StockService import StockService


class StockServiceTest(unittest.TestCase):

    def test_get_stock_by_label(self):
        oversee = OverseeService(batch_code="0000001")
        oversee.set_stock_array_by_label(["高股息精选"])
        df = oversee.load_data_with_label(start_date='20230401', end_date='20250120')

        self.assertIsNotNone(df)

    def test(self):
        stock_service = StockService()
        df = stock_service.get_all_stocks()
        self.assertIsNotNone(df)
        self.assertTrue(len(df) > 0)

        ts_code = '000001.SZ'
        start_date = '20210101'
        end_date = '20241231'
        merged_data = stock_service.merge_price_and_adj_data(ts_code, start_date, end_date)
        if merged_data is not None:
            print(merged_data.head())

        self.assertIsNotNone(merged_data)

        indicator_service = IndicatorService()
        cleaned_data: pd.DataFrame = indicator_service.prepare_price(merged_data)
        self.assertIsNotNone(cleaned_data)
        if cleaned_data is not None:
            print(cleaned_data.head())

        cleaned_data: pd.DataFrame = indicator_service.generate_talib(merged_data)
        if cleaned_data is not None:
            print(cleaned_data.head())

        cleaned_data = DataClearService(prices=cleaned_data).covert_to_float().remove_nan_rows().normalize_date().prices
        if cleaned_data is not None:
            print(cleaned_data.head())

    @patch('src.service.StockService.StockService.get_stock_price_data')
    @patch('src.service.StockService.StockService.get_stock_price_adj_data')
    def test_merge_price_and_adj_data(self, mock_get_adj_data, mock_get_price_data):
        # Mock data for price data
        price_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'trade_date': ['20230101', '20230102'],
            'open': [10.0, 10.5],
            'close': [10.5, 11.0]
        })

        # Mock data for adjusted price data
        adj_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'trade_date': ['20230101', '20230102'],
            'adj_factor': [1.1, 1.2],
            'close': [10.6, 11.1]  # This column should be ignored in the merge
        })

        # Set the return value of the mocked methods
        mock_get_price_data.return_value = price_data
        mock_get_adj_data.return_value = adj_data

        # Initialize StockService
        stock_service = StockService()

        # Call the merge method
        merged_data = stock_service.merge_price_and_adj_data('000001.SZ', '20230101', '20230102')

        # Verify the merged data
        expected_data = pd.DataFrame({
            'ts_code': ['000001.SZ', '000001.SZ'],
            'trade_date': ['20230101', '20230102'],
            'open': [10.0, 10.5],
            'close': [10.5, 11.0],  # Should be from price_data, not adj_data
            'adj_factor': [1.1, 1.2]
        })

        pd.testing.assert_frame_equal(merged_data, expected_data)


if __name__ == '__main__':
    unittest.main()
