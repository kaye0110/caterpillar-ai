import concurrent.futures
import datetime
import logging
import multiprocessing
import random
import string

import pandas as pd

from src.infra.repo.DataStore import DataStore
from src.service.DataClearService import DataClearService
from src.service.IndicatorService import IndicatorService
from src.service.StockService import StockService
from src.service.processor.LSTMProcessor2 import LSTMProcess
from src.service.processor.Processor import Processor
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
        self.stock_service = StockService()
        self.indicator_service = IndicatorService()
        if batch_code is None:
            self.batch_code = self.current_time + self.random_num_char
        else:
            self.batch_code = batch_code
        self.datastore = DataStore(self.batch_code)

        self.stock_array = None

    def set_single_stock_by_code(self, stock_code: str) -> "OverseeService":
        self.stock_array = self.stock_service.get_stock_by_code(stock_code)

        return self

    def set_stock_array_by_stock_code(self, stock_code: [str]) -> "OverseeService":
        if stock_code is None:
            return self

        file_name = f"{self.params["file_name_stock_basic"]}_{"_".join(stock_code)}"
        stock_array = self.datastore.load_original_data(file_name=file_name, file_type=self.params["file_type"])
        if stock_array is not None and len(stock_array) > 0:
            self.stock_array = stock_array
            return self

        stock_label = self.stock_service.get_tx_index_by_code(stock_code)
        ths_stock_label = self.stock_service.get_ths_index_by_code(stock_code)
        stock_label = pd.concat([stock_label, ths_stock_label], ignore_index=True)
        # 获取各列的唯一值
        unique_values = stock_label['stock_code'].unique()

        # 将多个唯一值数组合并为一个列表，并过滤掉空字符和 NaN
        combined_unique = (
            [str(item) for item in unique_values if item and str(item) != 'nan' and str(item) != '']
        )

        # 去重（如果需要）
        combined_unique = list(set(combined_unique))

        return self.set_stock_array_by_label(combined_unique)

    def set_stock_array_by_label(self, labels: [str]) -> "OverseeService":
        if labels is None:
            return self

        file_name = f"{self.params["file_name_stock_basic"]}_{"_".join(labels)}"
        stock_array = self.datastore.load_original_data(file_name=file_name, file_type=self.params["file_type"])
        if stock_array is not None and len(stock_array) > 0:
            self.stock_array = stock_array
            return self

        stock_code_const = self.stock_service.get_stock_code_by_index_const(labels)
        stock_code_tx = self.stock_service.get_stock_code_by_index_member(labels)
        stock_code_ths = self.stock_service.get_stock_code_by_ths_labels(labels)

        df = pd.concat([stock_code_const, stock_code_ths, stock_code_tx], ignore_index=True)
        filtered_df = df[~df['stock_name'].str.contains('st|ST|退市', case=False, na=False)]
        filtered_df = filtered_df[filtered_df['stock_code'].str.startswith(('600', '00'))]

        self.stock_array = filtered_df.drop_duplicates(subset=['stock_code'])

        self.datastore.save_original_data(self.stock_array, file_name=file_name, file_type=self.params["file_type"])

        return self

    def set_stock_array(self, stock_array: list) -> "OverseeService":
        self.stock_array = stock_array
        return self

    def run(self, stock_code: str, start_date: str, end_date: str) -> "Processor":
        df = self._load_data(stock_code, start_date, end_date)
        processor = LSTMProcess(name=stock_code, data=df, batch_code=self.batch_code).run_simple()

        return processor

    @time_counter(logger_name=__name__)
    def _indicate_data(self, stock_code: str, data: pd.DataFrame) -> pd.DataFrame:
        self.logger.info(f"indicate stock:{stock_code} size:{len(data)}")

        cleaned_data: pd.DataFrame = self.indicator_service.prepare_price(data)
        cleaned_data: pd.DataFrame = self.indicator_service.generate_talib(cleaned_data)
        cleaned_data = DataClearService(prices=cleaned_data).covert_to_float().remove_nan_rows().normalize_date().prices

        return cleaned_data

    def _load_stock_basic(self) -> pd.DataFrame:
        stock_basic = self.datastore.load_original_data(file_name=self.params["file_name_stock_basic"], file_type=self.params["file_type"])
        if stock_basic is None:
            self.logger.info("load stock basic from database.")
            stock_basic = self.stock_service.get_all_stocks()
            self.datastore.save_original_data(data=stock_basic, file_name=self.params["file_name_stock_basic"], file_type=self.params["file_type"])

        return stock_basic

    def _load_ths_label(self) -> pd.DataFrame:

        flatted_stock_label = self.datastore.load_original_data(file_name=self.params["file_name_stock_label"], file_type=self.params["file_type"])
        if flatted_stock_label is None:
            self.logger.info("load stock labels from database.")
            stock_labels = self.stock_service.get_ths_labels()

            flatted_stock_label = stock_labels.groupby('stock_code').agg({
                'index_name': lambda x: list(x),
                'index_code': lambda x: list(x)
            }).reset_index()

            # 创建 index_code_str 列，将 index_code 列表拼接成一个字符串
            flatted_stock_label['index_code_str'] = flatted_stock_label['index_code'].apply(lambda x: ','.join(map(str, x)))

            self.datastore.save_original_data(data=flatted_stock_label, file_name=self.params["file_name_stock_label"], file_type=self.params["file_type"])

        return flatted_stock_label

    def _load_stock_price(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame | None:

        file_name = f"{self.params["file_name_stock_price"]}_{stock_code}_{start_date}_{end_date}"

        stock_df = self.datastore.load_original_data(file_name=file_name, file_type=self.params["file_type"])
        if stock_df is not None:
            return stock_df

        # self.logger.info(f"load stock price from database.{stock_code} from {start_date} to {end_date}.")
        stock_df: pd.DataFrame = self.stock_service.merge_price_and_adj_data(ts_code=stock_code, start_date=start_date, end_date=end_date)
        if stock_df is None:
            return pd.DataFrame()
        # stock_with_indicator = self._indicate_data(stock_code, stock_df)

        self.datastore.save_original_data(data=stock_df, file_name=file_name, file_type=self.params["file_type"])

        return stock_df

    @time_counter(logger_name=__name__)
    def load_data_with_label(self, start_date: str, end_date: str) -> pd.DataFrame:

        if self.stock_array is None or len(self.stock_array) == 0 or self.stock_array['stock_code'].empty:
            raise Exception("stock array is empty.")

        stock_code_array = self.stock_array["stock_code"].values.tolist()
        self.logger.info(f"stock code array: {len(stock_code_array)} {stock_code_array}")

        file_name = f"{self.params["file_name_stock_with_label"]}_{start_date}_{end_date}"
        stock_with_label = self.datastore.load_original_data(file_name=file_name, file_type=self.params["file_type"])
        if stock_with_label is not None:
            return stock_with_label

        self.logger.info("load stock price and label from database.")

        # 异步执行获取 stock labels
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_label = executor.submit(self._load_ths_label)

            with multiprocessing.Pool(processes=multiprocessing.cpu_count()) as pool:
                results = pool.starmap(self._load_stock_price, [(stock_code, start_date, end_date) for stock_code in stock_code_array])

                # 合并 stock price dataframe
                stock_df = pd.concat(results, ignore_index=True)

                future_indicator = executor.submit(self._indicate_data, "all", stock_df)

                # 等待 stock label 结果
                flatted_stock_label = future_label.result()

                stock_df = future_indicator.result()

                # 合并数据
                stock_df = stock_df.merge(flatted_stock_label, left_on='ts_code', right_on='stock_code', how='left')

        self.datastore.save_original_data(data=stock_df, file_name=file_name, file_type=self.params["file_type"])

        return stock_df

    def _load_data(self, stock_code: str, start_date: str, end_date: str) -> pd.DataFrame:
        df = self.stock_service.get_all_stocks()
        merged_data = self.stock_service.merge_price_and_adj_data(stock_code, start_date, end_date)
        cleaned_data: pd.DataFrame = self.indicator_service.prepare_price(merged_data)
        cleaned_data: pd.DataFrame = self.indicator_service.generate_talib(merged_data)
        cleaned_data = DataClearService(prices=cleaned_data).covert_to_float().remove_nan_rows().normalize_date().prices
        return cleaned_data
