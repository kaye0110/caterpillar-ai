import logging

import pandas as pd

from src.infra.repo.DataStore import DataStore
from src.service.processor.v3.Config import Configuration
from src.tools.pycaterpillar_wrapper import time_counter


class Processor:

    def __init__(self, config: Configuration):
        self.logger = logging.getLogger(__name__)
        self.config = config
        self.store: DataStore = DataStore(config.batch_code)

        self.data = None
        self.indicators = None
        self.features = None
        self.targets = None

        self.selected_features = None

        self.stock_code = "all"

        self.feature_scaler = None
        self.target_scaler = None
        self.model = None

        self.X, self.y = None, None
        self.X_train = None
        self.X_test = None

        self.y_train = None
        self.y_test = None

        self.predict_result = None
        self.predict_value = None
        self.y_test_actual = None
        self.future_predict_result = None
        self.future_predict_value = None

    # @time_counter(logger_name=__name__)
    def prepare(self, data: pd.DataFrame, ) -> "Processor":
        self.data = data
        self.store.save_original_data(data=self.data, file_name="original_data")

        self.indicators = self.data.filter(like=self.config.indicator_prefix)
        self.features = self.data.filter(like=self.config.feature_prefix)
        self.targets = self.data[self.config.target_names[0]]

        return self

    def train(self) -> "Processor":
        raise NotImplementedError("Subclasses should implement this method.")

    def test(self) -> "Processor":
        raise NotImplementedError("Subclasses should implement this method.")

    def report(self) -> "Processor":
        raise NotImplementedError("Subclasses should implement this method.")

    def predict(self, future_data: pd.DataFrame) -> "Processor":
        raise NotImplementedError("Subclasses should implement this method.")
