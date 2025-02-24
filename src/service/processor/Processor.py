import logging

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from src.infra.repo.DataStore import DataStore
from src.tools.pycaterpillar_wrapper import time_counter


class Processor:
    params = {"random_state": 42,
              "test_size": 0.3,
              "n_estimators": 100,
              "max_depth": 100,
              "min_samples_split": 2,
              "min_samples_leaf": 1,
              "bootstrap": True,
              "indicator_prefix": "indicator_",
              "feature_prefix": "feature_",
              "target_columns": ["close"]}

    def __init__(self, name: str, data: pd.DataFrame, batch_code: str):
        self.logger = logging.getLogger(__name__)
        self.name = name
        self.data = data
        self.batch_code = batch_code
        self.store: DataStore = DataStore(batch_code)
        self.model = None
        self.indicators = None
        self.features = None
        self.targets = None
        self.selected_features_indices = None

        self.X_train = None
        self.X_train_selected = None

        self.X_test = None
        self.X_test_selected = None

        self.y_train = None
        self.y_test = None

        self.y_predict = None

    @time_counter(logger_name=__name__)
    def run_simple(self) -> "Processor":
        return self._save_data().load().split_train_test_data().model_init().model_train().model_test().model_report()

    def _save_data(self) -> "Processor":
        if self.data is None:
            raise "No data to save."

        self.store.save_original_data(data=self.data, file_name=self.name)
        return self

    def load(self) -> "Processor":
        """Load data from a parquet file."""

        self.indicators = self.data.filter(like=self.params["indicator_prefix"])
        self.features = self.data.filter(like=self.params["feature_prefix"])
        self.targets = self.data[self.params["target_columns"]]

        self.logger.info("Data loaded successfully")

        return self

    @time_counter(logger_name=__name__)
    def split_train_test_data(self, use_feature: bool = True) -> "Processor":
        """Split the data into training and testing sets."""
        self.logger.info("Splitting data into train and test sets")

        def create_time_window_data(df, features, target, window_size=10, forecast_horizon=3):
            X, y = [], []
            for i in range(len(df) - window_size - forecast_horizon + 1):
                # 确保使用整数索引
                X.append(df.iloc[i:i + window_size][features].values.flatten())
                y.append(df.iloc[i + window_size:i + window_size + forecast_horizon][target].values.flatten())
            return np.array(X), np.array(y)

        X, y = create_time_window_data(self.data, self.features.columns, self.targets.columns.tolist()[0])

        if use_feature:
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.params["test_size"], random_state=self.params["random_state"], shuffle=False)
        else:
            scaler = StandardScaler()
            x_scaled = scaler.fit_transform(self.indicators)
            self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(x_scaled, self.targets, test_size=self.params["test_size"], random_state=self.params["random_state"])

        self.logger.info("Data split completed")

        return self

    def model_init(self) -> "Processor":
        """Initialize the model."""
        raise NotImplementedError("Subclasses should implement this method.")

    def model_train(self) -> "Processor":
        """Train the model."""
        raise NotImplementedError("Subclasses should implement this method.")

    def model_test(self) -> "Processor":
        """Test the model."""
        raise NotImplementedError("Subclasses should implement this method.")

    def model_report(self) -> "Processor":
        """Generate a report of the model's performance."""
        raise NotImplementedError("Subclasses should implement this method.")

    def model_predict(self, x) -> "Processor":
        """Make predictions with the model."""
        raise NotImplementedError("Subclasses should implement this method.")
