import json
import logging
import os
from typing import Any

import joblib
import pandas as pd


class DataStore:
    params = {"export_path": "export",
              "original_data": "original_data",
              "cleaned_data": "cleaned_data",
              "model_data": "model_data",
              "model": "model",
              "predict_result": "predict_result",
              }

    def __init__(self, batch_code: str):
        self.logger = logging.getLogger(__name__)
        self.batch_code = batch_code
        self.export_path = None
        self.original_data_path = None
        self.cleaned_data_path = None
        self.model_data_path = None
        self.model_path = None
        self.predict_result_path = None

        self._generate_export_directory()

    def get_predict_result_path(self):
        return self.predict_result_path

    def save_model_feature(self, data: Any, file_name: str, file_type: str = "json") -> "DataStore":
        return self._save_data(data, file_name, "model", file_type)

    def save_model(self, data: Any, file_name: str, file_type: str = "pkl") -> "DataStore":
        return self._save_data(data, file_name, "model", file_type)

    def save_original_data(self, data: pd.DataFrame, file_name: str, file_type: str = "parquet") -> "DataStore":
        return self._save_data(data, file_name, "original", file_type)

    def save_cleaned_data(self, data: pd.DataFrame, file_name: str, file_type: str = "parquet") -> "DataStore":
        return self._save_data(data, file_name, "cleaned", file_type)

    def save_model_data(self, data: pd.DataFrame, file_name: str, file_type: str = "parquet") -> "DataStore":
        return self._save_data(data, file_name, "model_data", file_type)

    def save_predict_result(self, data: pd.DataFrame, file_name: str, file_type: str = "parquet") -> "DataStore":
        return self._save_data(data, file_name, "predict_result", file_type)

    def load_model(self, file_name: str, file_type: str = "pkl") -> Any:
        return self._load_data(file_name, "model", file_type)

    def load_original_data(self, file_name: str, file_type: str = "parquet") -> pd.DataFrame:
        return self._load_data(file_name, "original", file_type)

    def load_predict_result(self, file_name, file_type: str = "parquet"):
        return self._load_data(file_name, "predict_result", file_type)

    def _load_data(self, file_name: str, file_kind: str, file_type: str = "parquet"):
        # self.logger.info(f"Loading {file_kind} data from {file_name}.{file_type}")
        try:
            file_path = None
            if file_kind == "model" and file_type == "pkl":
                file_path = os.path.join(self.model_path, f"{file_name}.{file_type}")
                with open(file_path, 'rb') as file:
                    return joblib.load(file)

            elif file_kind == "model":
                file_path = os.path.join(self.model_path, f"{file_name}.{file_type}")
            elif file_kind == "original":
                file_path = os.path.join(self.original_data_path, f"{file_name}.{file_type}")
            elif file_kind == "cleaned":
                file_path = os.path.join(self.cleaned_data_path, f"{file_name}.{file_type}")
            elif file_kind == "model_data":
                file_path = os.path.join(self.model_data_path, f"{file_name}.{file_type}")
            elif file_kind == "predict_result":
                file_path = os.path.join(self.predict_result_path, f"{file_name}.{file_type}")
            elif file_kind == "model_feature":
                file_path = os.path.join(self.model_path, f"{file_name}.{file_type}")
            else:
                raise ValueError("Invalid file kind")

            if file_type == "parquet":
                return pd.read_parquet(file_path)
            elif file_type == "csv":
                return pd.read_csv(file_path)
            elif file_type == "json":
                with open(file_path, 'r', encoding='utf-8') as file:
                    return json.load(file)
            else:
                raise ValueError("Invalid file type")
        except Exception as e:
            self.logger.debug(f"Error loading data: file_name:{file_name} file_type:{file_type} file_kind:{file_kind} file_path:{file_path} {e}")
            return None

    def _save_data(self, data, file_name: str, file_kind: str, file_type: str = "parquet") -> "DataStore":
        if file_kind == "model" and file_type == "pkl":
            file_path = os.path.join(self.model_path, f"{file_name}.{file_type}")
            joblib.dump(data, file_path)
            # self.logger.info("Model saved successfully")

            return self
        elif file_kind == "model":
            file_path = os.path.join(self.model_path, f"{file_name}.{file_type}")
        elif file_kind == "original":
            file_path = os.path.join(self.original_data_path, f"{file_name}.{file_type}")
        elif file_kind == "cleaned":
            file_path = os.path.join(self.cleaned_data_path, f"{file_name}.{file_type}")
        elif file_kind == "model_data":
            file_path = os.path.join(self.model_data_path, f"{file_name}.{file_type}")
        elif file_kind == "predict_result":
            file_path = os.path.join(self.predict_result_path, f"{file_name}.{file_type}")
        elif file_kind == "model_feature":
            file_path = os.path.join(self.model_path, f"{file_name}.{file_type}")
        else:
            raise ValueError("Invalid file kind")

        if file_type == "parquet":
            data.to_parquet(file_path, index=False)
        elif file_type == "csv":
            data.to_csv(file_path, index=False)
        elif file_type == "json":
            with open(file_path, "w", encoding="utf-8") as file:
                json.dump(data, file, ensure_ascii=False, indent=4)
        else:
            raise ValueError("Invalid file type")

        return self

    def _generate_export_directory(self) -> "DataStore":
        """
        初始化数据导出目录
        :return: OverseeService instance self
        """
        project_path = DataStore.get_root_path()
        self.export_path = os.path.join(project_path, self.params["export_path"], self.batch_code)
        if not os.path.exists(self.export_path):
            os.makedirs(self.export_path)

        self.original_data_path = os.path.join(self.export_path, self.params["original_data"])
        if not os.path.exists(self.original_data_path):
            os.makedirs(self.original_data_path)

        self.cleaned_data_path = os.path.join(self.export_path, self.params["cleaned_data"])
        if not os.path.exists(self.cleaned_data_path):
            os.makedirs(self.cleaned_data_path)

        self.model_data_path = os.path.join(self.export_path, self.params["model_data"])
        if not os.path.exists(self.model_data_path):
            os.makedirs(self.model_data_path)

        self.model_path = os.path.join(self.export_path, self.params["model"])
        if not os.path.exists(self.model_path):
            os.makedirs(self.model_path)

        self.predict_result_path = os.path.join(self.export_path, self.params["predict_result"])
        if not os.path.exists(self.predict_result_path):
            os.makedirs(self.predict_result_path)

        return self

    @staticmethod
    def get_root_path() -> str:
        return os.path.abspath(os.path.join(os.path.dirname(__file__), '../../../'))

    @staticmethod
    def export_to_parquet(df: pd.DataFrame, path: str, file_name: str):
        file_path = os.path.join(path, file_name)
        df.to_parquet(file_path, index=False)

    @staticmethod
    def get_plotly_name(prefix: str, stock_code: str = "all"):
        return f'{prefix}_{stock_code}_plotly.html'

    @staticmethod
    def get_config_name(prefix: str, stock_code: str = "all"):
        return f'{prefix}_{stock_code}_config'

    @staticmethod
    def get_model_name(prefix: str, stock_code: str = "all"):
        return f'{prefix}_{stock_code}_model'

    @staticmethod
    def get_feature_scaler_name(prefix: str, stock_code: str = "all"):
        return f'{prefix}_{stock_code}_feature_scaler'

    @staticmethod
    def get_target_scaler_name(prefix: str, stock_code: str = "all"):
        return f'{prefix}_{stock_code}_target_scaler'

    @staticmethod
    def get_selected_features_name(prefix: str, stock_code: str = "all"):
        return f'{prefix}_{stock_code}_selected_features'

    def get_original_data_path(self, file_name, file_type: str = "parquet") -> str | None:
        file_path = self.generate_original_data_path(file_name, file_type=file_type)

        if os.path.exists(file_path):
            return str(file_path)

        return None

    def generate_original_data_path(self, file_name, file_type: str = "parquet") -> str | None:
        return self.get_data_path(file_name, file_kind="original", file_type=file_type)

    def get_data_path(self, file_name: str, file_kind: str, file_type: str = "parquet") -> str:
        if file_kind == "model" and file_type == "pkl":
            return os.path.join(self.model_path, f"{file_name}.{file_type}")
        elif file_kind == "model":
            return os.path.join(self.model_path, f"{file_name}.{file_type}")
        elif file_kind == "original":
            return os.path.join(self.original_data_path, f"{file_name}.{file_type}")
        elif file_kind == "cleaned":
            return os.path.join(self.cleaned_data_path, f"{file_name}.{file_type}")
        elif file_kind == "model_data":
            return os.path.join(self.model_data_path, f"{file_name}.{file_type}")
        elif file_kind == "predict_result":
            return os.path.join(self.predict_result_path, f"{file_name}.{file_type}")
        elif file_kind == "model_feature":
            return os.path.join(self.model_path, f"{file_name}.{file_type}")
        else:
            raise ValueError("Invalid file kind")
