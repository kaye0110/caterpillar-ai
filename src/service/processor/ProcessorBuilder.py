from src.infra.repo.DataStore import DataStore
from src.service.processor.torch.HybridModelProcess import HybridModelProcess
from src.service.processor.torch.XLSTMProcessor import XLSTMProcess
from src.service.processor.v3.Config import Configuration
from src.service.processor.v3.LSTMProcessor import LSTMProcess
from src.service.processor.v3.Processor import Processor


class ProcessorBuilder:

    @staticmethod
    def build_by_batch_code(batch_code: str, stock_code: str = "all") -> "Processor":
        datastore = DataStore(batch_code)
        config_json = datastore.load_model(datastore.get_config_name(batch_code, stock_code), file_type="json")
        config = Configuration.deserialize(config_json)

        if "LSTM" == config.model_type:
            processor = LSTMProcess(config)

        elif "XLSTM" == config.model_type:
            processor = XLSTMProcess(config)

        elif "hybrid" == config.model_type:
            processor = HybridModelProcess(config)

        else:
            raise Exception("Not implemented")

        processor.model = datastore.load_model(datastore.get_model_name(batch_code, stock_code), file_type="pkl")
        processor.feature_scaler = datastore.load_model(datastore.get_feature_scaler_name(batch_code, stock_code), file_type="pkl")
        processor.target_scaler = datastore.load_model(datastore.get_target_scaler_name(batch_code, stock_code), file_type="pkl")
        processor.selected_features = datastore.load_model(datastore.get_selected_features_name(batch_code, stock_code), file_type="json")

        return processor
