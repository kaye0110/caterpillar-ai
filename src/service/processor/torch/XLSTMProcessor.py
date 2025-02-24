import os
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import torch
import torch.nn as nn
import torch.optim as optim
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import DataLoader, Dataset

from src.service.processor.v3.Processor import Processor
from src.tools.pycaterpillar_wrapper import time_counter


class StockDataset(Dataset):
    def __init__(self, data, time_step, predict_days):
        self.data = data
        self.time_step = time_step
        self.predict_days = predict_days

    def __len__(self):
        return len(self.data) - self.time_step - self.predict_days + 1

    def __getitem__(self, index):
        X = self.data[index:(index + self.time_step), :-1]
        y = self.data[(index + self.time_step):(index + self.time_step + self.predict_days), -1]
        return torch.tensor(X, dtype=torch.float32), torch.tensor(y, dtype=torch.float32)


class XLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(XLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, output_size)
        self.dropout = nn.Dropout(dropout)  # 添加 dropout 层

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out = self.dropout(out)  # 应用 dropout
        out = self.fc(out[:, -1, :])
        return out


class XLSTMProcess(Processor):

    def __init__(self, config):
        super().__init__(config)
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)

    def _create_dataset(self, data, time_step, predict_days):
        if time_step is None:
            time_step = self.config.n_timestep
        if predict_days is None:
            predict_days = self.config.n_predict

        dataset = StockDataset(data, time_step, predict_days)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

    # @time_counter(logger_name=__name__)
    def prepare(self, data: pd.DataFrame, ) -> "XLSTMProcess":
        # self.logger.info("Preparing data for LSTMProcessor...")

        super().prepare(data)

        # 选择特征和目标
        features = self.features.columns.tolist()

        # 划分训练集和测试集
        test_size = int(len(self.data) * self.config.test_size)

        X = self.data[features]
        y = self.targets

        # 拟合 selector
        selector = RandomForestRegressor(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, random_state=self.config.random_state, n_jobs=self.config.n_jobs)
        selector.fit(X[:test_size], y[:test_size])

        # 使用 SelectFromModel
        model = SelectFromModel(selector, prefit=True, threshold=1.5 * np.mean(selector.feature_importances_))

        # 获取选择的特征索引
        selected_feature_indexes = model.get_support(indices=True)
        self.selected_features = X.iloc[:, selected_feature_indexes].columns.tolist()

        # 数据预处理
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.feature_scaler.fit_transform(self.data[self.selected_features + [self.config.target_names[0]]])

        self.train_loader = self._create_dataset(scaled_data[:test_size], self.config.n_timestep, self.config.n_predict)
        self.test_loader = self._create_dataset(scaled_data[test_size:], self.config.n_timestep, self.config.n_predict)

        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler.min_, self.target_scaler.scale_ = self.feature_scaler.min_[-1], self.feature_scaler.scale_[-1]

        self.store.save_model(data=self.config.serialize(), file_name=self.store.get_config_name(self.config.batch_code, self.stock_code), file_type="json")
        self.store.save_model(data=self.feature_scaler, file_name=self.store.get_feature_scaler_name(self.config.batch_code, self.stock_code))
        self.store.save_model(data=self.target_scaler, file_name=self.store.get_target_scaler_name(self.config.batch_code, self.stock_code))
        self.store.save_model_feature(data=self.selected_features, file_name=self.store.get_selected_features_name(self.config.batch_code, self.stock_code))

        return self

    def build_model(self) -> "XLSTMProcess":
        input_size = len(self.selected_features)
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_layers
        output_size = self.config.n_predict
        dropout = self.config.dropout

        self.model = XLSTM(input_size, hidden_size, num_layers, output_size, dropout).to(self.config.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        return self

    # @time_counter(logger_name=__name__)
    def train(self) -> "XLSTMProcess":
        # self.logger.info("Start to train LSTM model with hyperparameter tuning.")

        self.model.train()
        for epoch in range(self.config.epochs):
            for X_batch, y_batch in self.train_loader:
                X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(self.config.device)
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)
                loss.backward()
                self.optimizer.step()
            # self.logger.info(f'Epoch [{epoch + 1}/{self.config.epochs}], Loss: {loss.item():.4f}')

        self.store.save_model(data=self.model, file_name=self.store.get_model_name(self.config.batch_code, self.stock_code))

        return self

    def predict(self, future_data):
        self.model.eval()
        future_scaled_data = self.feature_scaler.transform(future_data[self.selected_features + [self.config.target_names[0]]])

        data_length = len(future_scaled_data)
        required_length = self.config.n_timestep + self.config.n_predict - 1

        # if data_length <= required_length:
        #     raise ValueError(f"Data length ({data_length}) is not sufficient for the given time_step ({self.config.n_timestep}) and predict_days ({self.config.n_predict}).")

        future_dataset = StockDataset(future_scaled_data, self.config.n_timestep, self.config.n_predict)
        future_loader = DataLoader(future_dataset, batch_size=self.config.batch_size, shuffle=False)

        all_predictions = []

        with torch.no_grad():
            for X_batch, _ in future_loader:
                X_batch = X_batch.to(self.config.device)
                outputs = self.model(X_batch)
                future_predict_value = self.target_scaler.inverse_transform(outputs.cpu().numpy())
                # self.logger.info(f'Future Predictions: {future_predict_value}')
                all_predictions.append(future_predict_value)

        if all_predictions:
            return np.concatenate(all_predictions, axis=0)

        return None

    def test(self) -> "XLSTMProcess":
        # self.logger.info("Strat to test LSTM model.")

        self.model.eval()
        with torch.no_grad():
            predictions, actuals = [], []
            for X_batch, y_batch in self.test_loader:
                X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(self.config.device)
                outputs = self.model(X_batch)
                predictions.append(outputs.cpu().numpy())
                actuals.append(y_batch.cpu().numpy())

        predictions = np.concatenate(predictions, axis=0)
        actuals = np.concatenate(actuals, axis=0)

        predictions = self.target_scaler.inverse_transform(predictions)
        actuals = self.target_scaler.inverse_transform(actuals)

        self.result_mse = mean_squared_error(actuals, predictions)
        self.result_mae = mean_absolute_error(actuals, predictions)
        self.result_r2 = r2_score(actuals, predictions)

        # self.logger.info(f'MSE: {self.result_mse:.4f}, MAE: {self.result_mae:.4f}, R2: {self.result_r2:.4f}')

        self.visualize_results(actuals, predictions, self.config.n_predict)

        return self

    def report(self) -> "XLSTMProcess":
        # self.logger.info("Model Summary:")
        # self.logger.info(self.model.summary())

        return self

    def visualize_results(self, actuals, predictions, n_predict):
        # 计算残差
        residuals = actuals - predictions

        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f"Residuals for {n_predict} Days", "Prediction Error Distribution", "Time Series Prediction"),
            vertical_spacing=0.1
        )

        # 添加残差图
        for i in range(n_predict):
            fig.add_trace(go.Scatter(
                x=np.arange(len(residuals)),
                y=residuals[:, i],
                mode='lines',
                name=f'Day {i + 1} Residuals'
            ), row=1, col=1)

        # 添加预测误差分布图
        for i in range(n_predict):
            fig.add_trace(go.Histogram(
                x=residuals[:, i],
                name=f'Day {i + 1} Error',
                opacity=0.5
            ), row=2, col=1)

        # 添加时间序列预测图
        for i in range(n_predict):
            fig.add_trace(go.Scatter(
                x=np.arange(len(actuals)),
                y=actuals[:, i],
                mode='lines',
                name=f'Day {i + 1} True'
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=np.arange(len(predictions)),
                y=predictions[:, i],
                mode='lines',
                name=f'Day {i + 1} Predicted'
            ), row=3, col=1)

        # 更新布局
        fig.update_layout(
            title_text="Interactive Visualization of Predictions",
            height=900,
            showlegend=True,
            legend_title='Legend',
            hovermode='x unified'  # 支持鼠标跟随十字线
        )

        # 保存为 HTML 文件
        fig_path = os.path.join(self.store.get_predict_result_path(), self.store.get_plotly_name(f"xlstm_test_plotly", self.stock_code))
        fig.write_html(fig_path)
