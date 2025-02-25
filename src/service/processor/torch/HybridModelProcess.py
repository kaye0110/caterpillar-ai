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


class OptimizedHybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(OptimizedHybridModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, dropout=dropout, batch_first=True, bidirectional=True)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        h_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        c_0 = torch.zeros(self.lstm.num_layers * 2, x.size(0), self.lstm.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h_0, c_0))
        out, _ = self.gru(out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


class HybridModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, dropout):
        super(HybridModel, self).__init__()
        self.xlstm = XLSTM(input_size, hidden_size, num_layers, output_size, dropout)
        self.gru = nn.GRU(hidden_size * 2, hidden_size, num_layers, dropout=dropout, batch_first=True)
        self.indrnn = nn.RNN(hidden_size, hidden_size, num_layers, nonlinearity='relu', batch_first=True)
        self.transformer = nn.Transformer(hidden_size, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dropout=dropout)
        self.fc = nn.Linear(hidden_size, output_size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = self.xlstm.forward(x)
        out, _ = self.gru(out)
        out, _ = self.indrnn(out)
        out = self.transformer(out, out)
        out = self.dropout(out)
        out = self.fc(out[:, -1, :])
        return out


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


class HybridModelProcess(Processor):
    def __init__(self, config):
        super().__init__(config)
        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        torch.manual_seed(self.config.random_state)

        self.model = None
        self.criterion = None
        self.optimizer = None
        self.train_loader = None
        self.test_loader = None
        self.feature_scaler = None
        self.target_scaler = None
        self.selected_features = None

    def _create_dataset(self, data, time_step, predict_days):
        if time_step is None:
            time_step = self.config.n_timestep
        if predict_days is None:
            predict_days = self.config.n_predict

        dataset = StockDataset(data, time_step, predict_days)
        return DataLoader(dataset, batch_size=self.config.batch_size, shuffle=False)

    def prepare(self, data: pd.DataFrame) -> "HybridModelProcess":

        super().prepare(data)

        # 选择特征和目标
        features = self.features.columns.tolist()

        test_size = int(len(data) * self.config.test_size)

        X = self.data[features]
        y = self.targets

        # 拟合 selector
        selector = RandomForestRegressor(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, random_state=self.config.random_state, n_jobs=self.config.n_jobs)
        selector.fit(X[:test_size], y[:test_size])

        # 如果指标数量太小的话，模型效果较差
        for rate in self.config.feature_rates:
            # 使用 SelectFromModel
            model = SelectFromModel(selector, prefit=True, threshold=rate * np.mean(selector.feature_importances_))

            # 获取选择的特征索引
            selected_feature_indexes = model.get_support(indices=True)
            self.selected_features = X.iloc[:, selected_feature_indexes].columns.tolist()
            if len(self.selected_features) >= self.config.feature_size_min:
                break

        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.feature_scaler.fit_transform(self.data[self.selected_features + [self.config.target_names[0]]])

        self.train_loader = self._create_dataset(scaled_data[:test_size], self.config.n_timestep, self.config.n_predict)
        self.test_loader = self._create_dataset(scaled_data[test_size:], self.config.n_timestep, self.config.n_predict)

        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler.min_, self.target_scaler.scale_ = self.feature_scaler.min_[-1], self.feature_scaler.scale_[-1]

        # # 数据预处理
        # self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        # scaled_data = self.feature_scaler.fit_transform(self.data[self.selected_features + [self.config.target_names[0]]])
        #
        # self.train_loader = self._create_dataset(scaled_data[:test_size], self.config.n_timestep, self.config.n_predict)
        # self.test_loader = self._create_dataset(scaled_data[test_size:], self.config.n_timestep, self.config.n_predict)
        #
        # self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        # self.target_scaler.min_, self.target_scaler.scale_ = self.feature_scaler.min_[-1], self.feature_scaler.scale_[-1]
        #

        self.store.save_model(data=self.config.serialize(), file_name=self.store.get_config_name(self.config.batch_code, self.stock_code), file_type="json")
        self.store.save_model(data=self.feature_scaler, file_name=self.store.get_feature_scaler_name(self.config.batch_code, self.stock_code))
        self.store.save_model(data=self.target_scaler, file_name=self.store.get_target_scaler_name(self.config.batch_code, self.stock_code))
        self.store.save_model_feature(data=self.selected_features, file_name=self.store.get_selected_features_name(self.config.batch_code, self.stock_code))

        return self

    def build_model(self) -> "HybridModelProcess":

        input_size = len(self.selected_features)
        hidden_size = self.config.hidden_size
        num_layers = self.config.num_layers
        output_size = self.config.n_predict
        dropout = self.config.dropout

        self.model = OptimizedHybridModel(input_size, hidden_size, num_layers, output_size, dropout).to(self.config.device)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.config.learning_rate)

        return self

    def train(self) -> "HybridModelProcess":
        self.model.train()
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5, verbose=True)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, 'min', patience=5, factor=0.5)
        for epoch in range(self.config.epochs):
            epoch_loss = 0
            for X_batch, y_batch in self.train_loader:
                # 将数据移动到目标设备（CPU/GPU）
                X_batch, y_batch = X_batch.to(self.config.device), y_batch.to(self.config.device)

                # 前向传播
                self.optimizer.zero_grad()
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

                # 反向传播和优化
                loss.backward()
                self.optimizer.step()
                epoch_loss += loss.item()

            # 在每个epoch结束时更新学习率调度器
            scheduler.step(epoch_loss)
            # 打印当前epoch的损失和学习率
            avg_epoch_loss = epoch_loss / len(self.train_loader)
            current_lr = self.optimizer.param_groups[0]['lr']
            # self.logger.info(f'Epoch {epoch + 1}/{self.config.epochs}, Loss: {avg_epoch_loss}, Learning Rate: {current_lr}')
            # print(f'Epoch {epoch + 1}/{self.config.epochs}, Loss: {epoch_loss / len(self.train_loader)}')

        self.store.save_model(data=self.model, file_name=self.store.get_model_name(self.config.batch_code, self.stock_code))

        return self

    def predict(self, future_data):
        self.model.eval()
        future_scaled_data = self.feature_scaler.transform(future_data[self.selected_features + [self.config.target_names[0]]])

        future_dataset = StockDataset(future_scaled_data, self.config.n_timestep, self.config.n_predict)
        future_loader = DataLoader(future_dataset, batch_size=self.config.batch_size, shuffle=False)

        all_predictions = []

        with torch.no_grad():
            for X_batch, _ in future_loader:
                X_batch = X_batch.to(self.config.device)
                outputs = self.model(X_batch)
                future_predict_value = self.target_scaler.inverse_transform(outputs.cpu().numpy())
                all_predictions.append(future_predict_value)

        if all_predictions:
            return np.concatenate(all_predictions, axis=0)

        return None

    def test(self) -> "HybridModelProcess":
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

        # Calculate existing metrics
        self.result_mse = mean_squared_error(actuals, predictions)
        self.result_mae = mean_absolute_error(actuals, predictions)
        self.result_r2 = r2_score(actuals, predictions)

        # Calculate additional metrics
        self.result_rmse = np.sqrt(self.result_mse)
        self.result_mape = np.mean(np.abs((actuals - predictions) / actuals)) * 100
        self.result_smape = np.mean(2.0 * np.abs(actuals - predictions) / (np.abs(actuals) + np.abs(predictions))) * 100
        self.result_explained_variance = 1 - np.var(actuals - predictions) / np.var(actuals)

        # Adjusted R² calculation
        n = len(actuals)
        p = predictions.shape[1]  # Number of features
        self.result_adjusted_r2 = 1 - (1 - self.result_r2) * (n - 1) / (n - p - 1)

        # Calculate composite score
        self.composite_score = self.evaluate_model_performance(
            self.result_mse, self.result_mae, self.result_r2, self.result_rmse,
            self.result_mape, self.result_smape, self.result_explained_variance, self.result_adjusted_r2
        )

        # Print or log the results
        self.logger.info(f'MSE: {self.result_mse}')
        self.logger.info(f'MAE: {self.result_mae}')
        self.logger.info(f'R²: {self.result_r2}')
        self.logger.info(f'RMSE: {self.result_rmse}')
        self.logger.info(f'MAPE: {self.result_mape}')
        self.logger.info(f'SMAPE: {self.result_smape}')
        self.logger.info(f'Explained Variance: {self.result_explained_variance}')
        self.logger.info(f'Adjusted R²: {self.result_adjusted_r2}')
        self.logger.info(f'Composite Score: {self.composite_score}')

        self.visualize_results(actuals, predictions, self.config.n_predict)

        return self

    def evaluate_model_performance(self, mse, mae, r2, rmse, mape, smape, explained_variance, adjusted_r2):
        """
        计算模型的综合得分，用于选择最优模型。
        """
        # 定义每个指标的权重
        weights = {
            'mse': 0.15,
            'mae': 0.15,
            'r2': 0.2,
            'rmse': 0.15,
            'mape': 0.1,
            'smape': 0.1,
            'explained_variance': 0.1,
            'adjusted_r2': 0.05
        }

        # 计算综合得分
        composite_score = (
                weights['mse'] * (1 / (1 + mse)) +  # 取倒数以便于最小化
                weights['mae'] * (1 / (1 + mae)) +  # 取倒数以便于最小化
                weights['r2'] * r2 +  # R² 越大越好
                weights['rmse'] * (1 / (1 + rmse)) +  # 取倒数以便于最小化
                weights['mape'] * (1 / (1 + mape)) +  # 取倒数以便于最小化
                weights['smape'] * (1 / (1 + smape)) +  # 取倒数以便于最小化
                weights['explained_variance'] * explained_variance +  # Explained Variance 越大越好
                weights['adjusted_r2'] * adjusted_r2  # Adjusted R² 越大越好
        )

        return composite_score

    def visualize_results(self, actuals, predictions, n_predict):
        residuals = actuals - predictions

        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f"Residuals for {n_predict} Days", "Prediction Error Distribution", "Time Series Prediction"),
            vertical_spacing=0.1
        )

        for i in range(n_predict):
            fig.add_trace(go.Scatter(
                x=np.arange(len(residuals)),
                y=residuals[:, i],
                mode='lines',
                name=f'Day {i + 1} Residuals'
            ), row=1, col=1)

        for i in range(n_predict):
            fig.add_trace(go.Histogram(
                x=residuals[:, i],
                name=f'Day {i + 1} Error',
                opacity=0.5
            ), row=2, col=1)

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

        fig.update_layout(
            title_text="Interactive Visualization of Predictions",
            height=900,
            showlegend=True,
            legend_title='Legend',
            hovermode='x unified'
        )

        fig_path = os.path.join(self.store.get_predict_result_path(), self.store.get_plotly_name(f"hybrid_model_test_plotly", self.stock_code))
        fig.write_html(fig_path)

    def report(self) -> "HybridModelProcess":
        return self
