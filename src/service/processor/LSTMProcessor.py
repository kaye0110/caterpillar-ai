import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.models import Sequential

from src.service.processor.Processor import Processor
from src.tools.pycaterpillar_wrapper import time_counter


class LSTMProcess(Processor):

    def __init__(self, name: str, data: pd.DataFrame, batch_code: str):
        super().__init__(name, data, batch_code)
        self.scaler = None

    @staticmethod
    def create_dataset(data, time_step=10):
        """
        准备训练数据
        :param data: pd.DataFrame
        :param time_step: 预测结果的时间窗口
        :return:
        """
        # 确保输入数据是 NumPy 数组
        if isinstance(data, pd.DataFrame):
            data = data.values

        X, y = [], []
        for i in range(len(data) - time_step - 10):
            X.append(data[i:(i + time_step), :-1])  # 选择特征列
            y.append(data[i + time_step:i + time_step + 10, -1])  # 选择目标列
        return np.array(X), np.array(y)

    @time_counter(logger_name=__name__)
    def split_train_test_data(self, use_feature: bool = True) -> "LSTMProcess":
        """Split the data into training and testing sets."""
        self.logger.info("Splitting data into train and test sets")

        # 数据归一化
        self.scaler = MinMaxScaler()
        scaled_data = self.scaler.fit_transform(self.indicators)

        X, y = LSTMProcess.create_dataset(data=scaled_data)

        # 划分训练和测试集 8:2
        train_size = int(len(X) * (1 - self.params["test_size"]))
        self.X_train, self.X_test = X[:train_size], X[train_size:]
        self.y_train, self.y_test = y[:train_size], y[train_size:]

        self.logger.info("Data split completed")

        return self

    @time_counter(logger_name=__name__)
    def model_init(self) -> "LSTMProcess":

        # 构建 LSTM 模型
        self.model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2])),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(10)
        ])

        self.model.compile(optimizer='adam', loss='mean_squared_error')
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, validation_data=(self.X_test, self.y_test), verbose=1)

        return self

    @time_counter(logger_name=__name__)
    def model_train(self) -> "LSTMProcess":
        """Train the RandomForest model."""
        self.logger.info("Training RandomForest model")

        self.store.save_model(data=self.model, file_name=self.name + "_LSTM_Model")

        self.logger.info("Model training completed")

        return self

    @time_counter(logger_name=__name__)
    def model_test(self) -> "LSTMProcess":
        """Test the RandomForest model."""
        self.logger.info("Testing RandomForest model")
        self.y_predict = self.model.predict(self.X_test)

        return self

    @time_counter(logger_name=__name__)
    def model_report(self) -> "LSTMProcess":
        """Generate a report of the RandomForest model's performance."""
        self.logger.info("Generating model performance report")

        # 反归一化
        # 检查形状
        print("X_test shape:", self.X_test.shape)
        print("y_test shape:", self.y_test.shape)

        # 反归一化
        # 假设 self.scaler 是一个已经拟合的 MinMaxScaler 或 StandardScaler
        y_test_rescaled = self.scaler.inverse_transform(self.X_test)
        y_pred_rescaled = self.scaler.inverse_transform(self.y_predict)

        # 计算误差指标
        mse = mean_squared_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
        mae = mean_absolute_error(y_test_rescaled.flatten(), y_pred_rescaled.flatten())
        rmse = np.sqrt(mse)

        self.logger.info(f'Mean Squared Error: {mse:.4f}')
        self.logger.info(f'Mean Absolute Error: {mae:.4f}')
        self.logger.info(f'Root Mean Squared Error: {rmse:.4f}')

        # 计算残差
        residuals = y_test_rescaled - y_pred_rescaled

        # 绘制残差图
        fig_residuals = px.line(x=np.arange(residuals.shape[0]), y=residuals.flatten(), title='Residuals')
        fig_residuals.show()

        # 绘制特征图
        fig_features = px.imshow(self.X_test[0], title='Feature Map of First Test Sample')
        fig_features.show()

        # 绘制预测误差分布图
        fig_error_dist = px.histogram(residuals.flatten(), nbins=50, title='Prediction Error Distribution')
        fig_error_dist.show()

        # 绘制时间序列预测图
        fig_time_series = go.Figure()

        # 选择 10 个样本进行可视化
        num_samples = 10
        sample_indices = np.random.choice(range(y_test_rescaled.shape[0]), num_samples, replace=False)

        for idx in sample_indices:
            fig_time_series.add_trace(go.Scatter(x=np.arange(10), y=y_test_rescaled[idx], mode='lines', name=f'Actual {idx}'))
            fig_time_series.add_trace(go.Scatter(x=np.arange(10), y=y_pred_rescaled[idx], mode='lines', name=f'Predicted {idx}'))

        fig_time_series.update_layout(title='Time Series Prediction for 10 Samples', height=1200, showlegend=True)

        # 保存为 HTML 文件
        fig_time_series.write_html("model_analysis.html")

        # 显示图表
        fig_time_series.show()

        return self

    @time_counter(logger_name=__name__)
    def model_predict(self, x):
        """Make predictions with the RandomForest model."""
        self.logger.info("Making predictions with RandomForest model")
        return self.model.predict(x)
