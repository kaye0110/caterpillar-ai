import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential
from tensorflow.keras.optimizers import RMSprop
from tensorflow.python.keras.callbacks import ReduceLROnPlateau

from src.service.processor.Processor import Processor
from src.tools.pycaterpillar_wrapper import time_counter


class LSTMProcess(Processor):

    def __init__(self, name: str, data: pd.DataFrame, batch_code: str):
        super().__init__(name, data, batch_code)
        self.scaler = None
        self.close_scaler = None

    @time_counter(logger_name=__name__)
    def split_train_test_data(self, use_feature: bool = True) -> "LSTMProcess":

        random.seed(self.params["random_state"])
        np.random.seed(self.params["random_state"])
        tf.random.set_seed(self.params["random_state"])

        # 选择特征和目标
        features = self.data.filter(like='feature_').columns.tolist()
        target = 'close'

        X = self.data[features]
        y = self.data[target]
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=self.params["test_size"], random_state=self.params["random_state"], shuffle=False)
        # 拟合 selector
        selector = RandomForestRegressor(n_estimators=self.params["n_estimators"], max_depth=self.params["max_depth"], random_state=self.params["random_state"], n_jobs=-1)
        selector.fit(self.X_train, self.y_train)

        # 使用 SelectFromModel
        model = SelectFromModel(selector, prefit=True, threshold=1.5 * np.mean(selector.feature_importances_))

        # 获取选择的特征索引
        self.selected_features_indices = model.get_support(indices=True)
        self.selected_features_indices = X.iloc[:, self.selected_features_indices].columns.tolist()

        # 数据预处理
        self.scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.scaler.fit_transform(self.data[self.selected_features_indices + [target]])

        # 创建数据集
        def create_dataset(data, time_step=10, predict_days=10):
            X, y = [], []
            for i in range(len(data) - time_step - predict_days + 1):
                X.append(data[i:(i + time_step), :-1])
                y.append(data[(i + time_step):(i + time_step + predict_days), -1])
            return np.array(X), np.array(y)

        time_step = 10
        predict_days = 10
        X, y = create_dataset(scaled_data, time_step, predict_days)

        # 划分训练集和测试集
        train_size = int(len(X) * 0.8)
        X_train, X_test = X[:train_size], X[train_size:]
        y_train, y_test = y[:train_size], y[train_size:]

        # 构建 LSTM 模型
        self.model = Sequential()
        self.model.add(Bidirectional(LSTM(units=96, return_sequences=True, input_shape=(X_train.shape[1], X_train.shape[2]))))
        self.model.add(Dropout(0.3))
        self.model.add(Bidirectional(LSTM(units=96, return_sequences=False)))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(units=10))  # 预测未来10天

        # 编译模型
        # 使用 Adam 优化器时，指定学习率
        optimizer = RMSprop(learning_rate=0.001)

        # 编译模型时使用正确的优化器
        self.model.compile(optimizer=optimizer, loss='mean_squared_error')

        # 设置回调函数
        early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)

        # 训练模型
        self.model.fit(X_train, y_train, epochs=300, batch_size=32, validation_data=(X_test, y_test),
                       callbacks=[early_stopping, reduce_lr], verbose=1)

        # 预测
        predictions = self.model.predict(X_test)

        # 反归一化
        # 只对预测的 'close' 列进行反归一化
        self.close_scaler = MinMaxScaler(feature_range=(0, 1))
        self.close_scaler.min_, self.close_scaler.scale_ = self.scaler.min_[-1], self.scaler.scale_[-1]
        predictions_inverted = self.close_scaler.inverse_transform(predictions)

        y_test_actual = self.close_scaler.inverse_transform(y_test)

        # 打印预测结果
        # print("Predictions:\n", predictions_inverted)
        # print("Actual:\n", y_test_actual)

        # 计算误差指标
        mse_list = []
        rmse_list = []
        target_mean_list = []
        target_std_list = []

        for i in range(10):
            # 计算每一天的误差
            mse = mean_squared_error(y_test[:, i], predictions[:, i])
            rmse = np.sqrt(mse)
            target_mean = np.mean(y_test[:, i])
            target_std = np.std(y_test[:, i])

            mse_list.append(mse)
            rmse_list.append(rmse)
            target_mean_list.append(target_mean)
            target_std_list.append(target_std)

            # 打印结果
            print(f"Day {i + 1}:")
            print(f"  Mean Squared Error: {mse}")
            print(f"  Root Mean Squared Error: {rmse}")
            print(f"  Target Mean: {target_mean}")
            print(f"  Target Std: {target_std}")
            print()

        # 如果需要将结果保存为 DataFrame
        results_df = pd.DataFrame({
            'Day': range(1, 11),
            'MSE': mse_list,
            'RMSE': rmse_list,
            'Target Mean': target_mean_list,
            'Target Std': target_std_list
        })

        print(results_df)

        # 计算残差
        residuals = y_test_actual - predictions_inverted

        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=("Residuals for 10 Days", "Prediction Error Distribution", "Time Series Prediction"),
            vertical_spacing=0.1
        )

        # 添加残差图
        for i in range(10):
            fig.add_trace(go.Scatter(
                x=np.arange(len(residuals)),
                y=residuals[:, i],
                mode='lines',
                name=f'Day {i + 1} Residuals'
            ), row=1, col=1)

        # 添加预测误差分布图
        for i in range(10):
            fig.add_trace(go.Histogram(
                x=residuals[:, i],
                name=f'Day {i + 1} Error',
                opacity=0.5
            ), row=2, col=1)

        # 添加时间序列预测图
        for i in range(10):
            fig.add_trace(go.Scatter(
                x=np.arange(len(y_test_actual)),
                y=y_test_actual[:, i],
                mode='lines',
                name=f'Day {i + 1} True'
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=np.arange(len(predictions_inverted)),
                y=predictions_inverted[:, i],
                mode='lines',
                name=f'Day {i + 1} Predicted'
            ), row=3, col=1)

        # 更新布局
        fig.update_layout(
            title_text="Interactive Visualization of Predictions",
            height=1920,
            showlegend=True,
            legend_title='Legend'
        )

        # 保存为 HTML 文件
        fig.write_html('interactive_visualization.html')

        # 显示图表
        fig.show()

        return self

    @time_counter(logger_name=__name__)
    def model_init(self) -> "LSTMProcess":

        return self

    @time_counter(logger_name=__name__)
    def model_train(self) -> "LSTMProcess":

        return self

    @time_counter(logger_name=__name__)
    def model_test(self) -> "LSTMProcess":

        return self

    @time_counter(logger_name=__name__)
    def model_report(self) -> "LSTMProcess":
        """Generate a report of the RandomForest model's performance."""

        return self

    @time_counter(logger_name=__name__)
    def model_predict(self, future_data):
        """Make predictions with the RandomForest model."""
        self.logger.info("Making predictions with RandomForest model")
        # 选择特征

        # 数据预处理
        # 使用与训练数据相同的 scaler
        future_scaled_data = self.scaler.transform(future_data[self.selected_features_indices + ['close']])

        def create_dataset(data, time_step=10, predict_days=10):
            X, y = [], []
            for i in range(len(data) - time_step - predict_days + 1):
                X.append(data[i:(i + time_step), :-1])
                y.append(data[(i + time_step):(i + time_step + predict_days), -1])
            return np.array(X), np.array(y)

        # 创建输入序列
        def create_future_dataset(data, time_step=10):
            X, y = [], []
            for i in range(len(data) - time_step + 1):
                X.append(data[i:(i + time_step)])
                y.append(data[i:(i + time_step), :-1])
            return np.array(X), np.array(y)

        time_step = 10
        X_future, y = create_future_dataset(future_scaled_data, time_step)

        # 确保 X_future 的形状适合模型输入
        if y.shape[0] > 0:
            # 预测未来价格
            y_input = y[-1].reshape(1, time_step, -1)

            future_predictions = self.model.predict(y_input)

            # 反归一化
            # 只对预测的 'close' 列进行反归一化
            future_predictions = self.close_scaler.inverse_transform(future_predictions)

            # 打印预测结果
            print("Future Predictions:")
            print(future_predictions)

            return future_predictions
        else:
            print("Not enough data to create input sequences for prediction.")

        return None
