import os
import random

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import tensorflow as tf
from keras.src.optimizers import Adam
from keras_tuner import RandomSearch
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import LSTM, Dense, Dropout, Bidirectional
from tensorflow.keras.models import Sequential

from src.service.processor.v3.Config import Configuration
from src.service.processor.v3.Processor import Processor
from src.tools.pycaterpillar_wrapper import time_counter


class LSTMProcess(Processor):

    def __init__(self, config: Configuration):
        super().__init__(config)

        random.seed(self.config.random_state)
        np.random.seed(self.config.random_state)
        tf.random.set_seed(self.config.random_state)

    # 创建数据集
    def _create_dataset(self, data: pd.DataFrame, time_step, predict_days):
        if time_step is None:
            time_step = self.config.n_timestep
        if predict_days is None:
            predict_days = self.config.n_predict

        X, y = [], []
        for i in range(len(data) - time_step - predict_days + 1):
            X.append(data[i:(i + time_step), :-1])
            y.append(data[(i + time_step):(i + time_step + predict_days), -1])
        return np.array(X), np.array(y)

    # 创建输入序列
    def _create_future_dataset(self, data, time_step):
        if time_step is None:
            time_step = self.config.n_timestep

        X, y = [], []
        for i in range(len(data) - time_step + 1):
            X.append(data[i:(i + time_step)])
            y.append(data[i:(i + time_step), :-1])
        return np.array(X), np.array(y)

    @time_counter(logger_name=__name__)
    def prepare(self, data: pd.DataFrame, ) -> "LSTMProcess":
        self.logger.info("Preparing data for LSTMProcessor...")

        super().prepare(data)

        # 选择特征和目标
        features = self.features.columns.tolist()

        # 划分训练集和测试集
        train_size = int(len(self.data) * self.config.test_size)

        X = self.data[features]
        y = self.targets

        # 拟合 selector
        selector = RandomForestRegressor(n_estimators=self.config.n_estimators, max_depth=self.config.max_depth, random_state=self.config.random_state, n_jobs=self.config.n_jobs)
        selector.fit(X[:train_size], y[:train_size])

        # 使用 SelectFromModel
        model = SelectFromModel(selector, prefit=True, threshold=1.75 * np.mean(selector.feature_importances_))

        # 获取选择的特征索引
        selected_feature_indexes = model.get_support(indices=True)
        self.selected_features = X.iloc[:, selected_feature_indexes].columns.tolist()

        # 数据预处理
        self.feature_scaler = MinMaxScaler(feature_range=(0, 1))
        scaled_data = self.feature_scaler.fit_transform(self.data[self.selected_features + [self.config.target_names[0]]])

        self.X, self.y = self._create_dataset(scaled_data, self.config.n_timestep, self.config.n_predict)

        self.logger.info("X shape: ")
        self.logger.info(self.X.shape)

        self.X_train, self.X_test = self.X[:train_size], self.X[train_size:]
        self.y_train, self.y_test = self.y[:train_size], self.y[train_size:]

        self.logger.info("X_train shape: ")
        self.logger.info(self.X_test.shape)

        self.target_scaler = MinMaxScaler(feature_range=(0, 1))
        self.target_scaler.min_, self.target_scaler.scale_ = self.feature_scaler.min_[-1], self.feature_scaler.scale_[-1]

        self.store.save_model(data=self.config.serialize(), file_name=self.store.get_config_name(self.config.batch_code), file_type="json")
        self.store.save_model(data=self.feature_scaler, file_name=self.store.get_feature_scaler_name(self.config.batch_code))
        self.store.save_model(data=self.target_scaler, file_name=self.store.get_target_scaler_name(self.config.batch_code))
        self.store.save_model_feature(data=self.selected_features, file_name=self.store.get_selected_features_name(self.config.batch_code))

        return self

    def build_model(self, hp):
        model = Sequential()
        # 调整 LSTM 单元数
        # model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2]))))
        # model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        # model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=256, step=32), return_sequences=False)))
        # model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        # model.add(Dense(units=self.config.n_predict))

        # model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2]))))
        # model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        # model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=True)))
        # model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        # model.add(Bidirectional(LSTM(units=hp.Int('units', min_value=32, max_value=512, step=32), return_sequences=False)))
        # model.add(Dropout(hp.Float('dropout_rate', min_value=0.1, max_value=0.5, step=0.1)))
        # model.add(Dense(units=self.config.n_predict))
        #
        # # 使用不同的优化器和学习率
        # model.compile(optimizer=Adam(learning_rate=hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4, 1e-5])), loss='mean_squared_error')

        input_shape = (self.X_train.shape[1], self.X_train.shape[2])

        # 第一层双向 LSTM
        """
        **第一层 LSTM**：
       - **作用**：作为输入层，它的主要任务是从原始输入数据中提取初步的时间序列特征。由于它是双向的（Bidirectional），它可以同时考虑到时间序列的过去和未来信息。
       - **参数**：
         - `units`: 表示 LSTM 单元的数量，决定了该层的输出维度。更多的单元可以捕捉到更复杂的特征，但也可能导致过拟合。
         - `return_sequences`: 设置为 `True`，表示该层会返回完整的输出序列，这对于堆叠多层 LSTM 是必要的。
         - `input_shape`: 输入数据的形状，通常是时间步长和特征数。
        """
        model.add(Bidirectional(
            LSTM(units=hp.Int('units_1', min_value=64, max_value=256, step=32),
                 return_sequences=True,
                 input_shape=input_shape)))
        model.add(Dropout(hp.Float('dropout_rate_1', min_value=0.3, max_value=0.5, step=0.05)))

        # 第二层双向 LSTM
        """
        - **作用**：进一步处理第一层提取的特征，捕捉更高层次的时间序列模式。继续使用双向 LSTM 可以增强对序列的理解。
       - **参数**：
         - `units`: 同样表示 LSTM 单元的数量，影响该层的复杂度和能力。
         - `return_sequences`: 继续设置为 `True`，以便为下一层提供完整的序列输出。
        """
        model.add(Bidirectional(
            LSTM(units=hp.Int('units_2', min_value=64, max_value=256, step=32),
                 return_sequences=True)))
        model.add(Dropout(hp.Float('dropout_rate_2', min_value=0.3, max_value=0.5, step=0.05)))

        # 第三层双向 LSTM
        """
        - **作用**：作为最后一层 LSTM，它的任务是将前两层提取的特征进行整合，生成最终的时间序列特征表示。由于 `return_sequences` 设置为 `False`，它只输出最后一个时间步的结果，这通常用于预测任务。
       - **参数**：
         - `units`: 决定了最终特征表示的维度。
         - `return_sequences`: 设置为 `False`，表示只输出最后一个时间步的结果。
        """
        model.add(Bidirectional(
            LSTM(units=hp.Int('units_3', min_value=64, max_value=256, step=32),
                 return_sequences=False)))
        model.add(Dropout(hp.Float('dropout_rate_3', min_value=0.01, max_value=0.5, step=0.005)))

        # 输出层
        model.add(Dense(units=self.config.n_predict))

        # 编译模型
        learning_rate = hp.Choice('learning_rate', values=[1e-2, 5e-3, 1e-3])
        optimizer = Adam(learning_rate=learning_rate)
        model.compile(optimizer=optimizer, loss='mean_squared_error')

        return model

    @time_counter(logger_name=__name__)
    def train(self) -> "LSTMProcess":
        self.logger.info("Start to train LSTM model with hyperparameter tuning.")

        tuner = RandomSearch(
            self.build_model,
            objective='val_loss',
            max_trials=self.config.max_trails,
            executions_per_trial=self.config.executions_per_trial,
            directory=self.store.model_data_path,
            project_name=self.config.batch_code + '_lstm_tuning_v3'
        )

        # 添加EarlyStopping回调
        early_stopping = EarlyStopping(monitor='val_loss',  # 监控验证损失
                                       patience=10,  # 如果验证损失在10个epoch内没有改善，则停止训练
                                       restore_best_weights=True  # 恢复具有最佳验证损失的模型权重
                                       )

        tscv = TimeSeriesSplit(n_splits=5)
        for train_index, test_index in tscv.split(self.X):
            X_train, X_test = self.X[train_index], self.X[test_index]
            y_train, y_test = self.y[train_index], self.y[test_index]

            # 在tuner.search中添加early_stopping回调
            tuner.search(X_train, y_train, epochs=self.config.epochs, validation_data=(X_test, y_test), callbacks=[early_stopping], verbose=1)

        # 获取最佳模型
        self.model = tuner.get_best_models(num_models=1)[0]
        self.store.save_model(data=self.model, file_name=self.store.get_model_name(self.config.batch_code))

        return self

    # @time_counter(logger_name=__name__)
    # def train(self) -> "LSTMProcess":
    #     self.logger.info("Strat to train LSTM model.")
    #     # 构建 LSTM 模型
    #     self.model = Sequential()
    #     self.model.add(Bidirectional(LSTM(units=128, return_sequences=True, input_shape=(self.X_train.shape[1], self.X_train.shape[2]))))
    #     self.model.add(Dropout(0.3))
    #     self.model.add(Bidirectional(LSTM(units=128, return_sequences=True)))
    #     self.model.add(Dropout(0.3))
    #     self.model.add(Attention())
    #     self.model.add(Dense(units=self.config.n_predict))  # 预测未来10天
    #
    #     # 编译模型
    #     # 使用 Adam 优化器时，指定学习率
    #     optimizer = RMSprop(learning_rate=0.001)
    #     optimizer = Adam(learning_rate=0.001)
    #
    #     # 编译模型时使用正确的优化器
    #     self.model.compile(optimizer=optimizer, loss='mean_squared_error')
    #
    #     # 设置回调函数
    #     early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    #     reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=0.0001)
    #
    #     # 训练模型
    #     tscv = TimeSeriesSplit(n_splits=5)
    #     for train_index, test_index in tscv.split(self.X):
    #         X_train, X_test = self.X[train_index], self.X[test_index]
    #         y_train, y_test = self.y[train_index], self.y[test_index]
    #         self.model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=120, batch_size=32, callbacks=[early_stopping, reduce_lr], verbose=1)
    #
    #     # self.model.fit(self.X_train, self.y_train, epochs=300, batch_size=32, validation_data=(self.X_test, self.y_test),
    #     #                callbacks=[early_stopping, reduce_lr], verbose=1)
    #
    #     self.store.save_model(data=self.model, file_name=self.store.get_model_name(self.config.batch_code))
    #
    #     return self

    def test(self) -> "LSTMProcess":
        self.logger.info("Strat to test LSTM model.")

        for layer in self.model.layers:
            if isinstance(layer, LSTM):
                layer.reset_states()

        # 预测
        self.predict_result = self.model.predict(self.X_test)

        # 反归一化
        # 只对预测的 'close' 列进行反归一化
        self.predict_value = self.target_scaler.inverse_transform(self.predict_result)
        self.y_test_actual = self.target_scaler.inverse_transform(self.y_test)

        # 打印预测结果
        # self.logger.info("Actual:")
        # self.logger.info(self.y_test_actual)
        # self.logger.info("Predictions: ")
        # self.logger.info(self.predict_value)

        return self

    def report(self) -> "LSTMProcess":
        self.logger.info("Model Summary:")
        self.logger.info(self.model.summary())

        # 计算误差指标
        mse_list = []
        rmse_list = []
        target_mean_list = []
        target_std_list = []
        mae_list = []
        r2_list = []

        for i in range(self.config.n_predict):
            # 计算每一天的误差
            mse = mean_squared_error(self.y_test[:, i], self.predict_result[:, i])
            rmse = np.sqrt(mse)
            target_mean = np.mean(self.y_test[:, i])
            target_std = np.std(self.y_test[:, i])

            mae = mean_absolute_error(self.y_test[:, i], self.predict_result[:, i])
            r2 = r2_score(self.y_test[:, i], self.predict_result[:, i])

            mse_list.append(mse)
            rmse_list.append(rmse)
            target_mean_list.append(target_mean)
            target_std_list.append(target_std)
            mae_list.append(mae)
            r2_list.append(r2)

            # 打印结果
            self.logger.info(f"Day {i + 1}:")
            self.logger.info(f"  Mean Squared Error: {mse}")
            self.logger.info(f"  Root Mean Squared Error: {rmse}")
            self.logger.info(f"  Target Mean: {target_mean}")
            self.logger.info(f"  Target Std: {target_std}")

        # 判断过拟合
        if any(val_loss > train_loss for val_loss, train_loss in zip(mse_list, rmse_list)):
            self.logger.warning("The model may be overfitting. Consider using more regularization or simplifying the model.")

        # 如果需要将结果保存为 DataFrame
        results_df = pd.DataFrame({
            'Day': range(1, self.config.n_predict + 1),
            'MSE': mse_list,
            'RMSE': rmse_list,
            'Target Mean': target_mean_list,
            'Target Std': target_std_list,
            'MAE': mae_list,
            'R2': r2_list
        })

        self.logger.info(results_df)

        # 计算残差
        residuals = self.y_test_actual - self.predict_value

        # 创建子图布局
        fig = make_subplots(
            rows=3, cols=1,
            subplot_titles=(f"Residuals for {self.config.n_predict} Days", "Prediction Error Distribution", "Time Series Prediction"),
            vertical_spacing=0.1
        )

        # 添加残差图
        for i in range(self.config.n_predict):
            fig.add_trace(go.Scatter(
                x=np.arange(len(residuals)),
                y=residuals[:, i],
                mode='lines',
                name=f'Day {i + 1} Residuals'
            ), row=1, col=1)

        # 添加预测误差分布图
        for i in range(self.config.n_predict):
            fig.add_trace(go.Histogram(
                x=residuals[:, i],
                name=f'Day {i + 1} Error',
                opacity=0.5
            ), row=2, col=1)

        # 添加时间序列预测图
        for i in range(self.config.n_predict):
            fig.add_trace(go.Scatter(
                x=np.arange(len(self.y_test_actual)),
                y=self.y_test_actual[:, i],
                mode='lines',
                name=f'Day {i + 1} True'
            ), row=3, col=1)
            fig.add_trace(go.Scatter(
                x=np.arange(len(self.predict_value)),
                y=self.predict_value[:, i],
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

        fig_path = os.path.join(self.store.get_predict_result_path(), self.store.get_plotly_name(f"lstm_test_plotly.html"))
        # 保存为 HTML 文件
        fig.write_html(fig_path)

        # 显示图表
        # fig.show()

        return self

    def predict(self, future_data: pd.DataFrame) -> "LSTMProcess":
        # 数据预处理
        # 使用与训练数据相同的 scaler
        future_scaled_data = self.feature_scaler.transform(future_data[self.selected_features + [self.config.target_names[0]]])

        with_close, without_close = self._create_future_dataset(future_scaled_data, self.config.n_timestep)

        # 确保 X_future 的形状适合模型输入
        if without_close.shape[0] > 0:
            # 预测未来价格
            without_close = without_close[-1].reshape(1, self.config.n_timestep, -1)

            for layer in self.model.layers:
                # self.logger.info(f"type of layer {type(layer)}")
                # 检查是否是 Bidirectional 层
                if isinstance(layer, Bidirectional):
                    # 获取正向和反向的 LSTM 层
                    forward_layer = layer.forward_layer
                    backward_layer = layer.backward_layer

                    # 检查并重置 LSTM 层的状态
                    if isinstance(forward_layer, LSTM):
                        result = forward_layer.reset_states()
                        # self.logger.info(f"Reset states: {result}")
                    if isinstance(backward_layer, LSTM):
                        result = backward_layer.reset_states()
                        # self.logger.info(f"Reset states: {result}")

            self.future_predict_result = self.model.predict(without_close)
            # 反归一化
            # 只对预测的 'close' 列进行反归一化
            self.future_predict_value = self.target_scaler.inverse_transform(self.future_predict_result)

            # 打印预测结果
            # self.logger.info("Future Predictions: ")
            # self.logger.info(self.future_predict_value)

        else:
            raise "Not enough data to create input sequences for prediction."

        return self
