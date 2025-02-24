import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestRegressor
from sklearn.feature_selection import SelectFromModel
from sklearn.metrics import mean_squared_error, mean_absolute_error

from src.service.processor.Processor import Processor
from src.tools.pycaterpillar_wrapper import time_counter


class RandomForestProcessor(Processor):

    @time_counter(logger_name=__name__)
    def model_init(self) -> "RandomForestProcessor":
        """Initialize the RandomForest model."""

        # 拟合 selector
        selector = RandomForestRegressor(n_estimators=self.params["n_estimators"], max_depth=self.params["max_depth"], random_state=self.params["random_state"])
        selector.fit(self.X_train, self.y_train)

        # 使用 SelectFromModel
        model = SelectFromModel(selector, prefit=True, threshold=1.5 * np.mean(selector.feature_importances_))

        # 获取选择的特征索引
        self.selected_features_indices = model.get_support(indices=True)

        # 转换训练和测试数据
        self.X_train_selected = model.transform(self.X_train)
        self.X_test_selected = model.transform(self.X_test)

        self.store.save_model_feature(data=self.selected_features_indices.tolist(), file_name=self.name + "_RandomForestFeatures")

        return self

    @time_counter(logger_name=__name__)
    def model_train(self) -> "RandomForestProcessor":
        """Train the RandomForest model."""
        self.logger.info("Training RandomForest model")

        self.model = RandomForestRegressor(n_estimators=self.params["n_estimators"], max_depth=self.params["max_depth"], random_state=self.params["random_state"])
        self.model.fit(self.X_train_selected, self.y_train)

        self.store.save_model(data=self.model, file_name=self.name + "_RandomForestModel")

        self.logger.info("Model training completed")

        return self

    @time_counter(logger_name=__name__)
    def model_test(self) -> "RandomForestProcessor":
        """Test the RandomForest model."""
        self.logger.info("Testing RandomForest model")
        self.y_predict = self.model.predict(self.X_test_selected)

        return self

    @time_counter(logger_name=__name__)
    def model_report(self) -> "RandomForestProcessor":
        """Generate a report of the RandomForest model's performance."""
        self.logger.info("Generating model performance report")

        # 计算评估指标
        mse = mean_squared_error(self.y_test, self.y_predict)
        mae = mean_absolute_error(self.y_test, self.y_predict)
        rmse = np.sqrt(mse)

        self.logger.info(f"Mean Squared Error: {mse}")
        self.logger.info(f"Mean Absolute Error: {mae}")
        self.logger.info(f"Root Mean Squared Error: {rmse}")

        # 1. 残差图
        residuals = self.y_test.astype(float) - self.y_predict.astype(float)
        # 1. 残差图
        fig_residuals = px.scatter(x=self.y_predict.flatten(), y=residuals.flatten(),
                                   labels={'x': 'Predicted Values', 'y': 'Residuals'},
                                   title='Residual Plot')
        fig_residuals.add_hline(y=0, line_dash="dash", line_color="red")

        # 2. 特征重要性图
        feature_importances = self.model.feature_importances_
        fig_feature_importance = px.bar(x=range(len(feature_importances)), y=feature_importances,
                                        labels={'x': 'Feature Index', 'y': 'Importance'},
                                        title='Feature Importance')

        # 3. 预测误差分布图
        errors = residuals.flatten()
        fig_error_dist = px.histogram(errors, nbins=20, title='Prediction Error Distribution', labels={'value': 'Error'})

        # 4. 时间序列预测图
        fig_time_series = go.Figure()
        for i in range(self.y_test.shape[1]):
            fig_time_series.add_trace(go.Scatter(y=self.y_test[:, i], mode='lines', name=f'True Day {i + 1}'))
            fig_time_series.add_trace(go.Scatter(y=self.y_predict[:, i], mode='lines', name=f'Pred Day {i + 1}', line=dict(dash='dash')))

        fig_time_series.update_layout(title='Time Series Forecast', xaxis_title='Sample Index', yaxis_title='Value')

        # 创建一个包含所有图表的子图
        fig = make_subplots(rows=4, cols=1, subplot_titles=("Residual Plot", "Feature Importance", "Prediction Error Distribution", "Time Series Forecast"))

        # 添加各个图表到子图
        for trace in fig_residuals.data:
            fig.add_trace(trace, row=1, col=1)

        for trace in fig_feature_importance.data:
            fig.add_trace(trace, row=2, col=1)

        for trace in fig_error_dist.data:
            fig.add_trace(trace, row=3, col=1)

        for trace in fig_time_series.data:
            fig.add_trace(trace, row=4, col=1)

        # 更新布局
        fig.update_layout(height=1200, showlegend=True)

        # 保存为 HTML 文件
        fig.write_html("model_analysis.html")

        # 显示图表
        fig.show()

        return self

    @time_counter(logger_name=__name__)
    def model_predict(self, x):
        """Make predictions with the RandomForest model."""
        self.logger.info("Making predictions with RandomForest model")
        return self.model.predict(x)
