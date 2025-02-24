import logging
import traceback

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.preprocessing import StandardScaler, MinMaxScaler

from src.tools.pycaterpillar_wrapper import time_counter


class DataClearService:
    def __init__(self, prices: pd.DataFrame):
        self.logger = logging.getLogger(__name__)
        self.prices = prices
        self.params = {"indicator_prefix": "indicator_", "feature_prefix": "feature_"}

    def _normalize_indicator(self, indicator_name: str, need_plot: bool = False) -> pd.Series:
        try:
            # 计算描述性统计
            feature_data = self.prices[indicator_name]
            mean = feature_data.mean()
            std = feature_data.std()
            min_val = feature_data.min()
            max_val = feature_data.max()
            unique_count = feature_data.nunique()
            skewness = feature_data.skew()
            kurtosis = feature_data.kurtosis()

            # 打印描述性统计
            self.logger.debug(f"Feature: {indicator_name} size: {len(feature_data)}")
            self.logger.debug(f"Mean: {mean}")
            self.logger.debug(f"Standard Deviation: {std}")
            self.logger.debug(f"Min: {min_val}")
            self.logger.debug(f"Max: {max_val}")
            self.logger.debug(f"Unique Values: {unique_count}")
            self.logger.debug(f"Skewness: {skewness}")
            self.logger.debug(f"Kurtosis: {kurtosis}")

            if need_plot:
                # 可视化分布
                plt.figure(figsize=(14, 6))

                plt.subplot(1, 3, 1)
                feature_data.hist(bins=50)
                plt.title(f'Histogram of {indicator_name}')

                plt.subplot(1, 3, 2)
                sns.boxplot(x=feature_data)
                plt.title(f'Boxplot of {indicator_name}')

                plt.subplot(1, 3, 3)
                sns.kdeplot(feature_data, shade=True)
                plt.title(f'Density Plot of {indicator_name}')

                plt.show()

            # 判断数据类型并进行归一化
            if unique_count < 10:
                self.logger.debug("This feature is likely categorical or binary due to low unique value count.")
                self.logger.debug("该特征可能是分类特征或二值特征，因为唯一值数量较少。")
                # 对于分类特征，通常不进行归一化，但可以进行独热编码
                # data = pd.get_dummies(data, columns=[feature_name])
                return self.prices[indicator_name]
            elif unique_count > len(feature_data) * 0.8:
                self.logger.debug("This feature is likely to be random due to high unique value count.")
                self.logger.debug("该特征可能是随机性数据，因为唯一值数量较多。")
                # 对于随机性数据，使用标准化
                scaler = StandardScaler()
                return scaler.fit_transform(self.prices[[indicator_name]])
            elif std > (max_val - min_val) * 0.1:
                self.logger.debug("This feature is likely to be random due to high standard deviation relative to range.")
                self.logger.debug("该特征可能是随机性数据，因为标准差相对于范围较大。")
                # 对于随机性数据，使用标准化
                scaler = StandardScaler()
                return scaler.fit_transform(self.prices[[indicator_name]])
            elif abs(skewness) > 1:
                self.logger.debug("This feature is likely to be skewed, indicating potential non-normal distribution.")
                self.logger.debug("该特征可能是偏态分布，表明数据可能不是正态分布。")
                # 对于偏态分布数据，使用标准化
                scaler = StandardScaler()
                return scaler.fit_transform(self.prices[[indicator_name]])
            elif kurtosis > 3:
                self.logger.debug("This feature has high kurtosis, indicating heavy tails or outliers.")
                self.logger.debug("该特征具有高峰度，表明可能存在重尾或异常值。")
                # 对于高峰度数据，使用标准化
                scaler = StandardScaler()
                return scaler.fit_transform(self.prices[[indicator_name]])
            else:
                self.logger.debug("This feature is likely to be continuous and normally distributed.")
                self.logger.debug("该特征可能是连续且正态分布的。")
                # 对于连续且正态分布的数据，使用归一化
                scaler = MinMaxScaler()
                return scaler.fit_transform(self.prices[[indicator_name]])

        except Exception as e:
            self.logger.error(f"Error occurred while normalizing feature {indicator_name}: {e} ")
            self.logger.error(traceback.format_exc())
            return self.prices[[indicator_name]]

    @time_counter(logger_name=__name__)
    def covert_to_float(self) -> "DataClearService":
        # 筛选出列名包含 'indicator_' 的列
        columns_to_convert = self.prices.filter(like=self.params['indicator_prefix']).columns

        # 将这些列的值转换为浮点数，并限制小数位数
        for column in columns_to_convert:
            self.prices[column] = self.prices[column].astype(float).round(8)

        return self

    @time_counter(logger_name=__name__)
    def remove_nan_rows(self) -> "DataClearService":
        # 获取数值类型的列
        numeric_cols = self.prices.filter(like=self.params["indicator_prefix"]).columns

        # 检查每一行是否有 NaN 或 inf
        def is_valid_row(row):
            numeric_row = None
            try:
                numeric_row = row[numeric_cols]

                return not numeric_row.isnull().any()
            except Exception as e:
                self.logger.error(f"Error occurred while checking row validity: {e} {numeric_row}")
                return False

        valid_rows = self.prices.apply(is_valid_row, axis=1)

        # 找到第一个所有字段都不为 NaN 或 inf 的行索引
        first_valid_index = valid_rows.idxmax() if valid_rows.any() else None

        if first_valid_index is not None:
            self.prices = self.prices.loc[first_valid_index:].copy().reset_index(drop=True)
        else:
            raise ValueError("No valid rows found in the DataFrame.")

        return self

    @time_counter(logger_name=__name__)
    def normalize_date(self) -> "DataClearService":

        indicators = self.prices.filter(like=self.params['indicator_prefix']).columns
        new_features = pd.DataFrame()

        for indicator_name in indicators:
            feature_series = self._normalize_indicator(indicator_name)
            if isinstance(feature_series, np.ndarray):
                feature_series = pd.Series(feature_series.flatten())

            feature_name = indicator_name.replace(self.params['indicator_prefix'], self.params['feature_prefix'])

            new_features = pd.concat([new_features, pd.Series(feature_series, name=feature_name)], axis=1)

            if len(feature_series) != len(self.prices):
                raise ValueError(f"Length of normalized feature {feature_name} does not match length of cleaned prices.")

        self.prices = pd.concat([self.prices, new_features], axis=1)
        # self.prices.drop(columns=indicators, inplace=True)

        return self
