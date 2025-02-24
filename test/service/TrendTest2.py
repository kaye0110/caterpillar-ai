import numpy as np
import pandas as pd
import plotly.graph_objects as go

# 示例数据
predict_close = np.array([
    67.23601532, 66.25403595, 66.63196564, 67.20308685, 65.1477356, 66.32671356,
    66.18839264, 64.61388397, 64.1767807, 63.65986633, 64.10240936, 63.84789658,
    62.8686676, 61.40155411, 62.19060135, 60.11246872, 61.89958954, 60.57504654,
    59.25986099, 60.96587753, 67.23601532, 66.25403595, 66.63196564, 67.20308685,
    65.1477356, 66.32671356, 66.18839264, 64.61388397, 64.1767807, 63.65986633,
    64.10240936, 63.84789658, 62.8686676, 61.40155411, 62.19060135, 60.11246872,
    61.89958954, 60.57504654, 59.25986099, 60.96587753, np.nan
])

# 将数据转换为 Pandas Series 以便使用 rolling 方法
predict_close_series = pd.Series(predict_close)

# 定义窗口大小
window_sizes = [3, 5, 10, 15, 20]

# 创建 Plotly 图表
fig = go.Figure()

# 遍历窗口大小
for window in window_sizes:
    # 计算滑动平均值（平滑曲线）
    smoothed_data = predict_close_series.rolling(window=window, min_periods=1).mean()

    # 计算一阶导数
    first_derivative = np.gradient(smoothed_data)

    # 计算二阶导数
    second_derivative = np.gradient(first_derivative)

    # 添加平滑曲线
    fig.add_trace(go.Scatter(
        x=np.arange(len(predict_close)),
        y=predict_close,
        mode='lines',
        name=f'Smoothed (Window={window})',
        visible=(window == 3)  # 默认只显示第一个窗口的结果
    ))

    # 添加一阶导数
    fig.add_trace(go.Scatter(
        x=np.arange(len(first_derivative)),
        y=first_derivative,
        mode='lines',
        name=f'First Derivative (Window={window})',
        visible=(window == 3)  # 默认只显示第一个窗口的结果
    ))

    # 添加二阶导数
    fig.add_trace(go.Scatter(
        x=np.arange(len(second_derivative)),
        y=second_derivative,
        mode='lines',
        name=f'Second Derivative (Window={window})',
        visible=(window == 3)  # 默认只显示第一个窗口的结果
    ))

# 添加下拉菜单
fig.update_layout(
    updatemenus=[
        {
            "buttons": [
                {
                    "method": "update",
                    "label": f"Window={window}",
                    "args": [
                        {"visible": [window == w for w in window_sizes for _ in range(3)]},
                        {"title": f"Smoothed Data and Derivatives (Window={window})"}
                    ]
                }
                for window in window_sizes
            ],
            "direction": "down",
            "showactive": True,
            "x": 0.1,
            "y": 1.2
        }
    ]
)

# 更新布局
fig.update_layout(
    title="Smoothed Data and Derivatives (Window=3)",
    xaxis_title="Time",
    yaxis_title="Value",
    legend_title="Legend",
    template="plotly_white"
)

# 显示图表
fig.show()
