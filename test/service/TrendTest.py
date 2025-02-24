import numpy as np
import plotly.graph_objects as go

# 示例数据
predict_close = np.array([

    67.23601532, 66.25403595, 66.63196564, 67.20308685, 65.1477356, 66.32671356,
    66.18839264, 64.61388397, 64.1767807, 63.65986633, 64.10240936, 63.84789658,
    62.8686676, 61.40155411, 62.19060135, 60.11246872, 61.89958954, 60.57504654,
    59.25986099, 60.96587753, 67.23601532, 66.25403595, 66.63196564, 67.20308685,
    65.1477356, 66.32671356, 66.18839264, 64.61388397, 64.1767807, 63.65986633,
    64.10240936, 63.84789658, 62.8686676, 61.40155411, 62.19060135, 60.11246872,
    61.89958954, 60.57504654, 59.25986099, 60.96587753
])

# 计算一阶导数
first_derivative = np.gradient(predict_close)

# 计算二阶导数
second_derivative = np.gradient(first_derivative)

# 打印结果
print("Predict Close:", predict_close)
print("First Derivative (Trend):", first_derivative)
print("Second Derivative (Acceleration):", second_derivative)

# 创建 Plotly 图表
fig = go.Figure()

# 添加原始数据
fig.add_trace(go.Scatter(
    x=np.arange(len(predict_close)),
    y=predict_close,
    mode='lines+markers',
    name='Predict Close',
    line=dict(color='blue')
))

# 添加一阶导数（趋势）
fig.add_trace(go.Scatter(
    x=np.arange(len(first_derivative)),
    y=first_derivative,
    mode='lines+markers',
    name='First Derivative (Trend)',
    line=dict(color='orange')
))

# 添加二阶导数（加速度）
fig.add_trace(go.Scatter(
    x=np.arange(len(second_derivative)),
    y=second_derivative,
    mode='lines+markers',
    name='Second Derivative (Acceleration)',
    line=dict(color='green')
))

# 更新布局
fig.update_layout(
    title="Predict Close, First Derivative (Trend), and Second Derivative (Acceleration)",
    xaxis_title="Time",
    yaxis_title="Value",
    legend_title="Legend",
    template="plotly_white"
)

# 显示图表
fig.show()
