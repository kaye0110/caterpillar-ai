import tensorflow as tf

from src.config.AppConfig import AppConfig

cfg = AppConfig()

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices())

devices = tf.config.list_physical_devices()
print("\nDevices: ", devices)

gpus = tf.config.list_physical_devices('GPU')
if gpus:
    details = tf.config.experimental.get_device_details(gpus[0])
    print("GPU details: ", details)

# with tf.device('/GPU:0'):
#     cifar = tf.keras.datasets.cifar100
#     (x_train, y_train), (x_test, y_test) = cifar.load_data()
#     model = tf.keras.applications.ResNet50(
#         include_top=True,
#         weights=None,
#         input_shape=(32, 32, 3),
#         classes=100, )
#
#     loss_fn = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
#     model.compile(optimizer="adam", loss=loss_fn, metrics=["accuracy"])
#     model.fit(x_train, y_train, epochs=5, batch_size=64)

import time

# 创建一个随机的大型矩阵
matrix_size = 10000
matrix_a = tf.random.normal((matrix_size, matrix_size))
matrix_b = tf.random.normal((matrix_size, matrix_size))

start_time = time.time()
result_gpu = tf.matmul(matrix_a, matrix_b)
end_time = time.time()
print("使用GPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

start_time = time.time()
result_gpu = tf.matmul(matrix_a, matrix_b)
end_time = time.time()
print("使用GPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

start_time = time.time()
result_gpu = tf.matmul(matrix_a, matrix_b)
end_time = time.time()
print("使用GPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

# 使用CPU执行矩阵乘法
with tf.device('/CPU:0'):
    start_time = time.time()
    result_cpu = tf.matmul(matrix_a, matrix_b)
    end_time = time.time()
    print("使用CPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

    start_time = time.time()
    result_gpu = tf.matmul(matrix_a, matrix_b)
    end_time = time.time()
    print("使用GPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

# 使用GPU执行矩阵乘法
with tf.device('/GPU:0'):
    start_time = time.time()
    result_gpu = tf.matmul(matrix_a, matrix_b)
    end_time = time.time()
    print("使用GPU执行矩阵乘法所需时间：", end_time - start_time, "秒")

    start_time = time.time()
    result_gpu = tf.matmul(matrix_a, matrix_b)
    end_time = time.time()
    print("使用GPU执行矩阵乘法所需时间：", end_time - start_time, "秒")
