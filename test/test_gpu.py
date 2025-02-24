import tensorflow as tf
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))

import torch
print("CUDA available: ", torch.cuda.is_available())
print("CUDA version: ", torch.version.cuda)