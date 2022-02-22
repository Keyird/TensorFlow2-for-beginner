import tensorflow as tf  # 导入TF库
from tensorflow.keras import datasets, Sequential, layers  # 导入TF子库
from chap19.model_save1 import network

network.load_weights('mnist.ckpt')
print('model is loaded!')