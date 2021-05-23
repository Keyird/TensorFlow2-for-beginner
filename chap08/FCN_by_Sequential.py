import tensorflow as tf
from tensorflow.keras import Sequential, layers

model = Sequential([
    layers.Dense(500, activation=tf.nn.relu),   # 创建隐藏层
    layers.Dense(10, activation=None),          # 创建输出层
])

# 模拟输入
x = tf.random.normal([2, 28*28])
# 输出
out = model(x)
# [2,10]
print(out.shape)
