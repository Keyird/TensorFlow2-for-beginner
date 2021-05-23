import tensorflow as tf
from tensorflow.keras import layers

# 新建各个网络层
fc1 = layers.Dense(500, activation=tf.nn.relu)
fc2 = layers.Dense(10, activation=tf.nn.relu)

# 前向计算
x = tf.random.normal([2, 28*28])  # 模拟输入
o1 = fc1(x)
o2 = fc2(o1)

# 输出层的shape：[2,10]
print(o2.shape)

# 获取中间层的权值矩阵
print(fc1.kernel)
# 获取中间层偏置向量
print(fc1.bias)
# 返回待优化参数列表
print(fc1.trainable_variables)
# 返回所有参数列表
print(fc1.variables)