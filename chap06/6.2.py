import tensorflow as tf
import numpy as np

""" 数据统计 """

# 1. 向量范数
x = tf.ones([2, 2])
tf.norm(x, ord=1)  # L1范数
tf.norm(x, ord=2)  # L2范数
tf.norm(x, ord=np.inf)  # 无穷范数

# 2. 最大值、最小值
x = tf.random.normal([4, 10])
tf.reduce_max(x, axis=1)  # 统计概率维度上的最大值（第2个维度）
tf.reduce_min(x, axis=1)  # 统计概率维度上的最小值（第2个维度）

# 3. 均值、和
x = tf.random.normal([4, 10])
tf.reduce_mean(x, axis=1)  # 统计概率维度上的均值（第2个维度）
tf.reduce_sum(x, axis=-1)  # 统计概率维度上的和（第2个维度）

# 4. 最大值索引、最小值索引
out = tf.random.normal([4, 10])
pred_max_index = tf.argmax(out, axis=1)  # 最大值索引（第二维度）
pred_min_index = tf.argmin(out, axis=1)  # 最小值索引 （第二维度）


