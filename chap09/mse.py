import tensorflow as tf

# 初始化6个数据，限制在0-9，表示类别标签
y = tf.constant([1, 2, 3, 9, 0, 8])
# one-hot编码
y = tf.one_hot(y, depth=10)
# 类型转换
y = tf.cast(y, dtype=tf.float32)

# 随机模拟的输出
out = tf.random.normal([6, 10])

# 构建损失函数
loss_1 = tf.reduce_mean(tf.square(y-out))
loss_2 = tf.square(tf.norm(y-out))/(6*10)
loss_3 = tf.reduce_mean(tf.losses.MSE(y, out))

print(loss_1)
print(loss_2)
print(loss_3)