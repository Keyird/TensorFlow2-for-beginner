import tensorflow as tf

# 合并
# 1. concat()
# 随机生成a,b两个张量
a = tf.random.normal([5, 40, 10])
b = tf.random.normal([3, 40, 10])
c = tf.concat([a, b], axis=0)
print(c.shape)

# 2. stack()
a = tf.random.normal([3, 40, 10])
b = tf.random.normal([3, 40, 10])
c = tf.stack([a, b], axis=0)
print(c.shape)

# 分割
# 3. unstack()
x = tf.random.normal([8,28,28,3])
result = tf.unstack(x, axis=0)

# 4. split()
x = tf.random.normal([8,28,28,3])
result = tf.split(x, axis=0, num_or_size_splits=[2,4,2])
print(result)
