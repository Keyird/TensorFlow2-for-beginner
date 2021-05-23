import tensorflow as tf

""" 填充 Padding """

""" 对二维矩阵进行填充 """
a = tf.reshape(tf.range(9), [3, 3])
print("原数组a：", a)

b = tf.pad(a, [[0, 0], [0, 0]])
print("经过第一次填充后：", b)

c = tf.pad(a, [[1, 0], [0, 0]])
print("经过第二次填充后：", c)


""" 对四维张量进行填充 """
x = tf.random.normal([4, 28, 28, 3])
print("x.shape：",x.shape)
x_pad = tf.pad(x, [[0, 0], [2, 2], [2, 2], [0, 0]])
print("x_pad.shape：", x_pad.shape)


""" 复制 """
a1 = tf.reshape(tf.range(9), [3, 3])
print("原数组a1：", a1)
b1 = tf.tile(a1, [1, 2])
print("在列方向上复制一倍：", b1)
c1 = tf.tile(a1, [2, 1])
print("在行方向上复制一倍：", c1)
d1 = tf.tile(a1, [2, 2])
print("在行、列方向上各复制一倍：", d1)


# 维度上的复制：[3,3] -> [2,3,3]
a2 = tf.reshape(tf.range(9), [3, 3])
print("原数组a2：", a2)
a2 = tf.expand_dims(a2, axis=0)  # 增加一个维度
b2 = tf.tile(a2, [2, 1, 1])
print("原数组b2：", b2)
