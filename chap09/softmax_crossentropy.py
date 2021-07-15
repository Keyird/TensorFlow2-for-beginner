import tensorflow as tf
from tensorflow.keras import Sequential, layers

# 构建网络
model = Sequential([
    layers.Dense(500, activation=tf.nn.relu),   # 创建隐藏层
    layers.Dense(5, activation=None),          # 创建输出层
])

# 模拟输入
x = tf.random.normal([2, 28*28])
# 输出 [2,5]
logits = model(x)

# 方案一：softmax与crossentropy融合，训练时数值稳定
y = tf.constant([[0, 1, 0, 0, 0], [0, 0, 1, 0, 0]])
loss_1 = tf.reduce_mean(tf.losses.categorical_crossentropy(y, logits, from_logits=True))
print(loss_1)

# 方案二：softmax与crossentropy融合，训练时数值稳定
y_true = tf.constant([1, 2])
y_true = tf.one_hot(y_true, depth=5)  # one_hot编码
loss_2 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, logits, from_logits=True))
print(loss_2)

# 方案三：softmax与crossentropy是分开的，数值不稳定
predict = tf.math.softmax(logits, axis=1)
loss_3 = tf.reduce_mean(tf.losses.categorical_crossentropy(y_true, predict, from_logits=False))
print(loss_3)


