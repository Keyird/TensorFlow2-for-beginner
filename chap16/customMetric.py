"""
note: 自定义指标，用compile进行装配
"""
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# 数据集准备
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

x_train = x_train.reshape(60000, 784).astype("float32") / 255.
x_test = x_test.reshape(10000, 784).astype("float32") / 255.

y_train = y_train.astype("float32")
y_test = y_test.astype("float32")

# 训练集
x_train = x_train[:-10000]
y_train = y_train[:-10000]

# 验证集
x_val = x_train[-10000:]
y_val = y_train[-10000:]

# 网络搭建
model = keras.Sequential([
    layers.Dense(64, activation="relu", name="layer1"),
    layers.Dense(64, activation="relu", name="layer2"),
    layers.Dense(10, activation="softmax", name="predictions"),
])
model.build(input_shape=[None, 28*28])

# 自定义评价指标
class CTP(keras.metrics.Metric):
    def __init__(self, name="categorical_true_positives", **kwargs):
        super(CTP, self).__init__(name=name, **kwargs)
        self.true_positives = self.add_weight(name="ctp", initializer="zeros")

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_pred = tf.reshape(tf.argmax(y_pred, axis=1), shape=(-1, 1))
        values = tf.cast(y_true, "int32") == tf.cast(y_pred, "int32")
        values = tf.cast(values, "float32")
        if sample_weight is not None:
            sample_weight = tf.cast(sample_weight, "float32")
            values = tf.multiply(values, sample_weight)
        self.true_positives.assign_add(tf.reduce_sum(values))

    def result(self):
        return self.true_positives

    def reset_states(self):
        # The state of the metric will be reset at the start of each epoch.
        self.true_positives.assign(0.0)

# 模型装配
model.compile(
    optimizer=keras.optimizers.RMSprop(learning_rate=1e-3),
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy', CTP()],
)

# 模型训练
model.fit(x_train, y_train, batch_size=64, epochs=5)