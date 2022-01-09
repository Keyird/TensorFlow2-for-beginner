"""
note: compile/fit/evaluate
author: AI JUN
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

# 模型搭建
def get_uncompiled_model():
    inputs = keras.Input(shape=(784,), name="input")
    x = layers.Dense(64, activation="relu", name="layer1")(inputs)
    x = layers.Dense(64, activation="relu", name="layer2")(x)
    outputs = layers.Dense(10, activation="softmax", name="predictions")(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model

# 模型装配
if True:
    # 自定义损失函数只需要y_true, y_pred两个参数
    def custom_mean_squared_error(y_true, y_pred):
        return tf.math.reduce_mean(tf.square(y_true - y_pred))
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.Adam(), loss=custom_mean_squared_error, metrics=['accuracy'])
else:
    # 需要其他参数时，则可以将 tf.keras.losses.Loss 类子类化
    class CustomMSE(keras.losses.Loss):
        def __init__(self, regularization_factor=0.1, name="custom_mse"):
            super().__init__(name=name)
            self.regularization_factor = regularization_factor

        def call(self, y_true, y_pred):
            mse = tf.math.reduce_mean(tf.square(y_true - y_pred))
            reg = tf.math.reduce_mean(tf.square(0.5 - y_pred))
            return mse + reg * self.regularization_factor
    model = get_uncompiled_model()
    model.compile(optimizer=keras.optimizers.Adam(), loss=CustomMSE(), metrics=['accuracy'])

# 模型训练
y_train_one_hot = tf.one_hot(y_train, depth=10)
model.fit(x_train, y_train_one_hot, batch_size=64, epochs=5)

# 评估测试集
print("Evaluate on test data")
y_test_one_hot = tf.one_hot(y_test, depth=10)
results = model.evaluate(x_test, y_test_one_hot, batch_size=128)
print("test loss, test acc:", results)
