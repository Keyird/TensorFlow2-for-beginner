"""
note: compile/fit/evaluate
author: AI JUN
date: 2022/1/5
"""

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

# 模型的装配
model.compile(
    optimizer=keras.optimizers.RMSprop(),  # Optimizer
    loss=keras.losses.SparseCategoricalCrossentropy(),
    metrics=['accuracy'],
)

# 模型的训练
history = model.fit(
    x_train,
    y_train,
    batch_size=64,
    epochs=10,
    validation_data=(x_val, y_val),  # at the end of each epoch
)

# 评估测试集
print("Evaluate on test data")
results = model.evaluate(x_test, y_test, batch_size=128)
print("test loss, test acc:", results)

# # 随机预测三张图片
# print("Generate predictions for 3 samples")
# predictions = model.predict(x_test[:4])
# print("predictions shape:", predictions.shape)


