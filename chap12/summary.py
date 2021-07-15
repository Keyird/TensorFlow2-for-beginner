import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# solution1
model = keras.Sequential()
model.add(layers.Dense(64, activation="relu", input_shape=(10,)))
#model.add(keras.Input(shape=(10, )))
#model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(32, activation="relu"))
model.add(layers.Dense(8, activation="relu"))
model.summary()

# solution2
model = keras.Sequential([
    layers.Reshape(target_shape=(28 * 28,), input_shape=(28, 28)),
    layers.Dense(64, activation=tf.nn.relu),
    layers.Dense(32, activation=tf.nn.relu),
    layers.Dense(10)
])
model.build(input_shape=[None, 28*28])
model.summary()
