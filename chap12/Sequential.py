import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

model = keras.Sequential([
        layers.Dense(2, activation="relu"),
        layers.Dense(3, activation="relu"),
        layers.Dense(4),
    ])

# detele some layers
model.pop()
print(len(model.layers))  # 2

# define a layer
layer = layers.Dense(3)
print(layer.weights)  # Empty

x = tf.ones((1, 4))  # input
y = layer(x)
print(layer.weights)  # Now it has weights, of shape (4, 3) and (3,)