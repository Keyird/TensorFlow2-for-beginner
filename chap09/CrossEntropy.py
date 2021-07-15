import tensorflow as tf

loss_1 = tf.losses.categorical_crossentropy([0,1,0,0], [0.25, 0.25, 0.25, 0.25])
loss_2 = tf.losses.categorical_crossentropy([0,1,0,0], [0, 0.99, 0, 0.01])
loss_3 = tf.losses.categorical_crossentropy([0,1,0,0], [0.99, 0, 0, 0.01])

print(loss_1)
print(loss_2)
print(loss_3)

