import tensorflow as tf

""" 设置下限 tf.maximum() """
a = tf.range(9)
print("a:", a)
b = tf.maximum(a, 2)  # 设置下限是2
print("下限限制为2后:", b)


""" 激活函数ReLU """
def relu(x):
    return tf.maximum(x, 0)

x = a - 5
print("x：", x)
y = relu(x)
print("y：", y)


""" 设置上限 tf.minimum() """
b1 = tf.minimum(a, 6)
print("上限限制为6后:", b1)


""" 同时设置上下限 tf.clip_by_value() """
b2 = tf.clip_by_value(a, 5, 8)
print("下限设置为5，上限设置为8后:", b2)



