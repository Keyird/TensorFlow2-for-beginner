import tensorflow as tf

"""  对一维向量进行排序  """
# 创建列表 [0,1,2,3,4]，并打乱顺序
a = tf.random.shuffle(tf.range(5))
print("打乱后的列表:", a)

# tf.sort()
# 升序
b = tf.sort(a)
print("升序排列后的列表：", b)

# 降序
c = tf.sort(a, direction="DESCENDING")
print("降序排列后的列表：", c)

# tf.argsort()
arise_index = tf.argsort(a)
print("升序列表各元素在原列表a中的索引号：", arise_index)

descend_index = tf.argsort(a, direction="DESCENDING")
print("降序列表各元素在原列表a中的索引号：", descend_index)

# tf.gather()
Arise_List = tf.gather(a, arise_index)
print("升序列表：", Arise_List)

DES_List = tf.gather(a, descend_index)
print("降序列表：", DES_List)


""" 对二维数组进行排序 """
x = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
print("二维矩阵x：", x)
x_arise = tf.sort(x)
print("排序后的二维矩阵x：", x_arise)
x_arise_index = tf.argsort(x)
print("升序列表各元素在原列表x中的索引号：", x_arise_index)


""" 获取top-K的数值以及索引 """
x = tf.random.uniform([3, 3], maxval=10, dtype=tf.int32)
print("二维矩阵x：", x)
res = tf.math.top_k(x, 2)
print("top-2值：", res.values)
print("top-2的索引值：", res.indices)