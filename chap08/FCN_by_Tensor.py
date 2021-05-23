import tensorflow as tf

# 输入层
# 输入两个样本，每个样本的特征长度是784
x = tf.random.normal([2, 784])

# 中间层
# 中间层节点为500，所以定义权重举阵w1的shape为[784, 500]
w1 = tf.Variable(tf.random.truncated_normal([784, 500], stddev=0.1))
# 定义中间层的偏置
b1 = tf.Variable(tf.zeros([500]))
# 中间层输出
o1 = tf.matmul(x, w1) + b1
#  激活函数
o1 = tf.nn.relu(o1) # 激活函数

# 输出层
# 输出层节点为10，进行10分类，故权重举阵w2的shape为[500,10]
w2 = tf.Variable(tf.random.truncated_normal([500, 10], stddev=0.1))
# 定义中间层的偏置
b2 = tf.Variable(tf.zeros([10]))
# 中间层输出
o2 = tf.matmul(o1, w2) + b2
# 激活函数
o2 = tf.nn.relu(o2)

# 打印输出shape：[2,10]
print(o2.shape)
