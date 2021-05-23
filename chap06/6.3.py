import tensorflow as tf

""" 张量比较 """

# 1. tf.equal()
out = tf.random.normal([100, 10])  # 随机生成一个张量，用来模拟输出结果
out = tf.nn.softmax(out, axis=1)  # 输出转换为概率值，缩放到0-1，且概率和为1
pred = tf.argmax(out, axis=1)  # 选取预测值（概率维度上的最大值），得到的是长度为100的向量

# 模拟100个真实标签，采用上节讲的均匀分布tf.random.uniform()来创建长度为100，值属于[0,9]区间的向量
y = tf.random.uniform([100], dtype=tf.int64, maxval=10) # 标签
# 通过 tf.equal(pred, y) 可以比较这 2个张量是否相等
out = tf.equal(pred, y)  # 预测值与真实值比较


# 2. tf.cast()
out = tf.cast(out, dtype=tf.float32)  # 布尔型转 int 型
correct_num = tf.reduce_sum(out)  # 统计 True 的个数


