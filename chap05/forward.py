import tensorflow as tf
from tensorflow.keras import datasets


# 加载数据集
(x, y), (x_test, y_test) = datasets.mnist.load_data()
# 转为张量
x = tf.convert_to_tensor(x, dtype=tf.float32) / 255.
y = tf.convert_to_tensor(y, dtype=tf.float32)
# 构建每一个batch数据
train_db = tf.data.Dataset.from_tensor_slices((x, y)).batch(128)
train_iter = iter(train_db)
sample = next(train_iter)
print('batch:', sample[0].shape, sample[1].shape)

# 初始化变量
w1 = tf.Variable(tf.random.truncated_normal([784, 256], stddev=0.1))
b1 = tf.Variable(tf.zeros([256]))
w2 = tf.Variable(tf.random.truncated_normal([256, 128], stddev=0.1))
b2 = tf.Variable(tf.zeros([128]))
w3 = tf.Variable(tf.random.truncated_normal([128, 10], stddev=0.1))
b3 = tf.Variable(tf.zeros([10]))

# 训练迭代10个epoches
for epoch in range(10):
    # x:[128,28,28], y:[128]
    for step, (x, y) in enumerate(train_db):   # step = nums/batch
        x = tf.reshape(x, [-1, 28*28])
        # 构建梯度环境
        with tf.GradientTape() as tape:
            # 第一层: [b,784]*[784,256]+[256] => [b,256]
            h1 = x@w1 + tf.broadcast_to(b1, [x.shape[0], 256])
            h1 = tf.nn.relu(h1)
            # 第二层：[b, 256] => [b, 128]
            h2 = h1@w2 + b2
            h2 = tf.nn.relu(h2)
            # 输出层：[b, 128] => [b,10]
            out = h2@w3 + b3
            # 将输出转换成热独码
            y_onehot = tf.one_hot(y, depth=10)
            # 建立mse损失函数
            loss = tf.square(y_onehot - out)
            loss = tf.reduce_mean(loss)

        # 计算梯度
        grads = tape.gradient(loss, [w1, b1, w2, b2, w3, b3])
        # 参数更新
        lr = 1e-3
        w1.assign_sub(lr * grads[0])
        b1.assign_sub(lr * grads[1])
        w2.assign_sub(lr * grads[2])
        b2.assign_sub(lr * grads[3])
        w3.assign_sub(lr * grads[4])
        b3.assign_sub(lr * grads[5])

        if step % 100 == 0:
            print(epoch, step, 'loss: ', float(loss))



