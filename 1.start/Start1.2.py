import tensorflow as tf  # 导入TF库
from tensorflow.keras import datasets, layers, Model, losses, optimizers, metrics # 导入TF子库

(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()  # 加载数据集，返回的是两个元组，分别表示训练集和测试集

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)/255.  # 归一化，将像素值缩放到0~1
x_test = tf.convert_to_tensor(x_test, dtype=tf.float32)/255.

# x_train = x_train[..., tf.newaxis]  # 增加维度 [60000,28,28]->[60000,28,28,1]
# x_test = x_test[..., tf.newaxis]
print(x_train.shape, y_train.shape)

# 使用 tf.data 来将数据集切分为 batch个一组，并对数据集进行打乱
train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train)).shuffle(10000).batch(32)
test_ds = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(32)

# 网络搭建
class Network(Model):
    def __init__(self):
        super(Network, self).__init__()
        self.conv1 = layers.Conv2D(32, 3, activation='relu')
        self.flatten = layers.Flatten()
        self.fc1 = layers.Dense(128, activation='relu')
        self.fc2 = layers.Dense(10, activation='softmax')

    def call(self, x):
        x = tf.reshape(x, (-1, 28, 28, 1))
        x = self.conv1(x)
        x = self.flatten(x)
        x = self.fc1(x)
        y = self.fc2(x)
        return y

network = Network()

# 确定目标损失函数、优化器、评价标准
loss_object = losses.SparseCategoricalCrossentropy()
optimizer = optimizers.Adam()
# 训练集上的损失值、精确度
train_loss = metrics.Mean(name='train_loss')
train_accuracy = metrics.SparseCategoricalAccuracy(name='train_accuracy')
# 测试集上的损失值、精确度
test_loss = metrics.Mean(name='test_loss')
test_accuracy = metrics.SparseCategoricalAccuracy(name='test_accuracy')

# 训练
def train_step(images, labels):
    with tf.GradientTape() as tape:  # 建立梯度环境
        predictions = network(images)  # 前向计算
        loss = loss_object(labels, predictions)  # 计算损失
    gradients = tape.gradient(loss, network.trainable_variables)  # 计算网络中各个参数的梯度
    optimizer.apply_gradients(zip(gradients, network.trainable_variables))  # 更新网络参数
    train_loss(loss)  # 计算训练损失
    train_accuracy(labels, predictions)  # 计算训练精确度

# 测试
def test_step(images, labels):
    predictions = network(images)  # 前向计算
    t_loss = loss_object(labels, predictions)  # 计算当前轮上的损失
    test_loss(t_loss)  # 计算测试集上的损失
    test_accuracy(labels, predictions)  # 计算测试集上的准确率

EPOCHS = 5

for epoch in range(EPOCHS):
    # 在下一个epoch开始时，重置评估指标：所有损失值、精确度清零
    train_loss.reset_states()
    train_accuracy.reset_states()
    test_loss.reset_states()
    test_accuracy.reset_states()
    # 训练
    for images, labels in train_ds:
        train_step(images, labels)
    # 测试
    for test_images, test_labels in test_ds:
        test_step(test_images, test_labels)
        print('Accuracy:{}'.format(test_accuracy.result()))
    # 打印训练结果
    template = 'Epoch {}, Loss: {}, Accuracy: {}, Test Loss: {}, Test Accuracy: {}'
    print(template.format(epoch+1,
                          train_loss.result(), train_accuracy.result(),
                          test_loss.result(), test_accuracy.result()))