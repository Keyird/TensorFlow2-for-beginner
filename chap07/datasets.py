import tensorflow as tf
from tensorflow import keras

# 加载mnist数据集
(x,y), (x_test, y_test) = keras.datasets.mnist.load_data()
print(x.shape, y.shape)
print(x.min(), x.max(), x.mean())

# 加载cifar数据集
(x,y), (x_test, y_test) = keras.datasets.cifar10.load_data()  # cifar10数据集
(x,y), (x_test, y_test) = keras.datasets.cifar100.load_data()  # cifar100数据集

# 打包成张量
db = tf.data.Dataset.from_tensor_slices(x_test)
next(iter(db))
db = tf.data.Dataset.from_trnsor_slices((x_test, y_test))
db = db.shuffle(10000)  # 打乱


def preprocess(x, y):
    """ 数据集预处理 """
    x = tf.cast(x, dtype=tf.float32)/255.
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=10)
    return x, y

db1 = db.map(preprocess)  # 批量预处理
db2 = db1.batch(31)  # 设置batch_size

def mnist_dataset():

    (x, y), (x_test, y_test) = keras.datasets.fashion_mnist.load_data()

    ds = tf.data.Dataset.from_tensor_slices((x, y))
    ds = ds.map(preprocess).shuffle(60000).batch(32)

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    ds_test = ds.map(preprocess).shuffle(60000).batch(32)

