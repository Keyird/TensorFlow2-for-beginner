import tensorflow as tf  # 导入TF库
from tensorflow.keras import datasets, Sequential, layers  # 导入TF子库

# 数据集预处理
def preprocess(x, y):
    x = tf.cast(x, dtype=tf.float32)/255.  # 归一化，将像素值缩放到0~1

    y = tf.cast(y, dtype=tf.int32)
    return x, y

# 数据集准备
def mnist_dataset():
    (x, y), (x_test, y_test) = datasets.mnist.load_data()
    db_train = tf.data.Dataset.from_tensor_slices((x, y))
    db_train = db_train.map(preprocess).shuffle(60000).batch(32)

    db_test = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    db_test = db_test.map(preprocess).shuffle(60000).batch(32)
    return db_train, db_test

# 模型搭建
network = Sequential([
    layers.Flatten(input_shape=(28, 28)),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.2),
    layers.Dense(10, activation='softmax')
])

# 数据集准备
db_train, db_test = mnist_dataset()

# 模型的装配
network.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 模型的训练
network.fit(db_train, epochs=5)

# 模型的评估
network.evaluate(db_test, verbose=2)