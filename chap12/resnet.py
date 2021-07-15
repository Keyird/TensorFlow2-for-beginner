import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, optimizers, datasets, Sequential, metrics

config = tf.compat.v1.ConfigProto(gpu_options=tf.compat.v1.GPUOptions(allow_growth=True))
sess = tf.compat.v1.Session(config=config)

def preprocess(x, y):
    # [-1~1]
    x = tf.cast(x, dtype=tf.float32) / 255. - 0.5
    y = tf.cast(y, dtype=tf.int32)
    y = tf.one_hot(y, depth=100)
    return x,y

# 数据集加载与准备
(x,y), (x_test, y_test) = datasets.cifar100.load_data()
y = tf.squeeze(y, axis=1)
y_test = tf.squeeze(y_test, axis=1)
print(x.shape, y.shape, x_test.shape, y_test.shape)
# 训练集
train_db = tf.data.Dataset.from_tensor_slices((x,y))
train_db = train_db.shuffle(50000).map(preprocess).batch(128)
print(y)
# 测试集
test_db = tf.data.Dataset.from_tensor_slices((x_test,y_test))
test_db = test_db.map(preprocess).batch(128)
print(y_test)

base_model = keras.applications.Xception(
    weights='imagenet',
    include_top=False,
    pooling='avg')

# Freeze the base model
base_model.trainable = False

# Use a Sequential model to add a trainable classifier on top
model = keras.Sequential([
    base_model,
    layers.Dense(100),
])


model.compile(optimizer=optimizers.Adam(lr=0.01),  # 指定优化器
                loss=tf.losses.CategoricalCrossentropy(from_logits=True),  # 指定采用交叉熵损失函数，包含Softmax
                metrics=['accuracy'])  # 指定评价指标为准备率

# 4.模型训练
history = model.fit(train_db, epochs=20, validation_data=test_db, validation_freq=2)
model.evaluate(test_db)  # 打印输出loss和accuracy

