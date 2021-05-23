import tensorflow as tf
from tensorflow import keras

def preprocess(x,y):
    """ x: 图片的路径，y：图片的数字编码 """
    x = tf.io.read_file(x)  # 读图片
    x = tf.image.decode_jpeg(x, channels=3)  # RGBA
    x = tf.image.resize(x, [244, 244])  # 图片缩放到 244x244
    y = tf.one_hot(y, depth=10)  # one_hot编码
    return x, y

(x,y), (x_test, y_test) = keras.datasets.mnist.load_data()

# 图片先缩放到稍大尺寸
x = tf.image.resize(x, [244, 244])
# 再随机裁剪到合适尺寸
x = tf.image.random_crop(x, [224, 224, 3])
print(x.shape)