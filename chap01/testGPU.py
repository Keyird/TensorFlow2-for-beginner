import tensorflow as tf

# 查看gpu数
gpus = tf.config.list_physical_devices('GPU')
print(gpus)

# 设置训练所需的GPU的memory随需求而增长
tf.config.experimental.set_memory_growth(gpus[0], True)
