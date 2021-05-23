import tensorflow as tf
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

""" 高阶操作 """

# 1. tf.gather()
# [b, w, h, c]
images = tf.random.normal([8, 28, 28, 3])

# 取第0至第3张图片
images_0_3 = tf.gather(images, [0, 1, 2, 3], axis=0)
print("images_0_3的shape：", images_0_3.shape)

# 也可通过索引来实现
images_0_3 = images[:4,...]
print("images_0_3的shape：", images_0_3.shape)

# 对第0和第2通道进行提取
out_image = tf.gather(images_0_3, [0, 2], axis=3)
print("out_image的shape：", out_image.shape)


# 2. tf.gather_nd()
# 共有 4 个班级，每个班级 35 个学生，8 门科目，保存成绩册的张量 shape 为[4,35,8]
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32)
# 抽查第 2 个班级的第 2 个同学的所有科目，第 3 个班级的第 3 个同学的所有科目，第 4 个班级的第 4 个同学的所有科目
y = tf.gather_nd(x,[[1,1],[2,2],[3,3]])
print("y：", y)
# 抽出班级 1，学生 1 的科目 2；班级 2，学生 2 的科目 3；班级 3，学生 3 的科目 4 的成绩，共有 3 个成绩数据
y1 = tf.gather_nd(x,[[1,1,2],[2,2,3],[3,3,4]])
print("y1：", y1)


# 3. tf.boolean_mask()
x = tf.random.uniform([4,35,8],maxval=100,dtype=tf.int32)
y2 = tf.boolean_mask(x, mask=[True, False, True, False], axis=0)
print("y2.shape：", y2.shape)


# 4. scatter_nd()
# 构造需要刷新数据的位置
indices = tf.constant([[4], [3], [1], [7]])
# 构造需要写入的数据
updates = tf.constant([4, 5, 1, 8])
# 在长度为 8 的全 0 向量上根据 indices 写入 updates
out = tf.scatter_nd(indices, updates, [8])
print("out:", out)


# 5. meshgrid()
x = tf.linspace(-8.,8,100) # 设置 x 坐标的间隔
y = tf.linspace(-8.,8,100) # 设置 y 坐标的间隔
x,y = tf.meshgrid(x,y) # 生成网格点，并拆分后返回
# x.shape, y.shape # 打印拆分后的所有点的 x,y 坐标张量 shape

z = tf.sqrt(x**2+y**2)
z = tf.sin(z)/z  # sinc 函数实现

fig = plt.figure()
ax = Axes3D(fig)

fig = plt.figure()
ax = Axes3D(fig)
# 根据网格点绘制 sinc 函数 3D 曲面
ax.contour3D(x.numpy(), y.numpy(), z.numpy(), 50)
plt.show()
