import numpy as np
import matplotlib.pyplot as plt

# 送入100个点，计算平均损失
def compute_error_for_line_given_points(b, w, points):
    totalError = 0
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # 累计平方和误差
        totalError += (y - (w * x + b)) ** 2
    # 平均损失
    return totalError / float(len(points))

# 更新权值w,b
def step_gradient(b_current, w_current, points, learningRate):
    b_gradient = 0
    w_gradient = 0
    N = float(len(points))
    # 计算100个点的平均梯度
    for i in range(0, len(points)):
        x = points[i, 0]
        y = points[i, 1]
        # grad_b = 2(wx+b-y)
        b_gradient += (2/N) * ((w_current * x + b_current) - y)
        # grad_w = 2(wx+b-y)*x
        w_gradient += (2/N) * x * ((w_current * x + b_current) - y)
    # 更新权值w,b
    new_b = b_current - (learningRate * b_gradient)
    new_w = w_current - (learningRate * w_gradient)
    return [new_b, new_w]

# 迭代训练num_iterations次
def gradient_descent_runner(points, starting_b, starting_w, learning_rate, num_iterations):
    b = starting_b
    w = starting_w
    # 迭代num_iterations次
    for i in range(num_iterations):
        b, w = step_gradient(b, w, np.array(points), learning_rate)
    return [b, w]


if __name__ == '__main__':

    # 数据准备
    points = np.genfromtxt("data.csv", delimiter=",")

    # 参数初始化
    learning_rate = 0.0001
    initial_b = 0
    initial_w = 0
    num_iterations = 1000

    # 比较前后损失值
    print("训练开始： b = {0}, w = {1}, error = {2}"
          .format(initial_b, initial_w,compute_error_for_line_given_points(initial_b, initial_w, points)))
    print("Running...")
    # 迭代训练1000次，获得更新后的权值w,b
    [b, w] = gradient_descent_runner(points, initial_b, initial_w, learning_rate, num_iterations)
    print("训练 {0} 轮后： b = {1}, w = {2}, error = {3}".
          format(num_iterations, b, w,compute_error_for_line_given_points(b, w, points)))

    # 通过matplotlib绘制拟合结果
    plt.scatter(points[:, 0], points[:, 1])
    x = np.arange(0, 100)
    y = w * x + b
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title("y = wx + b")
    plt.plot(x, y, color='r', linewidth=2.5)
    plt.show()

    # 任意输入一个x，预测y
    print("请任意输入一个x：")
    x = eval(input())
    print("y={:.3f}".format(w*x+b))