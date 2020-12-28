import numpy as np
import matplotlib.pyplot as plt


# 定义一个感知机类
class MyPerception():
    def __init__(self):
        self.w = np.zeros(train.shape[1])
        self.b = 0
        self.lr = 1

    def fit(self, train, y):
        i = 0
        while i < train.shape[0]:
            # 判断该点是否为误分类点，如果是那么就开始更新参数，如果不是就不更新参数
            if (y[i]*(self.w @ train[i].T + self.b)) <= 0:
                self.w += self.lr * y[i] *train[i].T
                self.b += self.lr*y[i]
                i = 0  # 如果该点是误分类点，更新好模型后要重新从数据集中第一个元素取值

            else:
                i += 1

        return self.w, self.b


def draw(train, y):
    # 画出训练数据中的点
    plt.scatter(train[:2, 0], train[:2, 1], marker='o', label='1')
    plt.scatter(train[2, 0], train[2, 1], marker='x', label='-1')

    # 画出分离超平面 w*x + b = 0 即 w0*x0 + w1*x1 + b = 0
    x0 = np.array([0, 6])  # 生成分离超平面上两点的x0值
    x1 = -(b + w[0]*x0) / w[1]  # 生成两点对应的x1值
    plt.plot(x0, x1, color='red')

    plt.axis([0, 6, 0, 4])  # 设置两坐标轴的起始值，后面两个是x1轴的起始值。使得看起来美观一些
    plt.xlabel('x0')
    plt.ylabel('x1')
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # 创造一个数据集
    train = np.array([[3, 3], [4, 3], [1, 1]])
    y = np.array([1, 1, -1])

    # 创建感知机类
    perception = MyPerception()
    w, b = perception.fit(train, y)

    print(w, b)
    draw(train, y)








