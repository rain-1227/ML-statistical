from sklearn.linear_model import Perceptron
import numpy as np

train = np.array([[3, 3], [3, 4], [1, 1]])
y = np.array([1, 1, -1])

"""
penalty是正则化，可以选l1正则化：稀疏特征值数量，l2正则化：平均权重值使权重值不要相差过大，alpha是正则化系数，值越大表明对模型约束越大
eta0是学习率(0,1]，tol是终止条件：上一次迭代损失值和这一次迭代损失之之差<tol，max_iter是最大迭代次数，如果tol不为0，则默认为1000
"""
# perceptron = Perceptron(penalty="l2", alpha=0.01, eta0=1, max_iter=50, tol=1e-3)
perceptron = Perceptron()
perceptron.fit(train, y)

print('w:{}, b:{}, n_iter:{}, acc:{:.1%}'.format(perceptron.coef_, perceptron.intercept_,
                                                 perceptron.n_iter_, perceptron.score(train, y)))

print(perceptron.predict(np.array([[2, 0]])))  # predict()方法用来给出新样本的预测值