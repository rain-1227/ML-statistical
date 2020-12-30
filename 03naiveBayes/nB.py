import numpy as np
import pandas as pd

# 1.构建数据集
X_train=np.array([[1,"S"],
                  [1,"M"],
                  [1,"M"],
                  [1,"S"],
                  [1,"S"],
                  [2,"S"],
                  [2,"M"],
                  [2,"M"],
                  [2,"L"],
                  [2,"L"],
                  [3,"L"],
                  [3,"M"],
                  [3,"M"],
                  [3,"L"],
                  [3,"L"]])

y_train=np.array([-1,-1,1,1,-1,-1,-1,1,1,1,1,1,1,1,-1])
# 将数据转为DataFram形式方便对数据后面的处理
X = pd.DataFrame(X_train)
y = pd.DataFrame(y_train)
X_new = np.array([2, "S"])


class NB():
    def __init__(self, r):
        self.r = r
        self.y_ = np.unique(y_train)
        self.x_prob = dict()

    def fit(self):
        y_n = y.value_counts()  # 统计y序列中-1和1出现的次数 -1：9， 1：6
        self.y_p = (y_n + self.r) / (len(y) + len(self.y_)*self.r)  # 计算出先验概率：y=1和y=-1的概率
        for idx in X.columns:  # X.colums是一个rang(0, 2, 1)
            for c in self.y_:  # 这里c指的是y取-1，1时
                p_x_y = X[(y == c).values][idx].value_counts()  # 计算出X每列的几种取值出现的次数
                for i in p_x_y.index:
                    self.x_prob[(idx, i, c)] = (p_x_y[i] + self.r) / (y_n[c] + p_x_y.shape[0]*self.r)

    def predict(self, X_new):
        res = []
        for y in self.y_:
            p_y = self.y_p[y]
            p_xy = 1
            for idx, x in enumerate(X_new):
                p_xy *= self.x_prob[(idx, x, y)]
            res.append(p_y * p_xy)

        print(res)
        return self.y_[np.argmax(res)]


nb = NB(0.2)
nb.fit()
y_predict = nb.predict(X_new)
print(y_predict)




