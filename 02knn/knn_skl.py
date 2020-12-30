from sklearn.neighbors import KNeighborsClassifier
import numpy as np

# 创建训练数据数据
X_train = np.array([[5, 4],
                    [9, 6],
                    [4, 7],
                    [2, 3],
                    [8, 1],
                    [7, 2]])
y_train = np.array([1, 1, 1, -1, -1, -1])
# 待预测数据
X_new = np.array([[5, 3], [8, 2]])  # 可以对多个数据同时预测，比自己编写的方便多了

# 调用sklearn模块进行预测
for k in (1, 3, 5):
    # 创建一个k近邻分类器对象，n_neighbors是k值，n_jobs是计算进程数，-1指多进程，还有一些其它参数按照默认值即可
    clf = KNeighborsClassifier(n_neighbors=k, n_jobs=-1)
    clf.fit(X_train, y_train)  # 根据我们的数据，自动选择合适的计算距离的算法，比如线性扫描，KD数，球数
    print(clf.kneighbors(X_new))  # 返回预测点的K近邻点
    predict = clf.predict(X_new)  # 对预测点进行分类
    acc = clf.score(X_new, [1, 1])  # 输入测试集，评价训练效果
    print("预测正确率：{:.0%}".format(acc))
    print('k={}, 分类结果：{}'.format(k, predict))

