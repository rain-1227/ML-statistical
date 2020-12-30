import numpy as np
"""
使用k-近邻算法分类一个电影是爱情片还是动作片,每个电影两个属性：打斗镜头和接吻镜头[1, 101]指打斗镜头为一，接吻镜头为101
"""

class Knn():
    def __init__(self, train_d, label, test_d, k):
        """
        :param train_d: 训练集数据
        :param test_d: 要测试的数据
        :param k: k近邻的参数k
        """
        self.train_d = train_d
        self.test_d = test_d
        self.k = k
        self.label = label

    def predict(self):
        # 计算预测点与每个训练集中的点的欧氏距离
        test_d_n = np.tile(self.test_d, (self.train_d.shape[0], 1))  # 将test_d变成与train_d维度相同的数组，方便后面计算
        print(test_d_n)
        distance = ((test_d_n - self.train_d)**2).sum(1)  # 这就是预测点与每个训练集中的点的欧氏距离组成的一个一维数组
        index = distance.argsort()  # 返回距离数组中从小到大排序的索引值，这个索引值是之前distance数组中的元素的索引值
        class_dict = {}  # 定义一个空的类别字典来记录上述index中对应label出现的次数

        for i in range(self.k):
            key = self.label[index[i]]
            class_dict[key] = class_dict.get(key, 0) + 1
            # dict.get(key, 0)是获得对应键的值,如果键的值不存在,且不指定的话则返回None,这里指定的是0,如果存在的话,则返回这个存在的值而不是0

        class_dict_sort = sorted(class_dict.items(), key=lambda x: x[1], reverse=True)
        # items()函数以返回可遍历的(键, 值)元组组成的列表[('动作片', 2), ('爱情片', 1)]
        # key为函数，指定按照待排序元素的哪一项进行排序x指的是(键, 值)元组，[1]表明按照值进行排序，reverse表明排序是从大到小，默认是从小到大的
        print(class_dict_sort)
        return class_dict_sort[0][0]


# 创建数据
train = np.array([[1, 101], [5, 89], [108, 5], [115, 8]])
label = ['爱情片', '爱情片', '动作片', '动作片']
test = np.array([40, 30])

knn = Knn(train, label, test, 3)
result = knn.predict()
print(result)