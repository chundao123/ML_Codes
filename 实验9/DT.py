import numpy as np

# 定义决策树中的节点类
class Node:
    def __init__(self):
        # 节点是否为叶子节点
        self.isleaf = False
        # 节点代表的类别
        self.node_class = None
        # 节点使用的属性
        self.attribute = None
        # 属性的划分值
        self.partition_value = None
        # 左子节点和右子节点的索引
        self.childnode1 = None
        self.childnode2 = None
        # 节点的熵
        self.entropy = 0

# 定义决策树类
class DecisionTree:
    def __init__(self):
        # 存储决策树的所有节点
        self.dt = []

    # 计算熵
    def cal_entropy(self, Y):
        _, counts = np.unique(Y, return_counts=True)  # 获取类别和对应的计数
        probabilities = counts / Y.size  # 计算每个类别的概率
        entropy = -np.sum(probabilities * np.log2(probabilities))  # 计算熵
        return entropy, Y.size  # 返回熵和样本数量
    
    # 根据属性划分数据集，并计算最小熵
    def attrPartition(self, X, Y, attr_index):
        attr_value_set = np.unique(X[:,attr_index])  # 获取属性的所有值
        part_val_set = (attr_value_set[1:] + attr_value_set[:-1]) / 2  # 计算属性值的中点
        minEnt, n = self.cal_entropy(Y)  # 初始熵和样本数量
        partition_value = attr_value_set[0]  # 初始化划分值
        for i in range(attr_value_set.size - 1):
            ent1, n1 = self.cal_entropy(Y[X[:,attr_index] < part_val_set[i]])  # 左分支的熵和样本数量
            ent2, n2 = self.cal_entropy(Y[X[:,attr_index] >= part_val_set[i]])  # 右分支的熵和样本数量
            if minEnt > (ent1 * n1 + ent2 * n2) / n:  # 找到最小熵的划分值
                partition_value = part_val_set[i]
                minEnt = (ent1 * n1 + ent2 * n2) / n
        return partition_value, minEnt  # 返回划分值和最小熵

    # 对属性进行排序，选择熵最小的属性
    def attr_sort(self, X, Y, attr):
        minEnt = np.zeros(attr.size)  # 初始化熵数组
        for i in range(attr.size):
            _,minEnt[i] = self.attrPartition(X,Y,attr[i])  # 计算每个属性的最小熵
        return attr[np.argsort(minEnt)]  # 返回按熵排序的属性索引

    # 生成节点
    def NodeGenerate(self, X, Y, attr):
        curNodeindex = len(self.dt)  # 当前节点的索引
        self.dt.append(Node())  # 添加新节点
        self.dt[curNodeindex].entropy, _ = self.cal_entropy(Y)  # 计算当前节点的熵

        while(attr.size != 0 and np.unique(X[:,attr[0]]).size == 1):  # 如果属性值相同，则跳过该属性
            attr = attr[1:]
        if(attr.size == 0):  # 如果没有属性了，设置为叶子节点
            self.dt[curNodeindex].isleaf = True
            self.dt[curNodeindex].node_class = np.argmax(np.bincount(Y))  # 设置节点类别为多数类
            return

        if(self.dt[curNodeindex].entropy == 0):  # 如果熵为0，设置为叶子节点
            self.dt[curNodeindex].isleaf = True
            self.dt[curNodeindex].node_class = Y[0]  # 设置节点类别为当前类别
            return

        curAttr = self.attr_sort(X, Y, attr)  # 对属性进行排序
        
        self.dt[curNodeindex].attribute = curAttr[0]  # 设置当前节点使用的属性
        self.dt[curNodeindex].partition_value, _ = self.attrPartition(X, Y, curAttr[0])  # 计算划分值
        
        class1_index = (X[:,curAttr[0]] < self.dt[curNodeindex].partition_value)  # 左分支的索引
        class2_index = (X[:,curAttr[0]] >= self.dt[curNodeindex].partition_value)  # 右分支的索引
        X1 = X[class1_index, :]  # 左分支的数据
        Y1 = Y[class1_index]  # 左分支的标签
        X2 = X[class2_index, :]  # 右分支的数据
        Y2 = Y[class2_index]  # 右分支的标签

        if(Y1.size == 0):  # 如果左分支没有样本，设置为叶子节点
            self.dt[curNodeindex].isleaf = True
            self.dt[curNodeindex].node_class = np.argmax(np.bincount(Y))  # 设置节点类别为多数类
            return
        if(Y2.size == 0):  # 如果右分支没有样本，设置为叶子节点
            self.dt[curNodeindex].isleaf = True
            self.dt[curNodeindex].node_class = np.argmax(np.bincount(Y))  # 设置节点类别为多数类
            return

        self.dt[curNodeindex].childnode1 = len(self.dt)  # 设置左子节点的索引
        self.NodeGenerate(X1, Y1, curAttr[1:])  # 递归生成左子树
        self.dt[curNodeindex].childnode2 = len(self.dt)  # 设置右子节点的索引
        self.NodeGenerate(X2, Y2, curAttr[1:])  # 递归生成右子树

    # 训练模型
    def fit(self, trainX, trainY):
        attr_index = np.arange(trainX.shape[1])  # 获取所有属性的索引
        self.dt = []  # 清空决策树
        self.NodeGenerate(trainX, trainY, attr_index)  # 生成决策树

    # 根据节点进行预测
    def predNode(self, Nodeindex, sample):
        if(self.dt[Nodeindex].isleaf == True):  # 如果是叶子节点，返回节点类别
            return self.dt[Nodeindex].node_class
        if(sample[self.dt[Nodeindex].attribute] < self.dt[Nodeindex].partition_value):  # 根据属性值决定走哪个分支
            return self.predNode(self.dt[Nodeindex].childnode1, sample)
        else:
            return self.predNode(self.dt[Nodeindex].childnode2, sample)

    # 对测试集进行预测
    def predict(self, testX):
        predY = np.zeros(testX.shape[0])  # 初始化预测结果
        for i in range(predY.shape[0]):  # 对每个样本进行预测
            predY[i] = self.predNode(0, testX[i,:])  # 从根节点开始预测
        return predY  # 返回预测结果