import numpy as np
from features.prepare_for_training import prepare_for_training


class LinearRegression:

    def __init__(self, data, labels, polynomial_degree=0, sinusoid_degree=0, normalize_data=True):
        """
        1. 对数据进行预处理操作
        2. 先得到所有的特征个数
        3. 初始化参数矩阵
        """
        (data_processed,
         features_mean,
         features_deviation) = prepare_for_training(data, polynomial_degree, sinusoid_degree, normalize_data=True)

        self.data = data_processed
        self.labels = labels
        self.features_mean = features_mean  # 均值
        self.features_deviation = features_deviation  # 标准差
        self.polynomial_degree = polynomial_degree
        self.sinusoid_degree = sinusoid_degree
        self.normalize_data = normalize_data

        num_features = self.data.shape[1]  # 数组data的列数
        self.theta = np.zeros((num_features, 1))  # 创建一个num_features行，1列的矩阵，元素全为0

    def train(self, alpha, num_iterations=500):
        """
        训练模块，执行梯度下降, alpha 学习率， num_iterations迭代次数
        """
        cost_history = self.gradient_descent(alpha, num_iterations)
        return self.theta, cost_history

    def gradient_descent(self, alpha, num_iterations):
        """
        实际迭代模块，会迭代num_iterations次
        """
        cost_history = []
        for _ in range(num_iterations):
            self.gradient_step(alpha)
            cost_history.append(self.cost_function(self.data, self.labels))
        return cost_history

    def gradient_step(self, alpha):
        """
        梯度 每步计算，梯度下降参数更新计算方法，注意是矩阵运算
        """
        num_examples = self.data.shape[0]  # 一共的样本个数，数组data的行数
        prediction = LinearRegression.hypothesis(self.data, self.theta)  # 预测值 = 当前数据 * 当前theta
        delta = prediction - self.labels  # 残差（delta） = 预测值 - 真实值
        theta = self.theta - alpha * (1 / num_examples) * (np.dot(delta.T, self.data)).T  # 疑惑暂不明白
        self.theta = theta

    # 损失计算
    def cost_function(self, data, lables):
        num_examples = data.shape[0]
        delta = LinearRegression.hypothesis(self.data, self.theta) - lables
        cost = (1 / 2) * np.dot(delta.T, delta) / num_examples  # 损失函数   疑惑暂不明白
        return cost[0][0]

    @staticmethod
    def hypothesis(data, theta):
        predictions = np.dot(data, theta)
        return predictions

    # 获取当前损失
    def get_cost(self, data, lables):
        data_process = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        return self.cost_function(data_process, lables)

    def predict(self, data):
        data_process = prepare_for_training(data, self.polynomial_degree, self.sinusoid_degree, self.normalize_data)[0]
        predictions = LinearRegression.hypothesis(data_process, self.theta)
        return predictions
