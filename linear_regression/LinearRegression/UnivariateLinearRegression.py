import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression

# 从CSV文件（逗号分隔值）中读取数据，并将其转换为DataFrame对象
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 从数据集中随机抽样一部分数据，frac = 0.8表示抽取80%的数据作为样本
train_data = data.sample(frac=0.8)
# 从data数据中删除train_data的内容，剩下的座位测试数据集
test_data = data.drop(train_data.index)

input_param_name1 = 'Economy..GDP.per.Capita.'
input_param_name2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name1]].values  # 取出名为input_param_name1该列的数据，并返回二维数组
y_train = train_data[[output_param_name]].values

x_test = test_data[input_param_name1].values  # 取出名为input_param_name1该列的数据，并返回出来一维数组
y_test = test_data[output_param_name].values

num_iteration = 600
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iteration)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

prediction_num = 100
x_predictions = np.linspace(x_train.min(), x_train.max(), prediction_num).reshape(prediction_num, 1)  # 生成一个等间隔的一维数组
y_predictions = linear_regression.predict(x_predictions)

print('预测的模型为：\n', linear_regression.theta)

plt.figure(figsize=(12, 10))
plt.subplot(2, 2, 1)
plt.scatter(x_train, y_train, label='Train data')  # 创建散点图 Train data
plt.scatter(x_test, y_test, label='Test data')  # 创建散点图 Test data
plt.xlabel(input_param_name1)  # 设置x轴标签名为 Economy..GDP.per.Capita
plt.ylabel(output_param_name)  # 设置y轴标签名为 Happiness.Score
plt.title('Happy')  # 设置标题名为 Happy
plt.legend()  # 添加图例

plt.subplot(2, 2, 2)
plt.plot(range(num_iteration), cost_history)  # 创建折线图
plt.xlabel('Iter')
plt.ylabel('cost')
plt.title('GD')

plt.subplot(2, 2, 3)
plt.scatter(x_train, y_train, label='Train data')  # 创建散点图 Train data
plt.scatter(x_test, y_test, label='Test data')  # 创建散点图 Test data
plt.plot(x_predictions, y_predictions, 'r', label='Prediction')
plt.xlabel(input_param_name1)  # 设置x轴标签名为 Economy..GDP.per.Capita
plt.ylabel(output_param_name)  # 设置y轴标签名为 Happiness.Score
plt.title('Happy')  # 设置标题名为 Happy
plt.legend()  # 添加图例
plt.show()  # 展示图像



