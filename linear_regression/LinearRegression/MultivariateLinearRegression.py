import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from linear_regression import LinearRegression
import plotly
import plotly.graph_objs as go

# 从CSV文件（逗号分隔值）中读取数据，并将其转换为DataFrame对象
data = pd.read_csv('../data/world-happiness-report-2017.csv')

# 从数据集中随机抽样一部分数据，frac = 0.8表示抽取80%的数据作为样本
train_data = data.sample(frac=0.8)
# 从data数据中删除train_data的内容，剩下的座位测试数据集
test_data = data.drop(train_data.index)

input_param_name1 = 'Economy..GDP.per.Capita.'
input_param_name2 = 'Freedom'
output_param_name = 'Happiness.Score'

x_train = train_data[[input_param_name1, input_param_name2]].values  # 取出名为input_param_name1该列的数据，并返回二维数组
y_train = train_data[[output_param_name]].values

x_test = test_data[[input_param_name1, input_param_name2]].values  # 取出名为input_param_name1该列的数据，并返回出来一维数组
y_test = test_data[output_param_name].values


plot_training_trace = go.Scatter3d(
    x=x_train[:, 0].flatten(),
    y=x_train[:, 1].flatten(),
    z=y_train.flatten(),
    name='Training Set',
    mode='markers',
    marker={
        'size':10,
        'opacity':1,
        'line':{
            'color':'rgb(255, 255, 255)',
            'width':1
        },
    }
)

plot_test_trace = go.Scatter3d(
    x=x_test[:, 0].flatten(),
    y=x_test[:, 1].flatten(),
    z=y_test.flatten(),
    name='Test Set',
    mode='markers',
    marker={
        'size': 10,
        'opacity': 1,
        'line': {
            'color': 'rgb(255, 255, 255)',
            'width': 1
        },
    }
)

plot_layout = go.Layout(
    title='Date Sets',
    scene={
        'xaxis': {'title': input_param_name1},
        'yaxis': {'title': input_param_name2},
        'zaxis': {'title': output_param_name}
    },
    margin={'l': 0, 'r': 0, 'b': 0, 't': 0}
)

plot_data = [plot_training_trace, plot_test_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)

num_iterations = 500
learning_rate = 0.01

linear_regression = LinearRegression(x_train, y_train)
(theta, cost_history) = linear_regression.train(learning_rate, num_iterations)

print('开始时的损失：', cost_history[0])
print('训练后的损失：', cost_history[-1])

plt.plot(range(num_iterations), cost_history)
plt.xlabel('Iterations')
plt.ylabel('Cost')
plt.title('Gradient Descent Progress')
plt.show()

prediction_num = 10
x_min = x_train[:, 0].min()
x_max = x_train[:, 0].max()
y_min = x_train[:, 1].min()
y_max = x_train[:, 1].max()
x_axis = np.linspace(x_min, x_max, prediction_num)
y_axis = np.linspace(y_min, y_max, prediction_num)
x_predictions = np.zeros((prediction_num * prediction_num, 1))
y_predictions = np.zeros((prediction_num * prediction_num, 1))

x_y_index = 0
for x_index, x_value in enumerate(x_axis):
    for y_index, y_value in enumerate(y_axis):
        x_predictions[x_y_index] = x_value
        y_predictions[x_y_index] = y_value
        x_y_index += 1

z_predictions = linear_regression.predict(np.hstack((x_predictions, y_predictions)))

plot_predictions_trace = go.Scatter3d(
    x=x_predictions.flatten(),
    y=y_predictions.flatten(),
    z=z_predictions.flatten(),
    name='Prediction Plane',
    mode='markers',
    marker={
        'size': 1,
    },
    opacity=0.8,
    surfaceaxis=2,
)

plot_data = [plot_training_trace, plot_test_trace, plot_predictions_trace]
plot_figure = go.Figure(data=plot_data, layout=plot_layout)
plotly.offline.plot(plot_figure)
