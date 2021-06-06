import numpy as np
import matplotlib.pyplot as plt

def sigmoid(z):
	"""
	сигмоидная функция активации
	:param z: wx + b
	:return: Y'
	"""
	return 1 / (1 + np.exp(-z))

def forward_propagation(X, W, b):
	"""
	Вычисляет операцию прямого распространения прецептрона и
	возвращает результат после применения сигмовидной функции активации
	:param X: входные данные
	:param W: вес
	:param b: смещение
	:return: прогноз
	"""
	weighted_sum = np.dot(X, W) + b
	prediction = sigmoid(weighted_sum)
	return prediction

def calculate_error(y, y_prediction):
	"""
	Находит ошибку бинарной кросс-энтропии
	:param y: метка
	:param y_prediction: прогноз
	:return: ошибка
	"""
	loss = np.sum(- y * np.log(y_prediction) - (1 - y) * np.log(1 - y_prediction)) # вычисляем ошибку
	return loss

def gradient(X, Y, Y_predicted):
	"""
	Градиент весов и смещения
	:param X: Входные данные
	:param Y: метка
	:param Y_predicted: прогноз
	:return: производные весов и смещения
	"""
	Error = Y_predicted - Y
	dW = np.dot(X.T, Error)
	db = np.sum(Error)
	return dW, db

def update_parameters(W, b, dW, db, learning_rate):
	"""
	Обновляем веса и смещение
	:param W: вес
	:param b: смещение
	:param dW: производная веса
	:param db: производная смещения
	:param learning_rate: скорость обучения
	:return: весб смещение
	"""
	W = W - learning_rate * dW #обновляем вес
	b = b - learning_rate * db #обновляем смещение
	return W, b

def train(X, Y, learning_rate, W, b, epochs, losses):
	"""
	Обучение перцептрона с помощью пакетного обговления
	:param X: входные данные
	:param Y: метки
	:param learning_rate: скорость обучения
	:param W: веса
	:param b: смещение
	:param epochs: эпохи
	:param losses: потери
	:return: Веса, смещение, потери
	"""
	for i in range(epochs):		#цикл по всем эпохам
		Y_predicted = forward_propagation(X, W, b) 		#вычисляем прямое распространение
		losses[i, 0] = calculate_error(Y, Y_predicted) 	#вычисляем ошибку
		dW, db = gradient(X, Y, Y_predicted) 	#вычисляем градиент
		W, b = update_parameters(W, b, dW, db, learning_rate) #обновляем параметры
	return W, b, losses

# Initialize parameters
# features
X = np.array(
   [[2.78, 2.55],
	[1.46, 2.36],
	[3.39, 4.40],
	[1.38, 1.85],
	[3.06, 3.00],
	[7.62, 2.75],
	[5.33, 2.08],
	[6.92, 1.77],
	[8.67, -0.24],
	[7.67, 3.50]])

Y = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1]) # target label
weights = np.array([0.0, 0.0]) # weights of perceptron
bias = 0.0 # bias value
epochs = 10000 # total epochs
learning_rate = 0.01 # learning rate
losses = np.zeros((epochs, 1)) # compute loss
print("Before training")
print("weights:", weights, "bias:", bias)
print("Target labels:", Y)
W, b, losses = train(X, Y, learning_rate, weights, bias, epochs, losses)

# Evaluating the performance
plt.figure()
plt.plot(losses)
plt.xlabel("EPOCHS")
plt.ylabel("Loss value")
plt.show()
plt.savefig('output/legend.png')

print("\nAfter training")
print("weights:", W, "bias:", b)
# Predict value
A2 = forward_propagation(X, W, b)
pred = (A2 > 0.5) * 1

print("Predicted labels:", pred)