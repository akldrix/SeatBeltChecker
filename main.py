import numpy as np


class SeatbeltDetector:
	def __init__(self, input_size, hidden_size, output_size=1):
		self.W1 = np.random.randn(hidden_size, input_size) * 0.01
		self.b1 = np.zeros((hidden_size, 1))

		self.W2 = np.random.randn(output_size, hidden_size) * 0.01
		self.b2 = np.zeros((output_size, 1))

	def relu(self, Z):
		return np.maximum(0, Z)

	def relu_derivative(self, Z):
		return Z > 0

	def sigmoid(self, Z):
		Z_clipped = np.clip(Z, -500, 500)
		return 1 / (1 + np.exp(-Z_clipped))

	def forward(self, X):
		self.Z1 = np.dot(self.W1, X) + self.b1

		self.A1 = self.relu(self.Z1)

		self.Z2 = np.dot(self.W2, self.A1) + self.b2

		self.A2 = self.sigmoid(self.Z2)

		return self.A2

	def compute_loss(self, A2, Y):
		m = Y.shape[1]

		epsilon = 1e-15

		log_probs = Y * np.log(A2 + epsilon) + (1 - Y) * np.log(1 - A2 + epsilon)

		loss = -(1 / m) * np.sum(log_probs)
		return np.square(loss)

	def backward(self, X, Y):
		m = X.shape[1]

		dZ2 = self.A2 - Y

		dW2 = (1 / m) * np.dot(dZ2, self.A1.T)

		db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)

		dZ1 = np.dot(self.W2.T, dZ2) * self.relu_derivative(self.Z1)

		dW1 = (1 / m) * np.dot(dZ1, X.T)

		db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

		return {"dW1": dW1, "db1": db1, "dW2": dW2, "db2": db2}

	def update_params(self, grads, learning_rate):

		self.W1 = self.W1 - learning_rate * grads["dW1"]
		self.b1 = self.b1 - learning_rate * grads["db1"]
		self.W2 = self.W2 - learning_rate * grads["dW2"]
		self.b2 = self.b2 - learning_rate * grads["db2"]

	def train(self, X, Y, num_epochs, learning_rate):
		for epoch in range(num_epochs):
			A2 = self.forward(X)

			loss = self.compute_loss(A2, Y)

			grads = self.backward(X, Y)

			self.update_params(grads, learning_rate)

			if epoch % 100 == 0:
				print("Эпоха {i} | Ошибка (Loss): {loss:.4f}".format(i=epoch, loss=loss))

	def save_model(self, filename="model_weights.npz"):
		np.savez(filename, W1=self.W1, b1=self.b1, W2=self.W2, b2=self.b2)
		print("Веса успешно сохранены в файл: {}".format(filename))

	def load_model(self, filename="model_weights.npz"):
		data = np.load(filename)
		self.W1 = data["W1"]
		self.b1 = data["b1"]
		self.W2 = data["W2"]
		self.b2 = data["b2"]
		print("Веса из {} загружены".format(filename))

	def predict(self, image):
		import cv2
		img = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
		if img is None:
			return "Путь к картинке неверно указан"
		img_resized = cv2.resize(img, (64, 64))

		X_test = img_resized.flatten().reshape(-1, 1) / 255.0

		prediction = self.forward(X_test)
		probability = np.squeeze(prediction)

		if probability >= 0.5:
			return "Ремень пристегнут"
		else:
			return "Ремень не пристегнут"
