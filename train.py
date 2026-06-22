import cv2
import os
import numpy as np
from main import SeatbeltDetector


def load_camera_data(folder_path, img_size=64):
	images = []
	labels = []

	belt_path = os.path.join(folder_path, 'belt')
	if os.path.exists(belt_path):
		for file in os.listdir(belt_path):
			img_path = os.path.join(belt_path, file)
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			if img is not None:
				img_resized = cv2.resize(img, (img_size, img_size))
				images.append(img_resized.flatten())
				labels.append(1)

	no_belt_path = os.path.join(folder_path, 'no_belt')
	if os.path.exists(no_belt_path):
		for file in os.listdir(no_belt_path):
			img_path = os.path.join(no_belt_path, file)
			img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
			if img is not None:
				img_resized = cv2.resize(img, (img_size, img_size))
				images.append(img_resized.flatten())
				labels.append(0)

	X = np.array(images).T / 255.0
	Y = np.array(labels).reshape(1, -1)

	return X, Y


X_real, Y_real = load_camera_data("TestImages")
m_real = X_real.shape[1]
print(f"Загружено {m_real} реальных кадров. Форма X: {X_real.shape}")

model = SeatbeltDetector(input_size=4096, hidden_size=64)
model.train(X_real, Y_real, 2000, learning_rate=0.01)
model.save_model()
