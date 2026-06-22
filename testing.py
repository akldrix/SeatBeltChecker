from main import SeatbeltDetector

model = SeatbeltDetector(input_size=4096, hidden_size=64)

model.load_model("model_weights.npz")

photo_path = "TestImages/test.jpg"

result = model.predict(photo_path)
print(f"Анализ фото {photo_path}:")
print(result)
