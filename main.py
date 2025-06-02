import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import argparse
import sys

# Загрузка сохраненной модели
model = tf.keras.models.load_model('E:\\pythonProject\\pythonProject\\CompliteCAR\\logo_car_classifier_model.h5')

class_names = {
    0: 'hyundai',
    1: 'lexus',
    2: 'mazda',
    3: 'mercedes',
    4: 'opel',
    5: 'skoda',
    6: 'toyota',
    7: 'volkswagen',
}

def predict_car_brand(img_path):
    try:
        img = image.load_img(img_path, target_size=(300, 300))
    except FileNotFoundError:
        print(f"Ошибка: файл '{img_path}' не найден.")
        sys.exit(1)

    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность батча
    img_array = img_array / 255.0  # Нормализация

    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Визуализация
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Марка автомобиля: {class_names[predicted_class]}\nУверенность: {confidence:.2f}")
    plt.show()

    return class_names[predicted_class], confidence

def main():
    parser = argparse.ArgumentParser(description="Классификация марки автомобиля по изображению")
    parser.add_argument('image_path', type=str, help='Путь к изображению автомобиля')
    args = parser.parse_args()

    brand, confidence = predict_car_brand(args.image_path)
    print(f"Марка автомобиля: {brand}")
    print(f"Уверенность: {confidence:.2%}")

if __name__ == "__main__":
    main()

