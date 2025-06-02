import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt

# Загрузка сохраненной модели
model = tf.keras.models.load_model('E:\\pythonProject\\car_classifier_model.h5')
# E:\\pythonProject\\pythonProject\\CAR\\
# E:\\pythonProject\\pythonProject\\LogoCar
# Словарь с названиями классов (должен соответствовать порядку при обучении)
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
    # Загрузка и подготовка изображения
    img = image.load_img(img_path, target_size=(300, 300))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)  # Добавляем размерность батча
    img_array = img_array / 255.0  # Нормализация

    # Предсказание
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # Визуализация
    plt.imshow(img)
    plt.axis('off')
    plt.title(f"Марка автомобиля: {class_names[predicted_class]}\nУверенность: {confidence:.2f}")
    plt.show()

    return class_names[predicted_class], confidence


# Пример использования
image_path = "C:\\Users\\1\\Desktop\\Снимок.PNG"  # Укажите путь к вашему изображению
brand, confidence = predict_car_brand(image_path)
print(f"Марка автомобиля: {brand}")
print(f"Уверенность: {confidence:.2%}")
