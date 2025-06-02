import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import matplotlib.pyplot as plt
import os

# === Загрузка модели ===
model = tf.keras.models.load_model('E:\\pythonProject\\pythonProject\\LogoCar\\car_classifier_model.h5')

# === Словарь классов ===
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

# === Пути к изображениям ===
image_paths = [
    "C:\\Users\\1\\Desktop\\images.jpg",
    "C:\\Users\\1\\Desktop\\Toyota-Logo-PNG-Clipart.png",
    "C:\\Users\\1\\Desktop\\Снимок.PNG",
    "C:\\Users\\1\\Desktop\\Снимок2.PNG",
    "C:\\Users\\1\\Desktop\\Снимок3.PNG",
    "C:\\Users\\1\\Desktop\\Снимок4.PNG",
    "C:\\Users\\1\\Desktop\\images433.jpg",
    "C:\\Users\\1\\Desktop\\Снимок6.PNG"
]

# === Размер изображений для модели ===
IMG_SIZE = (300, 300)

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=IMG_SIZE)
    img_array = image.img_to_array(img)
    img_array = img_array / 255.0  # Нормализация
    return img_array

# === Обработка и вывод ===
plt.figure(figsize=(16, 8))
for i, img_path in enumerate(image_paths):
    if not os.path.exists(img_path):
        print(f"[!] Файл не найден: {img_path}")
        continue

    img_array = preprocess_image(img_path)
    input_array = np.expand_dims(img_array, axis=0)

    predictions = model.predict(input_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])

    # === Визуализация ===
    plt.subplot(2, 4, i + 1)
    plt.imshow(image.array_to_img(img_array))
    plt.title(f"{class_names[predicted_class]}\n({confidence:.1%})")
    plt.axis('off')

plt.tight_layout()
plt.show()
