import tkinter as tk
from tkinter import filedialog, messagebox
from PIL import Image, ImageTk
import tensorflow as tf
from tensorflow.keras.preprocessing import image
import numpy as np
import os

# Загружаем модель один раз
model = tf.keras.models.load_model('E:car_classifier_model.h5')

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

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0) / 255.0
    return img_array

def predict(img_path):
    img_array = preprocess_image(img_path)
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions[0])
    confidence = np.max(predictions[0])
    return class_names[predicted_class], confidence

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Распознавание марки авто")
        self.geometry("600x500")

        # Поле ввода пути
        self.path_entry = tk.Entry(self, width=50)
        self.path_entry.pack(pady=10)

        # Кнопка выбора файла
        btn_browse = tk.Button(self, text="Выбрать файл", command=self.browse_file)
        btn_browse.pack()

        # Кнопка распознавания
        btn_predict = tk.Button(self, text="Распознать марку авто", command=self.on_predict)
        btn_predict.pack(pady=10)

        # Метка результата
        self.result_label = tk.Label(self, text="", font=("Arial", 14))
        self.result_label.pack(pady=10)

        # Холст для картинки
        self.img_label = tk.Label(self)
        self.img_label.pack()

    def browse_file(self):
        file_path = filedialog.askopenfilename(
            filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")]
        )
        if file_path:
            self.path_entry.delete(0, tk.END)
            self.path_entry.insert(0, file_path)

    def on_predict(self):
        img_path = self.path_entry.get().strip()
        if not os.path.isfile(img_path):
            messagebox.showerror("Ошибка", "Файл не найден!")
            return

        brand, confidence = predict(img_path)
        self.result_label.config(text=f"Марка автомобиля: {brand}\nУверенность: {confidence:.2%}")

        # Показываем изображение
        img = Image.open(img_path)
        img.thumbnail((400, 300))
        img_tk = ImageTk.PhotoImage(img)
        self.img_label.config(image=img_tk)
        self.img_label.image = img_tk  # нужно держать ссылку

if __name__ == "__main__":
    app = App()
    app.mainloop()