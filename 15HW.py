# -*- coding: utf-8 -*-
"""
Классификация рукописных английских букв A-Z с использованием полносвязной нейронной сети.
Требуется точность >97% на тестовой выборке.
"""

# 1. Импорт необходимых библиотек
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.keras import layers, models, utils

# 2. Загрузка данных из CSV-файла
# Предполагаем, что файл находится в той же директории
data_path = r'C:\Users\User\Desktop\15ht\A_Z_Handwritten_Data.csv'  # Укажите путь к файлу
# Чтение данных: первый столбец - метка (0-25), остальные 784 - пиксели 28x28
data = pd.read_csv(data_path, header=None)

# 3. Разделение на признаки (X) и метки (y)
labels = data.iloc[:, 0].values  # Первый столбец - метки
images = data.iloc[:, 1:].values  # Остальные 784 столбца - пиксели

# 4. Нормализация данных: приведение пикселей к диапазону [0, 1]
images = images.astype('float32') / 255.0

# 5. Разделение на обучающую и тестовую выборки (80%/20%)
X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size=0.2, random_state=42)

# 6. Преобразование меток в one-hot encoding (26 классов)
num_classes = 26
y_train = utils.to_categorical(y_train, num_classes)
y_test = utils.to_categorical(y_test, num_classes)

# 7. Построение модели (полносвязная нейронная сеть)
def build_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape, name='input'),
        layers.Dense(512, activation='relu', name='hidden1'),
        layers.Dense(256, activation='relu', name='hidden2'),
        layers.Dense(128, activation='relu', name='hidden3'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    return model

# Создание модели
input_shape = (X_train.shape[1],)  # 784 признака (28x28 пикселей)
model = build_model(input_shape, num_classes)

# 8. Компиляция модели
model.compile(
    optimizer='adam',  # Используем Adam для лучшей сходимости
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# 9. Обучение модели
history = model.fit(
    X_train, y_train,
    validation_split=0.2,  # 20% обучающих данных для валидации
    epochs=20,
    batch_size=128,
    verbose=1
)

# 10. Оценка модели на тестовой выборке
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f'Точность на тестовой выборке: {test_acc:.4f}')
print(f'Потери на тестовой выборке: {test_loss:.4f}')

# 11. Визуализация графиков точности и потерь
def plot_training_history(history):
    # График точности
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Обучающая выборка')
    plt.plot(history.history['val_accuracy'], label='Валидационная выборка')
    plt.title('График точности')
    plt.xlabel('Эпоха')
    plt.ylabel('Точность')
    plt.legend()

    # График потерь
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Обучающая выборка')
    plt.plot(history.history['val_loss'], label='Валидационная выборка')
    plt.title('График потерь')
    plt.xlabel('Эпоха')
    plt.ylabel('Потери')
    plt.legend()
    plt.show()

plot_training_history(history)

# 12. Подбор гиперпараметров (пример)
# Если точность недостаточна, можно изменить архитектуру или параметры обучения.
# Пример альтернативной модели с большим числом нейронов и dropout для регуляризации
def build_improved_model(input_shape, num_classes):
    model = models.Sequential([
        layers.Input(shape=input_shape, name='input'),
        layers.Dense(1024, activation='relu', name='hidden1'),
        layers.Dropout(0.3),
        layers.Dense(512, activation='relu', name='hidden2'),
        layers.Dropout(0.3),
        layers.Dense(256, activation='relu', name='hidden3'),
        layers.Dense(num_classes, activation='softmax', name='output')
    ])
    return model

# Построение и обучение улучшенной модели (если требуется)
improved_model = build_improved_model(input_shape, num_classes)
improved_model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

improved_history = improved_model.fit(
    X_train, y_train,
    validation_split=0.2,
    epochs=30,  # Увеличиваем число эпох
    batch_size=128,
    verbose=1
)

# Оценка улучшенной модели
improved_test_loss, improved_test_acc = improved_model.evaluate(X_test, y_test, verbose=0)
print(f'Точность улучшенной модели на тестовой выборке: {improved_test_acc:.4f}')
print(f'Потери улучшенной модели на тестовой выборке: {improved_test_loss:.4f}')

# Визуализация для улучшенной модели
plot_training_history(improved_history)

# Выводы по результатам
if improved_test_acc > 0.97:
    print("Цель достигнута: точность >97% на тестовой выборке.")
else:
    print("Требуется дальнейшая настройка гиперпараметров или архитектуры.")