import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Загрузка предварительно обученной модели ResNet50
base_model = tf.keras.applications.ResNet50(weights='imagenet', include_top=False)

# Замораживаем веса базовой модели, чтобы они не обновлялись в процессе обучения
base_model.trainable = False

# Создаем сеть детекции
def create_detection_model(base_model, num_classes=4):
    # Входной тензор
    input_tensor = tf.keras.Input(shape=(256, 256, 3))
    
    # Слой для масштабирования значений пикселей в диапазон [0, 1]
    x = layers.experimental.preprocessing.Rescaling(1./255)(input_tensor)
    
    # Подаем входной тензор в базовую модель ResNet без верхних слоев
    x = base_model(x, training=False)
    
    # Добавляем слои для детекции
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation='relu')(x)  # Произвольное количество нейронов
    class_output = layers.Dense(num_classes, activation='softmax', name='class_output')(x)  # Классификация фигур
    bbox_output = layers.Dense(4, name='bbox_output')(x)  # Регрессия для ограничивающих рамок
    
    # Создаем экземпляр модели с несколькими выходами
    model = models.Model(inputs=input_tensor, outputs=[class_output, bbox_output])
    
    return model

# Создаем модель
model = create_detection_model(base_model)

# Компилируем модель с выбранной функцией потерь и оптимизатором
model.compile(optimizer='adam',
              loss={'class_output': 'categorical_crossentropy', 'bbox_output': 'mean_squared_error'},
              metrics={'class_output': 'accuracy'})

# Выводим информацию о модели
model.summary()

# Создайте генераторы данных для обучения и валидации
train_datagen = ImageDataGenerator(rescale=1./255)  # Масштабирование значений пикселей для обучающего набора
val_datagen = ImageDataGenerator(rescale=1./255)    # Масштабирование значений пикселей для валидационного набора
test_datagen = ImageDataGenerator(rescale=1./255)  

# Укажите пути к каталогам с обучающими и валидационными данными
train_data_dir = 'generated_images/train'
val_data_dir = 'generated_images/val'
test_data_dir = 'generated_images/test'

# Создайте генераторы данных
batch_size = 32
train_generator = train_datagen.flow_from_directory(directory=train_data_dir,
                                                    target_size=(256, 256),
                                                    color_mode='rgb',
                                                    shuffle=False,
                                                    batch_size=32)
val_generator = val_datagen.flow_from_directory(directory=val_data_dir,
                                                    target_size=(256, 256),
                                                    color_mode='rgb',
                                                    shuffle=False,
                                                    batch_size=32)
test_generator = test_datagen.flow_from_directory(directory=test_data_dir,
                                                    target_size=(256, 256),
                                                    color_mode='rgb',
                                                    shuffle=False,
                                                    batch_size=32)

# Обучаем модель
history = model.fit(train_generator, epochs=3, validation_data=val_generator)

# Оцениваем модель на тестовых данных
eval_result = model.evaluate(test_generator)

print("Loss:", eval_result[0])
print("Accuracy:", eval_result[1])


import matplotlib.pyplot as plt

# Получите данные о потерях и точности из объекта history
train_loss = history.history['loss']
val_loss = history.history['val_loss']
train_accuracy = history.history['class_output_accuracy']
val_accuracy = history.history['val_class_output_accuracy']

# Постройте графики потерь
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(train_loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.legend()
plt.title('Loss vs. Epochs')

# Постройте графики точности
plt.subplot(1, 2, 2)
plt.plot(train_accuracy, label='Training Accuracy')
plt.plot(val_accuracy, label='Validation Accuracy')
plt.legend()
plt.title('Accuracy vs. Epochs')

plt.show()
