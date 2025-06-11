import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import random

# 資料夾路徑（請自行修改成你自己的）
train_dir = r'C:\Users\s8104\Desktop\Univerity_of_Taipei\3rd_second_semester\ML\data_split\train'
val_dir = r'C:\Users\s8104\Desktop\Univerity_of_Taipei\3rd_second_semester\ML\data_split\val'

# 1. Data Augmentation
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)
val_datagen = ImageDataGenerator(rescale=1./255)

# 2. 資料生成器
train_gen = train_datagen.flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='binary'
)
val_gen = val_datagen.flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode='binary', shuffle=False
)

# 3. 進階 CNN 架構
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.BatchNormalization(),
    layers.MaxPooling2D(2,2),

    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.BatchNormalization(),
    layers.Dropout(0.5),
    layers.Dense(1, activation='sigmoid')
])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# 4. 訓練
history = model.fit(train_gen, epochs=15, validation_data=val_gen)

# 5. Accuracy & Loss 曲線
plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Val')
plt.title('Accuracy Curve')
plt.legend()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Val')
plt.title('Loss Curve')
plt.legend()
plt.tight_layout()
plt.show()

# 6. 預測圖片展示（正確=綠, 錯誤=紅）
val_gen.reset()
x_batch, y_batch = next(val_gen)
y_pred_batch = model.predict(x_batch)
y_pred_cls_batch = (y_pred_batch > 0.5).astype(int).flatten()

plt.figure(figsize=(12,5))
idxs = list(range(len(x_batch)))
random.shuffle(idxs)
for i, idx in enumerate(idxs[:10]):
    plt.subplot(2,5,i+1)
    plt.imshow(x_batch[idx])
    color = 'green' if y_pred_cls_batch[idx] == y_batch[idx] else 'red'
    plt.title(f"P:{int(y_pred_cls_batch[idx])} T:{int(y_batch[idx])}", color=color)
    plt.axis('off')
plt.suptitle('Prediction Results (Green=Correct, Red=Wrong)', fontsize=16, y=1.08)
plt.tight_layout()
plt.show()

