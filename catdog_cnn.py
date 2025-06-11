from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt
import numpy as np

# 路徑請依你實際電腦資料夾調整
train_dir = r'C:\Users\s8104\Desktop\Univerity_of_Taipei\3rd_second_semester\ML\data_split\train'
val_dir = r'C:\Users\s8104\Desktop\Univerity_of_Taipei\3rd_second_semester\ML\data_split\val'

# 1. 讀圖資料
train_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    train_dir, target_size=(128, 128), batch_size=32, class_mode='binary'
)
val_gen = ImageDataGenerator(rescale=1./255).flow_from_directory(
    val_dir, target_size=(128, 128), batch_size=32, class_mode='binary'
)
# 2. 建立CNN
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(128,128,3)),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(1, activation='sigmoid')
])

# 3. 編譯與訓練
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
history = model.fit(train_gen, epochs=15, validation_data=val_gen)

# 4. 畫訓練/驗證曲線
plt.figure(figsize=(10,4))
plt.plot(history.history['accuracy'], label='train')
plt.plot(history.history['val_accuracy'], label='val')
plt.legend()
plt.title('Accuracy')
plt.show()
plt.subplot(1,2,2)
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='val')
plt.title('Loss Curve')
plt.legend()
plt.show()

val_gen.reset()
x, y = next(val_gen)
y_pred = model.predict(x)
y_pred_cls = (y_pred > 0.5).astype(int).flatten()

plt.figure(figsize=(12,6))
for i in range(10):
    plt.subplot(2,5,i+1)
    plt.imshow(x[i])
    color = "green" if y_pred_cls[i] == y[i] else "red"
    plt.title(f"Pred:{int(y_pred_cls[i])} / True:{int(y[i])}", color=color)
    plt.axis('off')
plt.tight_layout()
plt.suptitle('Predicted Results (Green=Correct, Red=Wrong)', fontsize=14, y=1.08)
plt.show()