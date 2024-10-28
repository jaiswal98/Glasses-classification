import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential
from keras.layers import Dense, Conv2D, Flatten, Input, Dropout, MaxPooling2D, BatchNormalization
from tensorflow.keras import datasets, layers, models
from tensorflow.math import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import EfficientNetB4, Xception, ResNet50
from tensorflow.keras.layers import GlobalAveragePooling2D, concatenate
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import load_model
from keras.preprocessing.image import ImageDataGenerator
from sklearn.utils import class_weight  # Import class_weight
import random
import os
import cv2
import warnings
warnings.filterwarnings('ignore')

train_dir = r"C:\Rahul\AI\Dataset\archive\non\train"
val_dir = r"C:\Rahul\AI\Dataset\archive\non\validate"

glasses_dir = os.path.join(train_dir, 'glasses')
no_glasses_dir = os.path.join(train_dir, 'noglasses')
val_glasses_dir = os.path.join(val_dir, 'glasses')
val_no_glasses_dir = os.path.join(val_dir, 'noglasses')

glasses_images = [os.path.join(glasses_dir, img) for img in os.listdir(glasses_dir)]
no_glasses_images = [os.path.join(no_glasses_dir, img) for img in os.listdir(no_glasses_dir)]
val_glasses_images = [os.path.join(val_glasses_dir, img) for img in os.listdir(val_glasses_dir)]
val_no_glasses_images = [os.path.join(val_no_glasses_dir, img) for img in os.listdir(val_no_glasses_dir)]

# DataFrames for glasses and no glasses
glasses_df = pd.DataFrame({'image_path': glasses_images, 'label': "Yes"})
no_glasses_df = pd.DataFrame({'image_path': no_glasses_images, 'label': "No"})
val_glasses_df = pd.DataFrame({'image_path': val_glasses_images, 'label': "Yes"})
val_no_glasses_df = pd.DataFrame({'image_path': val_no_glasses_images, 'label': "No"})

# Concatenate both DataFrames
train = pd.concat([glasses_df, no_glasses_df], ignore_index=True)
val = pd.concat([val_glasses_df, val_no_glasses_df], ignore_index=True)

# Shuffle the DataFrame
train = train.sample(frac=1).reset_index(drop=True)
val = val.sample(frac=1).reset_index(drop=True)

# Calculate class weights
class_labels = train['label'].values
class_weights = class_weight.compute_class_weight('balanced', classes=np.unique(class_labels), y=class_labels)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}  # Map classes to weights

# Plot Some Samples
plt.figure(figsize=(15, 10))
for i in range(20):
    plt.subplot(4, 5, i + 1)
    img_path = train.loc[i, "image_path"]
    img = cv2.imread(img_path)  
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis('off')
plt.tight_layout()
plt.show()

# Data Augmentation
train_datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.3,
    zoom_range=0.3,
    horizontal_flip=True,
    fill_mode='nearest',
    rescale=1.0/255.0
)

augmented_images = train_datagen.flow_from_dataframe(
    train,
    x_col='image_path',
    y_col='label',
    target_size=(320, 320),
    batch_size=16,
    class_mode='binary'
)

val_datagen = ImageDataGenerator(rescale=1.0/255.0)
val_generator = val_datagen.flow_from_dataframe(
    val,
    x_col='image_path',
    y_col='label',
    target_size=(320, 320),
    batch_size=16,
    class_mode='binary'
)

plt.figure(figsize=(15, 10))
for i in range(20):
    img, label = augmented_images.next()
    plt.subplot(4, 5, i + 1)
    plt.imshow(img[0])
    plt.axis('off')
plt.tight_layout()
plt.show()

# CNN Model
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(320, 320, 3)),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(64, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(128, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Conv2D(256, (3, 3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D(2, 2),
    Flatten(),
    Dense(256, activation='relu'),
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='binary_crossentropy', metrics=['accuracy'])

checkpoint = ModelCheckpoint('best_model.h5', monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
early_stopping = EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True)
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, min_lr=1e-7, verbose=1)

# Train the model with class weights
history = model.fit(
    augmented_images,
    validation_data=val_generator,
    epochs=100,
    batch_size=16,
    class_weight=class_weight_dict,  # Add class weights here
    callbacks=[early_stopping, checkpoint, reduce_lr]
)

# Loss-Accuracy Graph
plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()

plt.tight_layout()
plt.show()

model.save(r'C:\Rahul\AI\Models\glasses_classifier.h5')
