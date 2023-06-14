import pandas as pd
import numpy as np
import os
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical

# Load the image dataset (assuming it's already preprocessed)
image_data_dir = "/content/drive/MyDrive/data/train_images"
image_filenames = os.listdir(image_data_dir)
image_data = []
for filename in image_filenames:
    image_path = os.path.join(image_data_dir, filename)
    image = cv2.imread(image_path)
    image_data.append(image)
image_data = np.array(image_data)

# Load the labels CSV file
labels_data = pd.read_csv("/content/drive/MyDrive/data/labels.csv")

# Merge the image dataset with the labels based on a common key, such as the image filename
combined_data = pd.merge(labels_data, pd.DataFrame({"image_filename": image_filenames}), on="image_filename")

# Prepare the combined dataset for training
X = image_data
y = to_categorical(combined_data["label"])

# Split the combined dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Data augmentation
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=True
)

datagen.fit(X_train)

# Design the CNN model
model = Sequential()
model.add(Conv2D(128, kernel_size=(3, 3), activation="relu", input_shape=(image_data.shape[1:])))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(32, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten())
model.add(Dense(64, activation="relu"))
model.add(Dense(2, activation="softmax"))

# Compile the model
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])

# Train the model on the augmented training dataset
model.fit(datagen.flow(X_train, y_train, batch_size=16), epochs=100, validation_data=(X_test, y_test))

# Evaluate the model's performance on the testing dataset
score = model.evaluate(X_test, y_test, verbose=0)
print("Test loss:", score[0])
print("Test accuracy:", score[1])
