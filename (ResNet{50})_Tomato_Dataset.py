#!/usr/bin/env python
# coding: utf-8

# In[1]:


import os
import shutil
import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

# Define classes
classes = ["Tomato_Bacterial_spot", "Tomato_Early_blight", "Tomato_Late_blight","Tomato_Leaf_Mold","Tomato_Septoria_leaf_spot","Tomato_Spider_mites_Two-spotted_spider_mite","Tomato_Target_Spot","Tomato_Tomato_Yellow_Leaf_Curl_Virus","Tomato_Tomato_mosaic_virus","Tomato_healthy"]

# Define main directory
main_dir = "E:/IITR/Daksh/Datasets/Tomato_Dataset"

# Initialize ImageDataGenerator for data augmentation for train data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rotation_range=15,
    width_shift_range=0.1,
    height_shift_range=0.1,
    horizontal_flip=True,
    preprocessing_function=preprocess_input
)

# Load and augment the training data
train_generator = train_datagen.flow_from_directory(
    main_dir,
    target_size=(224, 224),  # ResNet50 input size
    batch_size=64,
    classes=classes,  # Specify the classes explicitly
    class_mode='sparse',  # Since we have sparse categorical labels
    shuffle=True,
)

# Initialize ImageDataGenerator without augmentation for test data
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(preprocessing_function=preprocess_input)

# Load the test data without augmentation
test_generator = test_datagen.flow_from_directory(
    main_dir,
    target_size=(224, 224),  # ResNet50 input size
    batch_size=64,
    classes=classes,  # Specify the classes explicitly
    class_mode='sparse',
    shuffle=False,  # No need to shuffle the test data
)

# Load the ResNet50 model with pretrained weights on ImageNet
resnet50 = ResNet50(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the layers in the ResNet50 model
for layer in resnet50.layers:
    layer.trainable = False

# Define your custom classification layers
x = layers.Flatten()(resnet50.output)
x = layers.Dense(128, activation='relu')(x)
output = layers.Dense(len(classes), activation='softmax')(x)

# Create the model
resnet_model = models.Model(resnet50.input, output)

# Compile the model with Adam optimizer and sparse categorical crossentropy loss
resnet_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Fit the model
history = resnet_model.fit(
    train_generator,
    epochs=70,
    validation_data=test_generator
)

# Save the Keras model
resnet_model.save("Tomato.h5")


# Evaluate the model
test_loss, test_accuracy = resnet_model.evaluate(test_generator)
print("Test Accuracy:", test_accuracy)

# Get the predicted classes for the test data
y_pred = np.argmax(resnet_model.predict(test_generator), axis=1)

# Get true labels
y_true = test_generator.classes

# Calculate confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
accuracy = np.trace(conf_matrix) / np.sum(conf_matrix)
accuracy_percentage = accuracy * 100

# Calculate precision, recall, and F1 score
classification_rep = classification_report(y_true, y_pred, target_names=classes)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, fmt="d", cmap="Blues", xticklabels=classes, yticklabels=classes)
plt.xlabel(f'Predicted labels\nAccuracy: {accuracy_percentage:.2f}%\n{classification_rep}')
plt.ylabel(f'True labels\nAccuracy: {accuracy_percentage:.2f}%')
plt.title(f'Confusion Matrix\nAccuracy: {accuracy_percentage:.2f}%')
plt.show()

# Plot training and validation accuracy
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()
plt.show()

# Plot training and validation loss
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()

# Get the predicted class name
predicted_class_name = classes[y_pred[3]]
print("Predicted class name:", predicted_class_name)

