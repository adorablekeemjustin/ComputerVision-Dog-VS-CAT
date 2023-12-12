import os
from matplotlib import pyplot as plt
import numpy as np
from sklearn.metrics import roc_curve, auc
from keras.preprocessing import image
import tensorflow as tf

# Define paths to your dataset folders (dogs and cats)
train_dogs_dir = '/Users/keemadorable/Downloads/archive/PetImages/Dog'
train_cats_dir = '/Users/keemadorable/Downloads/archive/PetImages/Cat'
valid_dogs_dir = '/Users/keemadorable/Downloads/archive/validation/dogs'
valid_cats_dir = '/Users/keemadorable/Downloads/archive/validation/cats'

# Define the image dimensions and batch sizes
img_width, img_height = 200, 200
batch_size = 32

# Create ImageDataGenerators for training and validation data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    '/Users/keemadorable/Downloads/archive/PetImages',
    classes=['Dog', 'Cat'],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

validation_generator = valid_datagen.flow_from_directory(
    '/Users/keemadorable/Downloads/archive/validation',
    classes=['dogs', 'cats'],
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)

# Build the CNN model
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3, 3), activation='relu', input_shape=(img_width, img_height, 3)),
    tf.keras.layers.MaxPooling2D(2, 2),
    # Add more layers as needed
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=8,
    epochs=15,
    validation_data=validation_generator,
    validation_steps=8)

# Evaluate the model
model.evaluate(validation_generator)

# Predict on validation data for ROC curve
validation_generator.reset()
preds = model.predict(validation_generator, verbose=1)

fpr, tpr, _ = roc_curve(validation_generator.classes, preds)
roc_auc = auc(fpr, tpr)

# Plot ROC curve
plt.figure()
lw = 2
plt.plot(fpr, tpr, color='darkorange',
         lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Predict on a new image
file_path = '/Users/keemadorable/Desktop/testAI/icy3.jpg'  # Replace this with your image file path
img = image.load_img(file_path, target_size=(img_width, img_height))
x = image.img_to_array(img)
x = np.expand_dims(x, axis=0)
x /= 255.

classes = model.predict(x)
print("Predicted Probability:", classes[0][0])

if classes[0][0] < 0.5:
    print("It's a cat")
else:
    print("It's a dog")
