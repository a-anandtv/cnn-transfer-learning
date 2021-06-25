# %%
from posixpath import pardir
# from keras.applications.vgg16 import VGG16
# from keras.models import Model
# from keras.applications.vgg16 import preprocess_input
# from keras.preprocessing.image import load_img, img_to_array
from numpy import expand_dims
from matplotlib import pyplot
import os
import numpy as np
from tensorflow.python.data.ops.dataset_ops import AUTOTUNE

# %%
# Building dataset with a data builder class

# Flag to control rebuilds
REBUILD_DATA = False

PARENT_DIR = os.path.abspath(os.path.dirname(__file__))
DATASET_DIR = os.path.join(PARENT_DIR, 'data/archive/SkinDatasets/datasets')
TRAIN_DIR = os.path.join(DATASET_DIR, 'train')
TEST_DIR = os.path.join(DATASET_DIR, 'test')

print (PARENT_DIR)
print (DATASET_DIR)

# %%
# Setting dataset parameters
# data_count = {'wrinkled': 0, 'no_wrinkles': 0}
# labels = {'wrinkled': 0, 'no_wrinkles': 1}
img_size = (160, 160)
batch_size = 32
test_percentage = 0.15

# %%
# Import missing package

import tensorflow as tf
from tensorflow.keras.preprocessing import image_dataset_from_directory
# %%
# Pre processing and building data

training_dataset = image_dataset_from_directory(TRAIN_DIR,
                                                shuffle=True,
                                                batch_size=batch_size,
                                                image_size=img_size)

# %%
# Testing dataset

testing_dataset = image_dataset_from_directory(TEST_DIR,
                                                shuffle=True,
                                                batch_size=batch_size,
                                                image_size=img_size)
# %%
# Visualize the first 12 images from the training dataset

class_labels = training_dataset.class_names

pyplot.figure(figsize=(10, 10))
for images, labels in training_dataset.take(1):
    for i in range(12):
        ax = pyplot.subplot(4, 3, i+1)
        pyplot.imshow(images[i].numpy().astype("uint8"))
        pyplot.title(class_labels[labels[i]])
        pyplot.axis("off")

pyplot.show()
# %%
# Preparing validation dataset
VALIDATION_DIR = os.path.join(DATASET_DIR, 'validation')

validation_dataset = image_dataset_from_directory(VALIDATION_DIR,
                                                    shuffle=True,
                                                    batch_size=batch_size,
                                                    image_size=img_size)
# %%

print ("No of validation batches: %d" % tf.data.experimental.cardinality(validation_dataset))
print ("No of testing batches: %d" % tf.data.experimental.cardinality(testing_dataset))
print ("No of training batches: %d" % tf.data.experimental.cardinality(training_dataset))
# %%
# Tuning for performancs
# From the tf documentation

AUTOTUNE = tf.data.AUTOTUNE

training_dataset = training_dataset.prefetch(buffer_size=AUTOTUNE)
# testing_dataset = testing_dataset.prefetch(buffer_size=AUTOTUNE)
# validation_dataset = validation_dataset.prefetch(buffer_size=AUTOTUNE)

# %%
# Testing an augmentation layer

img_augment = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.RandomFlip('horizontal'),
    tf.keras.layers.experimental.preprocessing.RandomRotation(0.2)
])

# Test on a random image
pyplot.figure(figsize=(10, 10))
for images,labels in training_dataset.take(1):
    for i in range(12):
        if class_labels[labels[i]] == 'wrinkled':
            test_image = images[i]
            for j in range(9):
                ax = pyplot.subplot(3, 3, j+1)
                augmented_img = img_augment(tf.expand_dims(test_image, axis=0))
                pyplot.imshow(augmented_img[0] / 255)
                pyplot.axis('off')
            break

pyplot.show()
# %%
# Testing a resize and rescale layer

img_rsc = tf.keras.Sequential([
    tf.keras.layers.experimental.preprocessing.Rescaling(1./127.5, offset= -1)
])

# pick a random input
img, lbl = next(iter(training_dataset))
img = img_rsc(img)
print ("Pixel values: Min: %d   Max: %d" % (img.numpy().min(), img.numpy().max()))

# %%
# Pulling the pretrained model without the top layer
# Importing the MobileNet V2 model developed at Google

image_shape = img_size + (3, )
base_model = tf.keras.applications.MobileNetV2(input_shape=image_shape,
                                                include_top=False,
                                                weights='imagenet')

# %%
# Test output dimensions with an image

img_features = base_model(img)
print ("Feature shape: ", img_features.shape)
# %%
# Building the new model

# Pause learning for the base model
base_model.trainable = False

# Average the feature extraction layers
global_average_layer = tf.keras.layers.GlobalAveragePooling2D()
# The Dense prediction layer
prediction_layer = tf.keras.layers.Dense(1)

# Build the model
inputs = tf.keras.layers.Input(shape=(160, 160, 3))
x = img_augment(inputs)
x = img_rsc(x)
x = base_model(x, training=False)
x = global_average_layer(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = prediction_layer(x)
model = tf.keras.Model(inputs, outputs)

# %%
# Compile the model

learning_rate = 0.0001
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), 
                loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
                metrics=['accuracy'])

model.summary()
# %%
# Testing the model (without training)

loss0, accuracy0 = model.evaluate(validation_dataset)

print ("Initial loss: %0.2f" % loss0)
print ("Initial accuracy: %.4f" % accuracy0)
# %%
# Train the model

epochs = 15

history = model.fit(training_dataset,
                    epochs=epochs,
                    validation_data=testing_dataset)
# %%
# Testing again (after training)

loss, accuracy = model.evaluate(validation_dataset)

print ("Trained model loss: %0.2f" % loss)
print ("Trained model accuracy: %.4f" % accuracy)
# %%
# Save the model

model.save('saved_model/wrinkle_detection')
# %%
