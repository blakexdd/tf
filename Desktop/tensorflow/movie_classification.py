# -*- coding: utf-8 -*-
"""
Created on Wed Nov 20 22:48:51 2019

@author: Gololobov
"""

# import libraries for network learning
import tensorflow as tf
from tensorflow import keras

import tensorflow_datasets as tfds
tfds.disable_progress_bar()

import numpy as np
import matplotlib.pyplot as plt

(train_data, test_data), info = tfds.load(
        # use pre-encoded version with 8k vocablurary
        'imdb_reviews/subwords8k',
        # return train and test datasets as a tuple
        split = (tfds.Split.TRAIN, tfds.Split.TEST),
        # return (example, label) pairs from the dataset
        as_supervised=True,
        # return info structure
        with_info=True)

encoder = info.features['text'].encoder

# preparing data for training 
BUFFER_SIZE = 1000

train_batches = (
        train_data
        .shuffle(BUFFER_SIZE)
        .padded_batch(32, train_data.output_shapes))

test_batches = (
        test_data
        .padded_batch(32, train_data.output_shapes))

# building a model
model = keras.Sequential([
        keras.layers.Embedding(encoder.vocab_size, 16),
        keras.layers.GlobalAveragePooling1D(),
        keras.layers.Dense(1, activation='sigmoid')])

model.summary()

# configure model 
model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# training model
history = model.fit(train_batches, 
                    epochs=10,
                    validation_data=test_batches,
                    validation_steps=30)

# evaluate the model
loss, accuracy = model.evaluate(test_batches)

print('Loss: ', loss)
print('Accuracy: ', accuracy)

#create graph of accuracy and loss over time
history_dict = history.history

#plotting graph
acc = history_dict['accuracy']
val_acc = history_dict['val_accuracy']
loss = history_dict['loss']
val_loss = history_dict['val_loss']

epochs = range(1, len(acc) + 1)
plt.subplot(2, 1, 1)
plt.plot(epochs, loss, 'bo', label='Training loss')
plt.plot(epochs, val_loss, 'b', label='Validation loss')
plt.title('training and validation loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.subplot(2, 1, 2)
plt.plot(epochs, acc, 'bo', label='Training acc')
plt.plot(epochs, val_acc, 'b', label='validation acc')
plt.title('Training and validationg accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend(loc = 'lower right')

plt.show()











