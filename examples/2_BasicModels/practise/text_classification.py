
import tensorflow as tf
from tensorflow import keras

import numpy as np

print(tf.__version__)

imdb = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = imdb.load_data(num_words=10000)

print("Training entries: {}, labels: {}".format(len(train_data), len(train_labels)))

# Check first review data
print train_data[0]

"""Review Decoder"""

# A dictionary mapping words to an integer index
word_index = imdb.get_word_index()

word_index = {k: (v+3) for k, v in word_index.items()}
word_index["<PAD>"] = 0
word_index["START"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

# print("Before padding:")
# print len(train_data[0]), len(train_data[1])
# print(train_data[0])

def decode_review(text):
    return ' '.join([reverse_word_index.get(i, '?') for i in text])


print(decode_review(train_data[0]))

# Pad arrays so reviews have same length, createan intengor tensor
# num_examples * max_length

"""Padding"""
train_data = keras.preprocessing.sequence.pad_sequences(train_data,
                                                        value=word_index['<PAD>'],
                                                        padding='post',
                                                        maxlen=256)

test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index['<PAD>'], padding='post', maxlen=256)

# print("After padding:")
# print len(train_data[0]), len(train_data[1])
# print(train_data[0]) # Tokenized output

"""Building the model"""

# Define vocabulary size
vocab_size = 10000

# Create sequential model
model = keras.Sequential()
model.add(keras.layers.Embedding(vocab_size, 16)) # Word embedding /  16 hidden units
model.add(keras.layers.GlobalAveragePooling1D()) # Flatten before using dense
model.add(keras.layers.Dense(16, activation=tf.nn.relu)) # 16 Hidden units
model.add(keras.layers.Dense(1, activation=tf.nn.sigmoid)) # sigmoid, because you want an output from 0 to 1 i.e. probability

model.summary()

"""Define loss function and optimizer"""

# To start training, always find loss function and optimize it

model.compile(optimizer=tf.train.AdamOptimizer(),
              loss='binary_crossentropy',
              metrics=['accuracy'])

"""Create validation set"""

x_val = train_data[:10000]
partial_x_train = train_data[10000:]

y_val = train_labels[:10000]
partial_y_train = train_labels[10000:]

"""Training the model"""

history = model.fit(partial_x_train,
                    partial_y_train,
                    epochs=40,
                    batch_size=512,
                    validation_data=(x_val, y_val),
                    verbose=1)

results = model.evaluate(test_data, test_labels)

print(results)