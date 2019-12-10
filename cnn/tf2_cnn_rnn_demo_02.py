import tensorflow as tf
import numpy as np
from sklearn.metrics import accuracy_score
import time

print(time.localtime(time.time()))

# tf.config.gpu.set_per_process_memory_fraction = 0.95
from tensorflow import keras

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = keras.utils.to_categorical(y_train)
y_test = keras.utils.to_categorical(y_test)

print(x_train.shape, ' ', y_train.shape)
print(x_test.shape, ' ', y_test.shape)

from tensorflow.keras import layers

model = keras.Sequential()

x_shape = x_train.shape
model.add(layers.Conv2D(input_shape=(x_shape[1], x_shape[2], x_shape[3]),
                        filters=32, kernel_size=(3, 3), strides=(1, 1),
                        padding='same', activation='relu'))
model.add(layers.MaxPool2D(pool_size=(2, 2)))

print(model.output_shape)

model.add(layers.Reshape(target_shape=(16 * 16, 32)))
model.add(layers.LSTM(50, return_sequences=False))

model.add(layers.Dense(10, activation='softmax'))

model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

# % % time
history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_split=0.1)
model.save_weights('tf2_cnn_rnn_demo_02_1.h5')

import matplotlib.pyplot as plt

plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

res = model.evaluate(x_test, y_test)

x_shape = x_train.shape
inn = layers.Input(shape=(x_shape[1], x_shape[2], x_shape[3]))
conv = layers.Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1),
                     padding='same', activation='relu')(inn)
pool = layers.MaxPool2D(pool_size=(2, 2), padding='same')(conv)
flat = layers.Flatten()(pool)
dense1 = layers.Dense(64)(flat)

reshape = layers.Reshape(target_shape=(x_shape[1] * x_shape[2], x_shape[3]))(inn)
lstm_layer = layers.LSTM(32, return_sequences=False)(reshape)
dense2 = layers.Dense(64)(lstm_layer)

merged_layer = layers.concatenate([dense1, dense2])
outt = layers.Dense(10, activation='softmax')(merged_layer)
model = keras.Model(inputs=inn, outputs=outt)
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

merged_layer = layers.concatenate([dense1, dense2])
outt = layers.Dense(10, activation='softmax')(merged_layer)
model = keras.Model(inputs=inn, outputs=outt)
model.compile(optimizer=keras.optimizers.Adam(),
              loss=keras.losses.CategoricalCrossentropy(),
              metrics=['accuracy'])
model.summary()

# % % time
history2 = model.fit(x_train, y_train, batch_size=64, epochs=5, validation_split=0.1)
model.save_weights('tf2_cnn_rnn_demo_02.h5')


plt.plot(history2.history['accuracy'])
plt.plot(history2.history['val_accuracy'])
plt.legend(['training', 'valivation'], loc='upper left')
plt.show()

print(time.localtime(time.time()))
