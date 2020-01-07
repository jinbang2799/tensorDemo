import tensorflow as tf
import time

print(time.localtime(time.time()))

tf.config.gpu.set_per_process_memory_fraction(0.95)
tf.config.gpu.set_per_process_memory_growth(True)
from tensorflow import keras

nb_classes = 10

(x_train, y_train), (x_test, y_test) = keras.datasets.cifar10.load_data()

y_train = y_train.reshape(y_train.shape[0])
y_test = y_test.reshape(y_test.shape[0])
x_train = x_train.astype('float32')
X_test = x_train.astype('float32')
x_train /= 255
X_test /= 255
y_train = to_categorical(y_train, nb_classes)
y_test = to_categorical(y_test, nb_classes)
