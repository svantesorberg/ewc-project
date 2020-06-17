from model import EWC_Network
import tensorflow as tf
import numpy as np

EPOCHS = 1

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

vec_len = 28*28

x_train90 = np.rot90(x_train, axes=(1,2), k=1)
x_test90 = np.rot90(x_test, axes=(1,2), k=1)

x_train90 = x_train90.reshape(x_train90.shape[0], vec_len) / 255.0
x_test90 = x_test90.reshape(x_test90.shape[0], vec_len) / 255.0

x_train = x_train.reshape(x_train.shape[0], vec_len) / 255.0
x_test = x_test.reshape(x_test.shape[0], vec_len) / 255.0

n_classes = len(np.unique(np.append(y_train, y_test)))

net = EWC_Network(EPOCHS, 64, x_train.shape[1:], len(np.unique(np.append(y_train, y_test))))

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)




net.add_task(x_train90, y_train, x_test90, y_test, name='Rotated MNIST')
net._train_model()
net.evaluate()
net.add_task(x_train, y_train, x_test, y_test, name='MNIST')
net._train_model(task_ids=[1])

net.evaluate()