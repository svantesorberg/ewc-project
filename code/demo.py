from model import EWC_Network
import tensorflow as tf
import numpy as np

EPOCHS = 3

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

vec_len = 28*28

#x_train90 = np.rot90(x_train, axes=(1,2), k=1)
#x_test90 = np.rot90(x_test, axes=(1,2), k=1)


# Normalize
x_train_original = x_train.reshape(x_train.shape[0], vec_len) / 255.0
x_test_original = x_test.reshape(x_test.shape[0], vec_len) / 255.0

# Create 2 permutations
seed1, seed2 = 42, 1337
permutation1 = np.random.RandomState(seed1).permutation(vec_len)
permutation2 = np.random.RandomState(seed2).permutation(vec_len)



x_train_permutation1 = np.array([x[permutation1] for x in x_train_original])
x_test_permutation1 = np.array([x[permutation1] for x in x_test_original])


x_train_permutation2 = np.array([x[permutation2] for x in x_train_original])
x_test_permutation2 = np.array([x[permutation2] for x in x_test_original])

print(x_train_original.shape)
print(x_train_permutation1.shape)
print(x_train_permutation2.shape)

#x_train90 = x_train90.reshape(x_train90.shape[0], vec_len) / 255.0
#x_test90 = x_test90.reshape(x_test90.shape[0], vec_len) / 255.0

n_classes = len(np.unique(np.append(y_train, y_test)))

net = EWC_Network(EPOCHS, 128, x_train_original.shape[1:], len(np.unique(np.append(y_train, y_test))))

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)



net.add_task(
    x_train_original, 
    y_train, 
    x_test_original, 
    y_test, 
    name='MNIST'
)
net.add_task(
    x_train_permutation1, 
    y_train, 
    x_test_permutation1, 
    y_test, 
    name='Permuted MNIST 1'
)
net.add_task(
    x_train_permutation2, 
    y_train, 
    x_test_permutation2, 
    y_test, 
    name='Permuted MNIST 2'
)

net._train_model([0])
net.evaluate()
net._train_model([1])
net.evaluate()
net._train_model([2])
net.evaluate()