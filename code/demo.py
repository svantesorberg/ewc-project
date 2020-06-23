from model import EWC_Network
import tensorflow as tf
import numpy as np

from helpers import plot_random_images

EPOCHS = 20
BATCH_SIZE = 64

DATASETS = 3
SEEDS = [42, 1337, 69, 420, 69420, 42069, 7331, 96, 42069420, 6969]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
input_shape = (28, 28, 1)
vec_len = 28*28

#x_train90 = np.rot90(x_train, axes=(1,2), k=1)
#x_test90 = np.rot90(x_test, axes=(1,2), k=1)


# Normalize
#x_train_original = x_train.reshape(x_train.shape[0], vec_len) / 255.0
#x_test_original = x_test.reshape(x_test.shape[0], vec_len) / 255.0
x_train_original = x_train / 255.0
x_test_original = x_test / 255.0

datasets = [(x_train_original, x_test_original)]

# Create n - 1 permutations
for i in range(DATASETS - 1):
    perm = np.random.RandomState(SEEDS[i]).permutation(vec_len)

    temp_train = x_train_original.reshape(x_train_original.shape[0], vec_len)
    temp_test = x_test_original.reshape(x_test_original.shape[0], vec_len)
    datasets.append(
        (
            np.array([x[perm].reshape(input_shape) for x in temp_train]),
            np.array([x[perm].reshape(input_shape) for x in temp_test])
        )
    )

#plot_random_images(datasets[0][0].reshape((60000, 28, 28)))

n_classes = len(np.unique(np.append(y_train, y_test)))

net = EWC_Network(EPOCHS, BATCH_SIZE, x_train_original.shape[1:], len(np.unique(np.append(y_train, y_test))))

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

# (fashion_train, fashion_train_label), (fashion_test, fashion_test_label) = tf.keras.datasets.fashion_mnist.load_data()

# fashion_test = fashion_test.reshape(fashion_test.shape[0], 28, 28, 1) / 255.0
# fashion_train = fashion_train.reshape(fashion_train.shape[0], 28, 28, 1) / 255.0
# fashion_train_label = tf.keras.utils.to_categorical(fashion_train_label, n_classes)
# fashion_test_label = tf.keras.utils.to_categorical(fashion_test_label, n_classes)



# net.add_task(
#     fashion_train,
#     fashion_train_label,
#     fashion_test,
#     fashion_test_label,
#     name = 'Fashion MNIST'
# )
# net._train_model([0])
# net.evaluate()

net.add_task(
        x_train_original, 
        y_train, 
        x_test_original, 
        y_test, 
        name='MNIST'
)
net._train_model([0])
net.evaluate()

for i, dataset in enumerate(datasets[1:]):
    net.add_task(
        dataset[0], 
        y_train, 
        dataset[1], 
        y_test, 
        name='Permuted MNIST ' + str(i+1)
    )

for i in range(1, DATASETS):
    net._train_model([i])
    net.evaluate()

#net.evaluate()