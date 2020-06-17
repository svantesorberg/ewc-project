from model import EWC_Network
import tensorflow as tf
import numpy as np

EPOCHS = 20
DATASETS = 6
SEEDS = [42, 1337, 69, 420, 69420, 42069, 7331, 96, 42069420, 6969]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

vec_len = 28*28

#x_train90 = np.rot90(x_train, axes=(1,2), k=1)
#x_test90 = np.rot90(x_test, axes=(1,2), k=1)


# Normalize
x_train_original = x_train.reshape(x_train.shape[0], vec_len) / 255.0
x_test_original = x_test.reshape(x_test.shape[0], vec_len) / 255.0

datasets = [(x_train_original, x_test_original)]

# Create n - 1 permutations
for i in range(DATASETS - 1):
    perm = np.random.RandomState(SEEDS[i]).permutation(vec_len)
    datasets.append(
        (
            np.array([x[perm] for x in x_train_original]),
            np.array([x[perm] for x in x_test_original])
        )
    )

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
    #net.evaluate()

net.evaluate()