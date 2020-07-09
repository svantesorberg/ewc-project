from model import EWC_Network
from permute_mnist import get_permuted_dataset, get_shuffled_dataset
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

EPOCHS = 30
BATCH_SIZE = 64
EWC_LAMBDA = 200

DATASETS = 5
SEEDS = [42, 1337, 69, 420, 69420, 42069, 7331, 96, 42069420, 6969]

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)

# Normalize
x_train_original = x_train / 255.0
x_test_original = x_test / 255.0

input_shape = x_train_original.shape[1:]
n_classes = len(np.unique(np.append(y_train, y_test)))

datasets = [(x_train_original, x_test_original)]

# Create n - 1 permutations
for i in range(DATASETS - 1):
    datasets.append(
        get_permuted_dataset(x_train_original, x_test_original)#, seed=SEEDS[i])
        #get_shuffled_dataset(x_train_original, x_test_original, square_side=7)#, seed=SEEDS[i])
    )

n_classes = len(np.unique(np.append(y_train, y_test)))

model = tf.keras.models.Sequential([
    tf.keras.layers.Input(shape=input_shape, name='input'),
    tf.keras.layers.Dropout(rate=0.2, name='dropout1'),
    tf.keras.layers.Conv2D(
        64, 
        (3, 3),
        padding='same',
        activation='relu',
        name='conv1-1'
    ),
    tf.keras.layers.MaxPooling2D(
        pool_size=(2,2),
        name='pool1'
    ),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(
        1024, 
        activation='relu',
        name='fc1'
    ),
    tf.keras.layers.Dropout(rate=0.5, name='dropout2'),
    tf.keras.layers.Dense(
        1024, 
        activation='relu', 
        name='fc2'
    ),
    tf.keras.layers.Dropout(rate=0.5, name='dropout3'),
    tf.keras.layers.Dense(
        n_classes, 
        activation='softmax',
        name='output'
    )
])


net = EWC_Network(  model, 
                    EPOCHS, 
                    BATCH_SIZE, 
                    EWC_LAMBDA,
                    learning_rate=1e-4,
                    num_tasks_to_remember=-1 # Remember everything
)

y_train = tf.keras.utils.to_categorical(y_train, n_classes)
y_test = tf.keras.utils.to_categorical(y_test, n_classes)

net.add_task(
        x_train_original, 
        y_train, 
        x_test_original, 
        y_test, 
        name='MNIST'
)

for i, dataset in enumerate(datasets[1:]):
    net.add_task(
        dataset[0], 
        y_train, 
        dataset[1], 
        y_test, 
        name='Permuted MNIST ' + str(i+1)
    )


PLOT = False

if PLOT:
    net.train_model(record_history=True)
    history_list = net.get_history()

    for i in range(DATASETS):
        task_x = [j for j in range(EPOCHS*i + 1, EPOCHS*DATASETS + 1)]
        task_y = history_list[i]['history']['accuracy']
        plt.ylim((0.0, 1.0))
        plt.plot(task_x, task_y, label = ('Task ' + str(i + 1)))
        if i > 0:
            plt.axvline(EPOCHS*i, 0, 1, 
                        color='grey', 
                        linestyle='dotted'
            )
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
else:
    net.train_model(record_history=False)
    net.evaluate()




