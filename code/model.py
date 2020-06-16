import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as kb

from helpers import get_name_or_id

class EWC_Network():
    
    def __init__(
        self, 
        n_epochs, 
        batch_size, 
        input_shape, 
        n_classes, 
        learning_rate = 1e-3
    ):
        self.tasks = []                         # Info/data for the tasks
        self.n_epochs = n_epochs                # Epochs used in training
        self.batch_size = batch_size            # Batch size used in training
        self.learning_rate = learning_rate      # LR used in training
        self.input_shape = input_shape          # Data input shape
        self.n_classes = n_classes              # Number of output classes

        self._build_model()

    # TODO    
    # This function will compute the ewc regularization terms for each task
    # and add them together. Keras then combines this with the network's 
    # loss function.
    def ewc_regularizer(self, weights):
        return 0 * tf.reduce_sum(tf.square(weights)) # Just a placeholder

    # Build a fully connected network with two hidden layers
    # Currently static width = 400
    def _build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.input_shape, name='input'),
            tf.keras.layers.Dense(
                400, 
                activation='relu',
                bias_regularizer=self.ewc_regularizer,
                kernel_regularizer=self.ewc_regularizer,
                name='fc1'
            ),
            tf.keras.layers.Dense(
                400, 
                activation='relu', 
                bias_regularizer=self.ewc_regularizer,
                kernel_regularizer=self.ewc_regularizer, 
                name='fc2'
            ),
            tf.keras.layers.Dense(
                self.n_classes, 
                activation='softmax', 
                name='output')
        ])

        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        print('Model built. Here is a summary:',end='\n\n')
        self.model.summary(print_fn=lambda s: print('\t' + s))
        print()

    # Train models on the tasks specified in tasks vector
    # in the order that they are specified in task_ids)
    def _train_model(self, task_ids = None):
        if not task_ids:
            task_ids = [i for i in range(len(self.tasks))]
             
        for task_id in task_ids:
            task = self.tasks[task_id]
            # Safety check - ensure nobody has messed around with task ordering
            assert(task_id == task['meta']['id']) 

            print('Training on task', get_name_or_id(task))

            self.model.fit(
                task['X_train'], 
                task['Y_train'],
                epochs = self.n_epochs,
                batch_size=self.batch_size
            )

            task['trained_weights'] = self.model.get_weights()
        

    def add_task(
        self, 
        X_train, Y_train, X_test, Y_test, 
        n_classes = None, 
        name = ''
    ):
        # If the number of classes is not specified, try to infer it
        if n_classes is None:
            n_classes = len(np.unique(np.append(Y_train, Y_test)))

        # Record the information about this particular task
        self.tasks.append({
            'X_train': X_train,         # Training input
            'Y_train': Y_train,         # Training output
            'X_test': X_test,           # Testing input
            'Y_test': Y_test,           # Testing output
            'n_classes': n_classes,     # Number of classes in data

            'meta': {                   # Info unrelated to network
                'id': len(self.tasks),  # Id of the task
                'name': name            # Human-readable task name
            }
        })


        print(
            'Successfully added task', get_name_or_id(self.tasks[-1])
        )


    # Simple evaluation of performance on specified tasks
    def evaluate(self, task_ids = None):
        if not task_ids:
            task_ids = [i for i in range(len(self.tasks))]
        
        for task_id in task_ids:
            task=self.tasks[task_id]
            assert(task_id == task['meta']['id'])
            loss, accuracy = self.model.evaluate(task['X_test'], task['Y_test'])
            print(
                'Task', get_name_or_id(task), 
                'test accuracy:', accuracy, 
                'loss:', loss)