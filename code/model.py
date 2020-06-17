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
    #
    # This class represents the regularization function in EWC, i.e.
    # the penalty term associated with the current parameters based 
    # on how much they differ from previously trained parameters for 
    # different tasks.
    class EWC_Regularizer(tf.keras.regularizers.Regularizer):
        def __init__(self, constant = 1):
            self.constant = constant
            self.trained_parameters_per_task = []
            self.fisher_diagonal_per_task = []
        
        def __call__(self, weights):
            penalty = 0
            
            for params, fisher in zip(
                self.trained_parameters_per_task, self.fisher_diagonal_per_task
            ):
                penalty += (
                    (self.constant / 2) * tf.reduce_sum(
                        tf.multiply(fisher, tf.square(weights - params))
                    )
                )

            return penalty

        # Add new parameters to the vector, used when a new task
        # has been learned
        def add_new_parameters(self, parameters):
            self.trained_parameters_per_task.append(parameters)

        def add_new_fisher_diagonal(self, fisher):
            self.fisher_diagonal_per_task.append(fisher)

        def get_constant(self):
            return self.constant
        
        def set_constant(self, constant):
            self.constant = constant
        
        

    def update_regularization_functions(self, task):
        # Updates the regularization functions for each layer,
        # adding learned parameters for the most recently learned task
        for i, layer in enumerate(self.model.layers):

            # TESTING
            layer.kernel_regularizer.set_constant(0.1)
            layer.bias_regularizer.set_constant(0.1)

            # Update weight parameters
            new_parameters = task['trained_parameters'][2*i]
            new_fisher = task['fisher_diagonal'][2*i]
            layer.kernel_regularizer.add_new_parameters(new_parameters)
            layer.kernel_regularizer.add_new_fisher_diagonal(new_fisher)

            # Update bias parameters
            new_parameters = task['trained_parameters'][2*i + 1]
            new_fisher = task['fisher_diagonal'][2*i + 1]
            layer.bias_regularizer.add_new_parameters(new_parameters)
            layer.bias_regularizer.add_new_fisher_diagonal(new_fisher)
    

    # Build a fully connected network with two hidden layers
    # Currently static width = 400
    def _build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.input_shape, name='input'),
            tf.keras.layers.Dense(
                600, 
                activation='relu',
                bias_regularizer=self.EWC_Regularizer(),
                kernel_regularizer=self.EWC_Regularizer(),
                name='fc1'
            ),
            tf.keras.layers.Dense(
                600, 
                activation='relu', 
                bias_regularizer=self.EWC_Regularizer(),
                kernel_regularizer=self.EWC_Regularizer(), 
                name='fc2'
            ),
            tf.keras.layers.Dense(
                self.n_classes, 
                activation='softmax',
                bias_regularizer=self.EWC_Regularizer(),
                kernel_regularizer=self.EWC_Regularizer(), 
                name='output')
        ])

        self._compile_model()

        print('Model built. Here is a summary:',end='\n\n')
        self.model.summary(print_fn=lambda s: print('\t' + s))
        print()

    def _compile_model(self):
        self.model.compile(
            # optimizer=tf.keras.optimizers.Adam(
            #     learning_rate=self.learning_rate
            # ),
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate
            ),
            # optimizer = tf.keras.optimizers.SGD(
            #    learning_rate=self.learning_rate
            # ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # Train models on the tasks specified in tasks vector
    # in the order that they are specified in task_ids)
    def _train_model(self, task_ids = None):
        if not task_ids:
            task_ids = [i for i in range(len(self.tasks))]
             
        for task_id in task_ids:
            task = self.tasks[task_id]
            # Safety check - ensure nobody has messed around with task ordering
            assert(task_id == task['meta']['id']) 

            print('Training on', get_name_or_id(task), end='...', flush=True)

            self.model.fit(
                task['X_train'], 
                task['Y_train'],
                epochs = self.n_epochs,
                batch_size=self.batch_size,
                verbose=False
            )

            print('\rTraining on', get_name_or_id(task), 'done!')
            
            task['trained_parameters'] = self.model.get_weights()

            # Fisher Matrix by sum
            # print('\nComputing empirical Fisher - this may take a while')

            # sums = [np.zeros(shape=(784, 800)), np.zeros(shape=(800,)), np.zeros(shape=(800, 800)), np.zeros(shape=(800,)), np.zeros(shape=(800, 10)), np.zeros(shape=(10,))]
            # for i in range(0, task['X_train'].shape[0], 10):
            #     progress = round(i / task['X_train'].shape[0] * 100)
            #     print('\r[{0}{1}] {2}%'.format('#'*int(progress/10), ' '*(10 - int(progress/10)), progress), end='')
            #     data = np.array([task['X_train'][i]])
            #     labels = np.array([task['Y_train'][i]])



            #     with tf.GradientTape() as tape:
            #         tape.watch(self.model.trainable_weights)
            #         predictions = self.model(data)
            #         loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            #     grads = tape.gradient(loss, self.model.trainable_weights)
            #     squared_gradients = list(map(tf.square, grads))

            #     for j in range(len(sums)):
            #         sums[j] = sums[j] + squared_gradients[j]

            # task['fisher_diagonal'] = sums
            # print(' Done!')
            
            # Fisher matrix by single batch
            data = task['X_train']
            labels = task['Y_train']
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_weights)
                predictions = self.model(data)
                loss = tf.keras.losses.categorical_crossentropy(labels, predictions)
            grads = tape.gradient(loss, self.model.trainable_weights)

            squared_gradients = list(map(lambda x: x**2, grads))
            task['fisher_diagonal'] = squared_gradients

            # After model has been trained, we need to recompile it
            # in order to use the updated the regularization function
            self.update_regularization_functions(task)
            self._compile_model()

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


        # print(
        #     'Successfully added', get_name_or_id(self.tasks[-1])
        # )


    # Simple evaluation of performance on specified tasks
    def evaluate(self, task_ids = None):
        if not task_ids:
            task_ids = [i for i in range(len(self.tasks))]
        
        print('\n##########################################')
        print('Evaluating all tasks\n')

        for task_id in task_ids:
            task=self.tasks[task_id]
            assert(task_id == task['meta']['id'])
            print('Evaluating', get_name_or_id(task), end='...\n')
            loss, accuracy = self.model.evaluate(
                task['X_test'], 
                task['Y_test'], 
                verbose=False
            )
            print(
                '\taccuracy:', round(accuracy, 4)
            )
            print(
                '\tloss:', round(loss, 4)
            )
        print('##########################################\n')