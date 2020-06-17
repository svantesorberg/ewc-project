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
            
            #tf.print(self.trained_parameters_per_task)
            for params, fisher in zip(self.trained_parameters_per_task, self.fisher_diagonal_per_task):
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
            layer.kernel_regularizer.set_constant(400)
            layer.bias_regularizer.set_constant(400)

            # Update weight parameters
            new_parameters = task['trained_parameters'][2*i]
            new_fisher = task['fisher_diagonal'][2*i]
            layer.kernel_regularizer.add_new_parameters(new_parameters)
            layer.kernel_regularizer.add_new_fisher_diagonal(new_fisher)

            # Update bias parameters
            new_parameters = task['trained_parameters'][2*i + 1]
            new_fisher = task['fisher_diagonal'][2*i + 1]
            layer.bias_regularizer.add_new_parameters(new_parameters)
            layer.kernel_regularizer.add_new_fisher_diagonal(new_fisher)
    

    # Build a fully connected network with two hidden layers
    # Currently static width = 400
    def _build_model(self):
        self.model = tf.keras.models.Sequential([
            tf.keras.layers.Input(shape=self.input_shape, name='input'),
            tf.keras.layers.Dense(
                400, 
                activation='relu',
                bias_regularizer=self.EWC_Regularizer(),
                kernel_regularizer=self.EWC_Regularizer(),
                name='fc1'
            ),
            tf.keras.layers.Dense(
                400, 
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
            #optimizer = tf.keras.optimizers.SGD(
            #    learning_rate=self.learning_rate
            #),
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

            print('Training on task', get_name_or_id(task))

            self.model.fit(
                task['X_train'], 
                task['Y_train'],
                epochs = self.n_epochs,
                batch_size=self.batch_size
            )
            
            task['trained_parameters'] = self.model.get_weights()

            # Fisher Matrix 
            print('--------------')
            print('Computing gradients')
            with tf.GradientTape() as outer_tape:
                outer_tape.watch(self.model.trainable_weights)
                with tf.GradientTape() as inner_tape:
                    inner_tape.watch(self.model.trainable_weights)
                    predictions = self.model(task['X_train'])
                    loss = tf.keras.losses.categorical_crossentropy(task['Y_train'], predictions)
                grads = inner_tape.gradient(loss, self.model.trainable_weights)
            second_derivative = outer_tape.gradient(grads, self.model.trainable_weights)
            #grads = list(map(lambda x: x.numpy(), grads))
            #squared_gradients = list(map(tf.square, grads))
            #task['fisher_diagonal'] = squared_gradients
            task['fisher_diagonal'] = list(map(tf.abs, second_derivative))
            print('--------------')
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
            print('Evaluating task', get_name_or_id(task))
            loss, accuracy = self.model.evaluate(task['X_test'], task['Y_test'])
            #print(
            #    'Task', get_name_or_id(task), 
            #    'test accuracy:', accuracy, 
            #    'loss:', loss
            #)