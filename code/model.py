import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as kb
import multiprocessing

from helpers import get_name_or_id
# E.G with VGG16 (138 000 000 parameters), a batch size of 16 would require ~8-9GB RAM

# FOR TESTING
np.random.seed(1337)
tf.random.set_seed(1337)

class EWC_Network():
    
    def __init__(
        self, 
        n_epochs, 
        batch_size, 
        input_shape, 
        n_classes, 
        learning_rate = 1e-4,
        gradient_batch_size = 1,
        ewc_lambda = 5000 # Tune this
    ):
        self.tasks = []                         # Info/data for the tasks
        self.n_epochs = n_epochs                # Epochs used in training
        self.batch_size = batch_size            # Batch size used in training
        self.learning_rate = learning_rate      # LR used in training
        self.input_shape = input_shape          # Data input shape
        self.n_classes = n_classes              # Number of output classes

        # Parameters not specific to the problem
        #   gradient_batch_size:
        #       The number of data points to combine when calculating
        #       the gradient that the empirical Fisher matrix is based on.
        #       For a true empirical Fisher, it should be 1 - but this 
        #       computation is rather slow. However, the norm of the gradient
        #       is affected by the batch size, so changing it is not really
        #       recommended unless other parameters, specifically 
        #       ewc_lambda are also tuned.
        #
        #   ewc_lambda:
        #       Multiplicative constant used when computing the regularization
        #       term.
        self.gradient_batch_size = gradient_batch_size 
        self.ewc_lambda = ewc_lambda 


        self._build_model()

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
        j = 0
        for i, layer in enumerate(self.model.layers):

            if layer.trainable and len(layer.trainable_weights) > 0:

                # Update weight parameters
                new_parameters = task['trained_parameters'][2*j]
                new_fisher = task['fisher_diagonal'][2*j]
                layer.kernel_regularizer.add_new_parameters(new_parameters)
                layer.kernel_regularizer.add_new_fisher_diagonal(new_fisher)

                # Update bias parameters
                new_parameters = task['trained_parameters'][2*j + 1]
                new_fisher = task['fisher_diagonal'][2*j + 1]
                layer.bias_regularizer.add_new_parameters(new_parameters)
                layer.bias_regularizer.add_new_fisher_diagonal(new_fisher)
                j += 1

    
    def _build_model(self):
        self.model = tf.keras.models.Sequential([
            #tf.keras.layers.Input(shape=self.input_shape, name='input'),
            tf.keras.layers.Input(shape=(28, 28, 1), name='input'),
            tf.keras.layers.Conv2D(
                64, 
                (2, 2),
                padding='same',
                bias_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda),
                kernel_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda),
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
                bias_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda),
                kernel_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda),
                name='fc1'
            ),
            tf.keras.layers.Dense(
                1024, 
                activation='relu', 
                bias_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda),
                kernel_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda), 
                name='fc2'
            ),
            tf.keras.layers.Dense(
                self.n_classes, 
                activation='softmax',
                bias_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda),
                kernel_regularizer=self.EWC_Regularizer(constant=self.ewc_lambda), 
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
                verbose=True
            )

            print('\rTraining on', get_name_or_id(task), 'done!')
            #task['trained_parameters'] = self.model.get_weights()

            self.store_trained_parameters(task)
            self.compute_fisher(task)

            # After model has been trained, we need to recompile it
            # in order to use the updated the regularization function
            self.update_regularization_functions(task)
            self._compile_model()

    def store_trained_parameters(self, task):
        task['trained_parameters'] = [
            tf.convert_to_tensor(np.copy(weight)) 
            for weight in self.model.get_weights()
        ]
        

    def compute_fisher(self, task):
        # Fisher matrix by batches
        print('\nComputing empirical Fisher - this may take a while')
        data = task['X_train']
        labels = task['Y_train']

        num_splits = int(data.shape[0] / self.gradient_batch_size)
        data_split = np.array_split(data, num_splits)
        label_split = np.array_split(labels, num_splits)

        # Determine the shape of the returned gradient and initialize
        # data structure that will hold the cumulative gradients data
        # w.r.t. the batches 
        sums = [
            np.zeros(shape = w.shape) for w in self.model.trainable_weights
        ]

        for i, (batch_data, batch_label) in enumerate(
            zip(data_split, label_split)
        ):
            progress = round(i / num_splits * 100)
            print('\r[{0}{1}] {2}%'.format('#'*int(progress/10), ' '*(10 - int(progress/10)), progress), end='')
            with tf.GradientTape() as tape:
                tape.watch(self.model.trainable_weights)
                predictions = self.model(batch_data)
                # Calculates loss for each data point in the batch
                loss = tf.keras.losses.categorical_crossentropy(
                    batch_label, 
                    predictions
                )

            # Using tape.gradients() directly on the loss tensor will
            # sum all the gradients together, which may not be ideal,
            # but speeds up computation. Another consequence of this
            # is that the batch size in the gradient computation
            # affects what the value lambda in the regularization 
            # function should be. We try to avoid this by dividing the 
            # gradient by the batch size, thus getting the 
            # "average gradient" for that batch.
            # For an exact (empirical) Fisher, use a batch size of 1.
            gradient = tape.gradient(
                loss, 
                self.model.trainable_weights
            )

            # Add the square batch_size times to compensate,
            # This may be a bit "cheaty" but at least we get 
            # values that are similar regardless of batch size,
            # which means we shouldn't have to adjust lambda too much
            sums = [
                s + 
                (
                    (gradient[j])**2
                )
                for j, s in enumerate(sums)
            ]

        # Divide by number of data points to normalize, note that 
        # this normalization will not yield comparable results 
        # for different batch sizes!
        task['fisher_diagonal'] = [s / num_splits for s in sums]
        # print(
        #     'Fisher sum:',
        #     list(map(np.sum, task['fisher_diagonal']))
        # )
        # print(
        #     'Fisher average:',
        #     list(map(np.average, task['fisher_diagonal']))
        # )

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