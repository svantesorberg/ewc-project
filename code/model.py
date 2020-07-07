import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as kb
import multiprocessing
import os
import tempfile

# FOR TESTING
np.random.seed(1337)
tf.random.set_seed(1337)

class EWC_Network():
    # DOCS

    class MultipleValidationSets(tf.keras.callbacks.Callback):
        # Shamelessly copied (more or less) from 
        # https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras

        def __init__(self, validation_sets, verbose=0, batch_size=None):
            """
            :DOCS:
            """
            super(EWC_Network.MultipleValidationSets, self).__init__()
            self.validation_sets = validation_sets
            for validation_set in validation_sets:
                assert(len(validation_set) == 3)
            self.epoch = []
            self.history = {}
            self.verbose = verbose
            self.batch_size = batch_size

        def on_train_begin(self, logs=None):
            self.epoch = []
            self.history = {}

        def on_epoch_end(self, epoch, logs=None):
            logs = logs or {}
            self.epoch.append(epoch)

            #for k, v in logs.items():
            #    self.history.setdefault(k, []).append(v)
            
            for validation_set in self.validation_sets:
                validation_data, validation_target, set_name = validation_set
            
                results = self.model.evaluate(
                    x = validation_data,
                    y = validation_target,
                    verbose = self.verbose,
                    batch_size = self.batch_size
                )

                for i, result in enumerate(results):
                    valuename = self.model.metrics[i].name
                    self.history.setdefault(set_name, {}).setdefault(
                        valuename, []
                    ).append(result)
                    #self.history.setdefault(valuename, []).append(result)
    
    class EWC_Regularizer(tf.keras.regularizers.Regularizer):
        # This class represents the regularization function in EWC, i.e.
        # the penalty term associated with the current parameters based 
        # on how much they differ from previously trained parameters for 
        # different tasks.
        def __init__(   self, 
                        constant = 1,
                    ):
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

        def get_config(self):
            return {'constant': self.constant}

        @classmethod
        def from_config(cls, config):
            return cls(**config)


        def set_parameters(self, parameters):
            self.trained_parameters_per_task = []
            for p in parameters:
                self.trained_parameters_per_task.append(p)
            #self.trained_parameters_per_task = parameters

        def set_fisher(self, fisher):
            self.fisher_diagonal_per_task = fisher

        def get_constant(self):
            return self.constant
        
        def set_constant(self, constant):
            self.constant = constant


    def __init__(
        self,
        model, 
        n_epochs, 
        batch_size, 
        input_shape, 
        n_classes, 
        num_tasks_to_remember = -1,
        learning_rate = 1e-4,
        gradient_batch_size = 1,
        ewc_lambda = 10000 # Tune this
    ):
        self.tasks = []                         # Info/data for the tasks
        self._model = model                     # Base keras model
        self.n_epochs = n_epochs                # Epochs used in training
        self.batch_size = batch_size            # Batch size used in training
        self.learning_rate = learning_rate      # LR used in training
        self.input_shape = input_shape          # Data input shape
        self.n_classes = n_classes              # Number of output classes
        self._num_tasks_to_remember = num_tasks_to_remember

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
        self._ewc_layer_indexes = []

        self._determine_ewc_layers()
        self._set_regularization_functions()
        self._compile_model()

        print('Model built. Here is a summary:',end='\n\n')
        self._model.summary(print_fn=lambda s: print('\t' + s))
        print()

    
        
    # Determine which layers should use the EWC regularizer.
    # For now, the user has no choice. All Dense and Conv2D 
    # layers will use EWC.
    def _determine_ewc_layers(self):
        for idx, layer in enumerate(self._model.layers):
            if (
                isinstance(layer, tf.keras.layers.Conv2D) or 
                isinstance(layer, tf.keras.layers.Dense)
            ):
                self._ewc_layer_indexes.append(idx)

    # Set EWC regularization function for relevant layers
    def _set_regularization_functions(self):
        # This is more complicated than you would expect,
        # because keras does not apply these changes
        # (even with model.compile()).
        # We have to perform the obtuse process of 
        # saving the model config and weights and then 
        # reloading it... Since we're only doing it at
        # the beginning we can actually skip the weights...
        # This feels very ugly, but is apparently the way to do it. 
        # At least we don't have to do it between task.
        # https://sthalles.github.io/keras-regularizer/

        for layer_idx in self._ewc_layer_indexes:
            layer = self._model.get_layer(index=layer_idx)
            layer.bias_regularizer = self.EWC_Regularizer(constant=self.ewc_lambda)
            layer.kernel_regularizer = self.EWC_Regularizer(constant=self.ewc_lambda)
        self._reload_model()

    def _reload_model(self):
        # Save config and weights
        model_json = self._model.to_json()
        #tmp_weights_path = os.path.join(tempfile.gettempdir(), 'weights.h5')
        #self._model.save_weights(tmp_weights_path)

        # Reload from config & restore weights
        with tf.keras.utils.custom_object_scope(
            {"EWC_Regularizer": self.EWC_Regularizer}
        ):
            self._model = tf.keras.models.model_from_json(model_json)
            #self._model.load_weights(tmp_weights_path, by_name=True)


        print(self._model.layers[0].kernel_regularizer.constant)

    # Update the regularization function with the parameters of the n latest
    # learned tasks
    def _update_regularization_functions(self):
        # All tasks that have been learned
        available = [
            i for i, task in enumerate(self.tasks) 
            if 'trained_parameters' in task
        ]

        # Determine which tasks should be remembered at this step
        if self._num_tasks_to_remember == -1:
            # Remember everything that has been learned
            tasks_to_remember = available
        else:
            # Only remember the latest num_tasks_to_remember tasks
            tasks_to_remember = available[-self._num_tasks_to_remember:]
        
        # Now add the learned parameters and fisher diagonal
        # for these tasks to each layer that uses EWC
        for idx in self._ewc_layer_indexes:
            layer = self._model.get_layer(index=idx)
            kernel_params = [
                self.tasks[task_id]['trained_parameters'][idx]['kernel']
                for task_id in tasks_to_remember
            ]
            kernel_fisher = [
                self.tasks[task_id]['fisher_diagonal'][idx]['kernel']
                for task_id in tasks_to_remember
            ]

            layer.kernel_regularizer.set_parameters(kernel_params)
            layer.kernel_regularizer.set_fisher(kernel_fisher)

            bias_params = [
                self.tasks[task_id]['trained_parameters'][idx]['bias']
                for task_id in tasks_to_remember
            ]
            bias_fisher = [
                self.tasks[task_id]['fisher_diagonal'][idx]['bias']
                for task_id in tasks_to_remember
            ]

            layer.bias_regularizer.set_parameters(bias_params)
            layer.bias_regularizer.set_fisher(bias_fisher)

    
    def _build_model(self):
        pass

    def _compile_model(self):
        self._model.compile(
            optimizer = tf.keras.optimizers.RMSprop(
                learning_rate=self.learning_rate
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    # Train models on the tasks specified in tasks vector
    # (in the order that they appear)
    def train_model(self, record_history = False):

        for i, task in enumerate(self.tasks):
            print(  'Training on', 
                    self._get_name_or_id(task), 
                    flush=True
            )

            if record_history:
                task['history'] = {}
                # We only care about the validation sets for the
                # datasets we have trained the model on so far
                validation_data = [
                    (task['X_test'], task['Y_test'], self._get_name_or_id(task))
                    for task in self.tasks[:i+1]
                ]
                history_callback = self.MultipleValidationSets(validation_data)
                callbacks = [history_callback]
            else:
                callbacks = None


            self._model.fit(
                task['X_train'], 
                task['Y_train'],
                epochs = self.n_epochs,
                batch_size=self.batch_size,
                verbose=True,
                callbacks=callbacks
            )

            if record_history:
                # Save the results of all the relevant tasks
                for trained_task in self.tasks[:i+1]:
                    history_from_trained_task = history_callback.history[
                        self._get_name_or_id(trained_task)
                    ]
                    for metric, values in history_from_trained_task.items():
                        trained_task['history'].setdefault(metric, []).extend(
                            values
                        )

            print('\rTraining on', self._get_name_or_id(task), 'done!')

            self._store_trained_parameters(task)
            self._compute_fisher(task)

            # After model has been trained, we need to recompile it
            # in order to use the updated the regularization function
            self._update_regularization_functions()
            self._compile_model()
            
            # ONLY FOR DEBUG PURPOSES
            #self.evaluate()

    def _store_trained_parameters(self, task):
        parameters_per_layer = {}

        for idx in self._ewc_layer_indexes:
            weights = self._model.get_layer(index=idx).get_weights()
            kernel_weights = weights[0]
            bias_weights = weights[1]

            parameters_per_layer[idx] = {
                'kernel': tf.convert_to_tensor(np.copy(kernel_weights)),
                'bias': tf.convert_to_tensor(np.copy(bias_weights))
            }
        
        task['trained_parameters'] = parameters_per_layer

    def _compute_fisher(self, task):
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

        layer_weights = {
            idx: {
                'kernel': self._model.get_layer(index=idx).weights[0],
                'bias': self._model.get_layer(index=idx).weights[1]
            }
            for idx in self._ewc_layer_indexes    
        }

        sums = {
            idx: {
                'kernel': np.zeros(shape = layer_weights[idx]['kernel'].shape),
                'bias': np.zeros(shape = layer_weights[idx]['bias'].shape)
            }
            for idx in self._ewc_layer_indexes
        }

        for i, (batch_data, batch_label) in enumerate(
            zip(data_split, label_split)
        ):
            progress = round(i / num_splits * 100)
            print('\r[{0}{1}] {2}%'.format('#'*int(progress/10), ' '*(10 - int(progress/10)), progress), end='')
            with tf.GradientTape() as tape:
                predictions = self._model(batch_data)
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
                layer_weights
            )

            for idx in self._ewc_layer_indexes:
                sums[idx]['kernel'] += gradient[idx]['kernel']**2
                sums[idx]['bias'] += gradient[idx]['bias']**2
            

        # Divide by number of data points to normalize, note that 
        # this normalization will not yield comparable results 
        # for different batch sizes (so it's kind of pointless)!
        for idx in self._ewc_layer_indexes:
            sums[idx]['kernel'] /= num_splits
            sums[idx]['bias'] /= num_splits

        task['fisher_diagonal'] = sums
        print()

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


    # Simple evaluation of performance on specified tasks
    def evaluate(self, task_ids = None):
        if not task_ids:
            task_ids = [i for i in range(len(self.tasks))]
        
        print('\n##########################################')
        print('Evaluating all tasks\n')

        for task_id in task_ids:
            task=self.tasks[task_id]
            assert(task_id == task['meta']['id'])
            print('Evaluating', self._get_name_or_id(task), end='...\n')
            loss, accuracy = self._model.evaluate(
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

    def get_history(self):
        """Get the recorded history for each task"""
        ret_list = []
        for task in self.tasks:
            if 'history' not in task:
                return [] 
            else:
                ret_list.append({
                    'name': self._get_name_or_id(task),
                    'history': task['history']
                })
        return ret_list

    def _get_name_or_id(self, task):
        if task['meta']['name']:
            return task['meta']['name']
        else:
            return 'task ' + task['meta']['id']