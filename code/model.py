import tensorflow as tf
import numpy as np
import tensorflow.keras.backend as kb
import os
import tempfile

class EWC_Network():
    """Use Elastic Weight Consolidation (EWC) in a Keras sequential model.

    Usage:
        The EWC Network is built from a Keras sequential model. The EWC 
        regularization is applied only to the Dense and Conv2D layers of the 
        input model.

        Tasks can be added to the network by defining their training and 
        test data, as well as a task name. The metrics (at the moment only 
        loss & accuracy) for each trained task can be recorded while training 
        new tasks, giving the user a sense of how the performance of the
        network degrades on previous tasks as new ones are learnt.

        A typical use case could be:
            1. Create a Keras sequential model
            2. Instantiate EWC_Network based on said model
            3. Add tasks to the EWC_Network
            4. Train the EWC_Network on all tasks
            5. Analyze the performance 
    """

    class _MultipleValidationSets(tf.keras.callbacks.Callback):
        # Callback used to record metrics for multiple test sets,
        # i.e. test sets for all the tasks that have been trained already.
        # Mostly copied from:
        # https://stackoverflow.com/questions/47731935/using-multiple-validation-sets-with-keras

        def __init__(self, validation_sets, verbose=0, batch_size=None):
            super(EWC_Network._MultipleValidationSets, self).__init__()
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
            
            # Iterate over all the validation sets on which we should
            # record metrics, and store them in history
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
    
    class _EWC_Regularizer(tf.keras.regularizers.Regularizer):
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
            # We want to add one penalty term for each previously learned
            # task, so we must iterate over them and compute sum the 
            # individual penalties.
            for params, fisher in zip(
                self.trained_parameters_per_task, self.fisher_diagonal_per_task
            ):
                penalty += (
                    (self.constant / 2) * tf.reduce_sum(
                        tf.multiply(fisher, tf.square(weights - params))
                    )
                )
            return penalty

        def set_parameters(self, parameters):
            self.trained_parameters_per_task = parameters

        def set_fisher(self, fisher):
            self.fisher_diagonal_per_task = fisher

        def get_constant(self):
            return self.constant
        
        def set_constant(self, constant):
            self.constant = constant

        def get_config(self):
            # Required for saving, e.g. as JSON.
            return {'constant': self.constant}

        @classmethod
        def from_config(cls, config):
            # Required for restoring.
            return cls(**config)


    def __init__(
        self,
        model, 
        n_epochs, 
        batch_size,  
        ewc_lambda, # Tune this
        learning_rate = 1e-4,
        num_tasks_to_remember = -1, # Default is remember all tasks
        gradient_batch_size = 1,
    ):
        """Constructor for the EWC Network

            Parameters:
                model:                  Sequential Keras model (not compiled - 
                                        it will be recompiled with ADAM 
                                        optimizer and metrics = ['accuracy'])
                n_epochs:               Number of epochs to train each task
                batch_size:             Batch size used in training
                ewc_lambda:             Constant multiplier used on the EWC 
                                        regularization term. Higher values mean
                                        an increased importance of remembering 
                                        old tasks compared to performing well
                                        on new ones. This should be tuned for
                                        the user's purposes.
                learning_rate:          Learning rate used in training,
                                        default 1e-4
                num_tasks_to_remember:  Number old tasks to remember when 
                                        training a new task (-1 for remembering 
                                        everything, 0 for remembering nothing),
                                        default -1.
                
                gradient_batch_size:    Batch size used when computing the 
                                        gradient used for the Empirical Fisher 
                                        Matrix, default 1.
                
        """
        
        self.n_epochs = n_epochs                
        self.batch_size = batch_size            
        self.learning_rate = learning_rate     

        self._tasks = []                       
        self._model = model  
        # Num_tasks_to_remember must be: -1 for all tasks, else > 0
        assert(num_tasks_to_remember >= -1)                  
        self._num_tasks_to_remember = num_tasks_to_remember

        self.ewc_lambda = ewc_lambda 

        # A note on gradient_batch_size:
        #   The Empirical Fisher Matrix (or in this case, its diagonal) is
        #   based on computing gradients of the loss function. For computing
        #   the empirical Fisher diagonal, the loss gradients for each data
        #   point is squared and they are then added together (and normalized
        #   by dividing with the number of data points). 
        #   Now, this gradient can be computed on batches of the input data. 
        #   This is essentially the average (or sum - this can be controlled 
        #   by the user) of the loss gradient for each data point in the batch. 
        #   This is a lot quicker and is fine for the purposes of EWC, as the 
        #   average gradient will still be representative of how "important" a 
        #   particular weight is for a given task. However, performing the 
        #   computation in batches will affect the magnitude of the computed 
        #   penalty, since 
        #       ((g_1 + ... + g_n)/n)**2 != (g_1**2 + ... + g_n**2).
        #   this can be remedied *somewhat* by proper normalization. 
        #   Thus, tuning the constant used in the penalty calculation,
        #   ewc_lambda, requires taking the gradient batch size into 
        #   consideration.
        #
        #   I've opted for computing the Fisher matrix "properly" with a 
        #   default batch size of 1.

        self.gradient_batch_size = gradient_batch_size 
        
        # Initialize a field to keep track of which layers use EWC
        self._determine_ewc_layers() 

        # Set EWC regularization function for relevant layers
        self._set_regularization_functions()

        # Compile the model
        self._compile_model()

        print('Model built. Here is a summary:',end='\n\n')
        self._model.summary(print_fn=lambda s: print('\t' + s))
        print()

    
        
    
    def _determine_ewc_layers(self):
        # Determine which layers should use the EWC regularizer.
        # For now, the user has no choice. All Dense and Conv2D 
        # layers will use EWC.
        self._ewc_layer_indexes = []
        for idx, layer in enumerate(self._model.layers):
            if (
                isinstance(layer, tf.keras.layers.Conv2D) or 
                isinstance(layer, tf.keras.layers.Dense)
            ):
                self._ewc_layer_indexes.append(idx)

    def _set_regularization_functions(self):
        for layer_idx in self._ewc_layer_indexes:
            layer = self._model.get_layer(index=layer_idx)
            layer.bias_regularizer = self._EWC_Regularizer(constant=self.ewc_lambda)
            layer.kernel_regularizer = self._EWC_Regularizer(constant=self.ewc_lambda)
        
        # "Applying" these changes is more complicated than you would expect, 
        # because keras does not do it for you (even with model.compile()).
        # We have to perform the obtuse process of saving the model config (and 
        # weights if we had any pre-set weights) and then reloading it... 
        # Since we're only doing it at the beginning we may actually skip the 
        # weights. If any user would like to incorporate a model with 
        # pre-trained weights, they should un-comment and test the lines 
        # associated with backing up and restoring weights.
        
        # This solution feels very ugly, but is apparently the way to do it. 
        # At least we don't have to do it between tasks...
        # https://sthalles.github.io/keras-regularizer/
        self._reload_model()

    def _reload_model(self):
        # Save config
        model_json = self._model.to_json()
        # Save weights
        #tmp_weights_path = os.path.join(tempfile.gettempdir(), 'weights.h5')
        #self._model.save_weights(tmp_weights_path)

        # Reload from config & restore weights
        with tf.keras.utils.custom_object_scope(
            {"_EWC_Regularizer": self._EWC_Regularizer}
        ):
            self._model = tf.keras.models.model_from_json(model_json)
            #self._model.load_weights(tmp_weights_path, by_name=True)

    
    def _update_regularization_functions(self):
        # Update the regularization functions for each layer with the 
        # parameters of the n latest learned tasks

        # Determine all tasks that have been learned so far
        available = [
            i for i, task in enumerate(self._tasks) 
            if 'trained_parameters' in task
        ]

        # Determine which tasks should be remembered at this step
        # depending on num_tasks_to_remember
        if(
            self._num_tasks_to_remember == -1 or 
            # This second check is actually not needed because 
            # if a = [1, 2, 3] then e.g. a[-54] is still [1, 2, 3]
            self._num_tasks_to_remember > len(available) 
        ):
            # Remember everything that has been learned
            tasks_to_remember = available
        elif self._num_tasks_to_remember == 0:
            tasks_to_remember = []
        else:
            # Only remember the latest num_tasks_to_remember tasks
            tasks_to_remember = available[-self._num_tasks_to_remember:]
        
        # Now add the learned parameters and fisher diagonal
        # for these tasks to each layer that uses EWC
        for idx in self._ewc_layer_indexes:
            layer = self._model.get_layer(index=idx)
            kernel_params = [
                self._tasks[task_id]['trained_parameters'][idx]['kernel']
                for task_id in tasks_to_remember
            ]
            kernel_fisher = [
                self._tasks[task_id]['fisher_diagonal'][idx]['kernel']
                for task_id in tasks_to_remember
            ]

            layer.kernel_regularizer.set_parameters(kernel_params)
            layer.kernel_regularizer.set_fisher(kernel_fisher)

            bias_params = [
                self._tasks[task_id]['trained_parameters'][idx]['bias']
                for task_id in tasks_to_remember
            ]
            bias_fisher = [
                self._tasks[task_id]['fisher_diagonal'][idx]['bias']
                for task_id in tasks_to_remember
            ]

            layer.bias_regularizer.set_parameters(bias_params)
            layer.bias_regularizer.set_fisher(bias_fisher)

    def _compile_model(self):
        # Compile the model with default optimizer and accuracy metrics
        # In a future version, the user could be given more control over this
        self._model.compile(
            optimizer = tf.keras.optimizers.Adam(
                learning_rate=self.learning_rate
            ),
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

    def train_model(self, record_history = False):
        """ Train the model on the tasks that have been added, (in that order).

            Parameters:
                record_history: If set to True, metrics will be recorded while
                                training. Specifically, metrics are recorded
                                for any task that is currently being trained,
                                or has already been trained, on. Default is 
                                False.
        """

        for i, task in enumerate(self._tasks):
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
                    for task in self._tasks[:i+1]
                ]
                history_callback = self._MultipleValidationSets(validation_data)
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
                for trained_task in self._tasks[:i+1]:
                    history_from_trained_task = history_callback.history[
                        self._get_name_or_id(trained_task)
                    ]
                    for metric, values in history_from_trained_task.items():
                        trained_task['history'].setdefault(metric, []).extend(
                            values
                        )

            print('\rTraining on', self._get_name_or_id(task), 'done!')

            # After training, we want to save these "optimal" parameters for
            # the task, as well as the (empirical) Fisher diagonal
            self._store_trained_parameters(task)
            self._compute_fisher(task)

            # After model has been trained, we need to update the regularization
            # functions for the EWC layer to take into account the most recently
            # trained task.
            self._update_regularization_functions()
            # Recompilation is needed after updating the regularization
            # function. It does not change the current weights.
            self._compile_model()

    def _store_trained_parameters(self, task):
        # Store the parameters that have been learned from a task
        # and associate them with the relevant layer

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
        print('\nComputing empirical Fisher - this may take a while')
        data = task['X_train']
        labels = task['Y_train']

        # Split the data based on the desired batch size
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

        # Compute gradient for each batch
        for i, (batch_data, batch_label) in enumerate(
            zip(data_split, label_split)
        ):
            progress = round(i / num_splits * 100)
            print(  '\r[{0}{1}] {2}%'.format('#'*int(progress/10), 
                    ' '*(10 - int(progress/10)), progress), 
                    end=''
            )
            with tf.GradientTape() as tape:
                predictions = self._model(batch_data)
                # Loss for the batch
                loss = tf.keras.losses.categorical_crossentropy(
                    batch_label, 
                    predictions
                    # If we want to control how the loss is calculated for 
                    # a batch we can do it here, for example using
                    # reduction=tf.keras.losses.Reduction.SUM
                    # or
                    # reduction=tf.keras.losses.Reduction.NONE
                    # see:
                    # https://keras.io/api/losses/probabilistic_losses/#categoricalcrossentropy-class
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
            

        # Divide by number of splits to normalize, note that 
        # this normalization will not yield comparable results 
        # for different batch sizes (so it's somewhat pointless)!
        for idx in self._ewc_layer_indexes:
            sums[idx]['kernel'] /= num_splits
            sums[idx]['bias'] /= num_splits

        task['fisher_diagonal'] = sums
        print()

    def add_task(
        self, 
        X_train, Y_train, X_test, Y_test, 
        name = ''
    ):
        """Add a task to the model.

        Parameters:
            X_train:    Training data
            Y_train:    Training labels
            X_test:     Test data
            Y_test:     Test labels
            name:       reader-friendly name for the task
        """
        self._tasks.append({
            'X_train': X_train,         # Training input
            'Y_train': Y_train,         # Training output
            'X_test': X_test,           # Testing input
            'Y_test': Y_test,           # Testing output

            'meta': {                   # Info unrelated to network
                'id': len(self._tasks), # Id of the task
                'name': name            # Human-readable task name
            }
        })

    def evaluate(self):
        """Evaluate the network's performance sequentially on all tasks"""

        print('\n##########################################')
        print('Evaluating all tasks\n')

        for task in self._tasks:
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
        """Get the recorded history for each task. If the network is trained
        with record_history = True, the history is recorded for each epoch
        that a task is either being trained, or has already been trained, on.

            Returns:
                A list of dictionaries with the following structure
                {
                    name: <task name or id>,
                    history: {
                        <metric>: <list of the recorded metric for each epoch>
                    }
                }

                If history has not been recorded for any task, an empty list
                is returned.
            
            Example output (two tasks are trained for three epochs each):
                [
                    {
                        name: "First task",
                        history: {
                            loss: [0.2, 0.1, 0.05, 0.04, 0.02, 0.01]
                            accuracy: [0.88, 0.94, 0.96, 0.95, 0.92, 0.92]
                        }
                    },
                    {
                        name: "Second task",
                        history: {
                            loss: [0.45, 0.4, 0.32],
                            accuracy: [0.84, 0.92, 0.95]
                        }
                    }
                ]
        """
        ret_list = []
        for task in self._tasks:
            if 'history' in task:
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
