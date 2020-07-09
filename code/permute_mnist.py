import numpy as np

def get_permuted_dataset(dataset_train, dataset_test, seed=None):
    """Permute the images in the (MNIST )dataset by changing the positions 
    of the pixels in a random manner. Could probably be used to permute other 
    image datasets as well, but no guarantees.

        Parameters:
            dataset_train:  training data (MNIST)
            dataset_test:   test data (MNIST)
            seed = None:    random seed to use for permutation

        Returns:
            A tuple (train, test) with permuted training and test 
            data in the same order and shape as the input data.
    """
    n_train = dataset_train.shape[0]
    n_test = dataset_test.shape[0]
    input_shape = dataset_train.shape[1:]
    # This might be wonky if the dataset contains multiple channels
    # for images that we may not want to permute...
    # Fortunately, MNIST doesn't.
    vec_len = 1
    for dim in input_shape:
        vec_len *= dim

    permutation = np.random.RandomState(seed=seed).permutation(vec_len)
    permuted_train = np.array(
        [
            x[permutation].reshape(input_shape) 
            for x in dataset_train.reshape(n_train, vec_len)
        ]
    )
    permuted_test = np.array(
        [
            x[permutation].reshape(input_shape) 
            for x in dataset_test.reshape(n_test, vec_len)
        ]
    )
    return (permuted_train, permuted_test)

def get_shuffled_dataset(dataset_train, dataset_test, square_side=7, seed=None):
    """Shuffle the images in the dataset (MNIST) by dividing it into squares
    of equal size and shuffling the position of those squares within
    the image.

        Parameters:
            dataset_train:      training data (MNIST)
            dataset_test:       test data (MNIST)
            square_side = 7:    size of the squares that the images are divided
                                into (must be a divider of 28)
            seed = None:    random seed to use

        Returns:
            A tuple (train, test) with permuted training and test 
            data in the same order and shape as the input data.
    """

    # N.b. this implementation is ridiculously inefficient and should probably
    # not be used by anyone in their right mind.

    assert(28 % square_side == 0)
    num_splits = int(28 / square_side)
    
    n_train = dataset_train.shape[0]
    n_test = dataset_test.shape[0]
    input_shape = dataset_train.shape[1:]
    
    dataset_train = dataset_train.reshape(n_train, 28, 28)
    dataset_test = dataset_test.reshape(n_test, 28, 28)
    
    permutation = np.random.RandomState(seed=seed).permutation((num_splits**2))

    new_train = []
    new_test = []

    for data in dataset_train:
        lines = []
        for i in range(num_splits):
            lines.append(np.hsplit(np.vsplit(data, num_splits)[i], num_splits))
            
        
        squares = np.array([item for sublist in lines for item in sublist])
        
        #Permute 
        squares = squares[permutation]
        

        # Group them together again
        grouped = []
        for i in range(num_splits):
            grouped.append(squares[i*num_splits:i*num_splits+num_splits])
        # Stitch back together
        tobestacked = list(map(np.hstack, grouped))

        new_train.append(np.vstack(tobestacked).reshape(input_shape))
    
    for data in dataset_test:
        lines = []
        for i in range(num_splits):
            lines.append(np.hsplit(np.vsplit(data, num_splits)[i], num_splits))
            
        
        squares = np.array([item for sublist in lines for item in sublist])
        
        #Permute 
        squares = squares[permutation]
        

        # Group them together again
        grouped = []
        for i in range(num_splits):
            grouped.append(squares[i*num_splits:i*num_splits+num_splits])
        # Stitch back together
        tobestacked = list(map(np.hstack, grouped))
        new_test.append(np.vstack(tobestacked).reshape(input_shape))

    return (np.array(new_train), np.array(new_test))



        
        

    
