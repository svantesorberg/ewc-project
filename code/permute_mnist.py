import numpy as np

def get_permuted_dataset(dataset_train, dataset_test, seed=None):
    n_train = dataset_train.shape[0]
    n_test = dataset_test.shape[0]
    input_shape = dataset_train.shape[1:]
    # This might be wonky if the dataset contains multiple channels
    # for images that we may not want to permute...
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