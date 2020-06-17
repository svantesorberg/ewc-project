import numpy as np

def one_hot_encode(data, nb_classes = None):
    """Convert an iterable of indices to one-hot encoded labels."""
    if not nb_classes:
        nb_classes = len(np.unique(data))

    targets = np.array(data).reshape(-1)
    return np.eye(nb_classes)[targets]

def get_name_or_id(task):
    if task['meta']['name']:
        return task['meta']['name']
    else:
        return 'task ' + task['meta']['id']