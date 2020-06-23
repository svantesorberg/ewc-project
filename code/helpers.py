import numpy as np
import matplotlib.pyplot as plt

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

class MyPlot():
    def __init__(self,
                 nrows, 
                 ncols, 
                 figsize):
        self.fig, self.axes = plt.subplots(nrows=nrows, 
                                           ncols=ncols, 
                                           figsize=figsize)

def plot_random_images(images, examples=16, fig_suptitle=None, figsize=(8,8), fpath=None):

    imgs_index = np.random.choice(np.arange(len(images)), examples, replace=False)

    plot = MyPlot(int(examples/np.sqrt(examples)), 
                  int(examples/np.sqrt(examples)), 
                  figsize=figsize)
    plot.axes = plot.axes.ravel()
    image_shape = images[0].shape
    for idx, _ in enumerate(plot.axes):
        X = images[imgs_index[idx]]
        if len(image_shape) == 2:
            plot.axes[idx].imshow(X=X,cmap="gray")
        else:
            plot.axes[idx].imshow(X=X)
        plot.axes[idx].axis('off')
    plot.fig.suptitle(fig_suptitle, fontsize=16)
    if fpath:
        plot.fig.savefig(fpath)
    else:
        plt.show()