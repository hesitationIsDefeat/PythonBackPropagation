import numpy as np

def to_onehot(y_train, num_classes):
    y_train_onehot = np.zeros((y_train.size, num_classes))

    y_train_onehot[np.arange(y_train.size), y_train] = 1
    return y_train_onehot