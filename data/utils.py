import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from scipy import ndimage
import torch
from torch.utils.data import TensorDataset


def get_prepared_data():
    """
    Loads and resizes the 28x28 MNIST images to 16x16 to match the data as described in the paper. Resized images are then ebedded into a 28x28 image for better performance in convolution. 
    Training set consists of 60,000 images and lables. Test set consists of 10,000 images and lables. The data is returned as a NumPy Array.

    Returns:
        (x_train, y_train), (x_test, y_test): Prepared images and labels for the training and testing process.
    """

    (x_train, y_train), (x_test, y_test) =  tf.keras.datasets.mnist.load_data()


    # scaling pixle values to [-1, 1]
    x_train = ((x_train.astype("float32") / 255.0) * 2) - 1
    x_test = ((x_test.astype("float32") / 255.0) * 2) - 1

    print(x_train[0])


    # resizing image to 16x16

    x_train = ndimage.zoom(x_train, (1, 16.0/28.0, 16.0/28.0), order=0)
    x_test = ndimage.zoom(x_test, (1, 16.0/28.0, 16.0/28.0), order=0)

    # embedding into a 28x28 array with pixel border -1

    new_train = np.full((60000, 28, 28), -1.0)
    new_test = np.full((10000, 28, 28), -1.0)

    row, col = 6, 6
    
    new_train[:, row:row+16, col:col+16] = x_train
    new_test[:, row:row+16, col:col+16] = x_test


    return (new_train, y_train), (new_test, y_test)


def create_dataset(images, lables):
    """
    Creates a PyTorch Dataset used for the the training process

    Params:
        images: NumPy Array of Image Data
        lables: NumPy Array of Lables
    
    Returns:
        TensorDataset: Dataset of the passed arguments
    """
    
    image_tensor = torch.from_numpy(images).float().unsqueeze(1)
    lable_tensor = torch.from_numpy(lables).long()
    return TensorDataset(image_tensor, lable_tensor)


def display_digit(image_set, lable_set, index):
    """displays the image at given index and its lable"""
    plt.imshow(image_set[index], cmap="gray")
    plt.title(lable_set[index])
    plt.show()















