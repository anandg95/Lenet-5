from tensorflow.examples.tutorials.mnist import input_data
import numpy as np
import os


def preprocess_data():
    train_dir = os.path.join(os.getcwd(), "train_dir")
    mnist = input_data.read_data_sets(train_dir, reshape=False)

    X_train, y_train = mnist.train.images, mnist.train.labels
    X_test, y_test = mnist.test.images, mnist.test.labels
    X_validation, y_validation = mnist.validation.images, mnist.validation.labels

    print("Image shape: " + str(X_train[0].shape))

    # mnist is 28x28, model expects 32x32 input images
    # X_train is of shape (N_images, 28, 28, 1) - so pad axes 2 and 3 with (2,2) each
    X_train = np.pad(X_train, ((0, 0), (2, 2), (2, 2), (0, 0)), mode="constant")
    X_validation = np.pad(X_validation, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant")
    X_test = np.pad(X_test, ((0, 0), (2, 2), (2, 2), (0, 0)), "constant")

    return X_train, y_train, X_validation, y_validation, X_test, y_test
