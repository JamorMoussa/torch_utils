import matplotlib.pyplot as plt
import numpy as np


__all__ = ["plot_accuracy", "plot_loss"]

def plot_accuracy(train_acc: list, test_acc: list):

    train_acc = np.array(train_acc)
    test_acc = np.array(test_acc)

    plt.plot(train_acc)
    plt.plot(test_acc)
    
    plt.show()


def plot_loss(train_loss: list, test_loss: list):

    train_loss = np.array(train_loss)
    test_loss = np.array(test_loss)

    plt.plot(train_loss)
    plt.plot(test_loss)
    
    plt.show()
