import matplotlib.pyplot as plt
import numpy as np
from os import listdir, makedirs
from os.path import isfile, join, isdir

def save_fig(name, folder):
    path_to_res = join("../theory/", folder)

    # Save fig to disk
    if not isdir(path_to_res):
        makedirs(path_to_res)
    plt.savefig(join(path_to_res, name))

def ReLU(x):
    return np.maximum(x, 0)

def softplus(x):
    x = list(map(lambda x: 1 + np.exp(x), x))
    return np.log(x)

def heavyside(x):
    return np.heaviside(x, 0)

def sigmoid(x):
    return list(map(lambda x: np.divide(1, (1 + np.exp(-x))), x))

def tanh(x):
    return np.tanh(x)

def plot():
    x = np.arange(-3,3, step=0.01)
    plt.plot(x, ReLU(x), label="ReLU")
    plt.plot(x, softplus(x), label="softplus")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    save_fig("relu_softplus.png","")
    plt.show()

    plt.plot(x, heavyside(x), label="heaviside")
    plt.plot(x, sigmoid(x), label="sigmoid")
    plt.plot(x, tanh(x), label="tanh")

    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
    save_fig("heavyside_sigmoid_tanh.png", "")
    plt.show()

plot()