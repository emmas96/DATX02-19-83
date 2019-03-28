import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


class heatMap:
    def __plot(self, data):
        ax = sns.heatmap(data=data, square=True, annot=True, cmap="Blues_r", vmin=0, vmax=1, fmt=".1f", linewidths=".5")
        self.fig = ax
        plt.show()

    def __init__(self, data):
        self.__plot(data)

    def updateHeatMap(self, data):
        self.__plot(data)


if __name__ == "__main__":
    plot = heatMap(np.random.rand(3, 3))

    while True:
        input("New random")
        plot.updateHeatMap(np.random.rand(3, 3))

