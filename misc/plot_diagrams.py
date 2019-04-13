import matplotlib.pyplot as plt
import csv
import os

#Will add all files from a dir into the plot
def plot_dir(dir):
    if not os.path.isdir(dir):
        print(f"\"{dir}\" is not a dir")
        return

    plotted = 0
    for file in os.listdir(dir):
        #skip dirs
        if not os.path.exists(dir + "/" + file):
            print("Skipped")
            continue

        plot_file(dir, file)
        plotted += 1

        if plotted % 5 == 0:
            plot_show()

    plot_show()


def plot_show():
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Interesting Graph\nCheck it out')
    plt.legend()
    plt.show()


def plot_file(dir, name):
    x = []
    y = []

    with open(dir + "/" + name, 'r') as csvfile:
        plots = csv.reader(csvfile, delimiter=',')
        for row in plots:
            x.append(int(row[0]))
            y.append(int(row[2])/x[-1])

        plt.plot(x, y, label=name[18:])


plot_dir("../Data/Frozen/train")