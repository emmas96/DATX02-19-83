import matplotlib.pyplot as plt
import csv
from os import listdir
from os.path import isfile, join
import pprint


def __group_by_run(files):
    temp = {}
    for f in files:
        group_name = f[:-8]
        if group_name not in temp:
            temp[group_name] = [f]
        else:
            temp[group_name].append(f)

    return temp


def plot_frozen_res(dir):
    # Read files from dir and group them
    all_runs = __group_by_run([f for f in listdir(dir) if isfile(join(dir, f))])

    for run_params, files in all_runs.items():
        print(run_params)
        tot_epsilon = []
        tot_win = []
        epochs = []

        for file in files:

            curr_wins = []
            curr_epochs = []

            with open(join(dir, file), 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # Check if first run
                if len(epochs) == 0:
                    for row in data:
                        epochs.append(int(row[0]))
                        tot_win.append(int(row[2]))
                        tot_epsilon.append(float(row[3]))

                        curr_wins.append(int(row[2]))
                        curr_epochs.append(int(row[0]))
                else:
                    for row in data:
                        epoch = int(row[0]) - 1
                        if epoch >= len(tot_win) - 1:
                            continue
                        tot_win[epoch] += int(row[2])
                        tot_epsilon[epoch] += float(row[3])

                        curr_wins.append(int(row[2]))
                        curr_epochs.append(int(row[0]))

            plt.plot(curr_epochs, curr_wins, alpha=0.3, label=f"run: {file[-5:-4]}")

        print(f"Len tot_win: {len(tot_win)}, len epochs: {len(epochs)}")

        # Calc avg
        print(f"Tot win: {tot_win}")
        tot_win = list(map(lambda x: float(x)/len(files), tot_win))
        print(f"avg win: {tot_win}\n")

        # Plot graphs
        plt.plot(epochs, tot_win, label=run_params)
        plt.title(run_params)
        plt.show()

    plt.xlabel('epoch')
    plt.ylabel('tot win')
    #plt.legend()

    plt.show()


plot_frozen_res("../Data/Frozen/train")
