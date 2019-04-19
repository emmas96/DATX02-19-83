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

    problems = {}

    for run_params, files in all_runs.items():
        tot_epsilon = []
        tot_win = []
        epochs = []

        for file in files:
            with open(join(dir, file), 'r') as csvfile:
                data = csv.reader(csvfile, delimiter=',')

                # Check if first run
                if len(epochs) == 0:
                    print("Was here")
                    for row in data:
                        epochs.append(int(row[0]))
                        tot_win.append(int(row[2]))
                        tot_epsilon.append(float(row[3]))
                else: # Not first run
                    print("now here")
                    for row in data:
                        epoch = int(row[0]) - 1
                        if epoch >= len(tot_win) - 1:
                            continue
                        tot_win[epoch] += int(row[2])
                        tot_epsilon[epoch] += float(row[3])
                    print(f"tot_win: {len(tot_win)}")


plot_frozen_res("../Data/Frozen/train")
