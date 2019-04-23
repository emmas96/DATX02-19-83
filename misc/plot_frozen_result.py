import matplotlib.pyplot as plt
import csv
from os import listdir, makedirs
from os.path import isfile, join, isdir
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


def create_axis():
    fig, ax_win = plt.subplots()
    ax_win.set_label('epoch')
    ax_win.set_ylabel('tot win')
    ax_epsi = ax_win.twinx()
    ax_epsi.set_ylabel('epsilon')
    fig.tight_layout()

    return fig, ax_win, ax_epsi


# Plot one plot per run
def plot_frozen_res(dir):
    # Read files from dir and group them
    all_runs = __group_by_run([f for f in listdir(dir) if isfile(join(dir, f))])

    for run_params, files in all_runs.items():
        print(run_params)
        tot_epsilon = []
        tot_win = []
        epochs = []
        first_file = True

        fig, ax_win, ax_epsi = create_axis()

        for file in files:

            curr_wins = []
            curr_epochs = []

            with open(join(dir, file), 'r') as csv_file:
                data = csv.reader(csv_file, delimiter=',')

                for row in data:
                    # Parse data
                    epoch = int(row[0])
                    win = int(row[2])
                    epsilon = float(row[3])

                    if first_file:
                        epochs.append(epoch)
                        tot_win.append(win)
                        tot_epsilon.append(epsilon)
                    else:
                        index = epoch - 1

                        tot_win[index] += win
                        tot_epsilon[index] += epsilon

                    curr_wins.append(win)
                    curr_epochs.append(epoch)

            first_file = False
            ax_win.plot(curr_epochs, curr_wins, alpha=0.2, label=f"run: {file[-5:-4]}")

        print(f"Len tot_win: {len(tot_win)}, len epochs: {len(epochs)}\n")

        # Calc avg
        avg_win = list(map(lambda x: float(x)/len(files), tot_win))
        avg_epsilon = list(map(lambda x: float(x)/len(files), tot_epsilon))

        # Plot avg graph
        ax_win.plot(epochs, avg_win, label="avg")
        ax_epsi.plot(epochs, avg_epsilon, label="epsilon")
        plt.title(run_params)

        # Add one common legend
        lines_win, labels_win = ax_win.get_legend_handles_labels()
        lines_epsi, labels_epsi = ax_epsi.get_legend_handles_labels()
        ax_win.legend(lines_win + lines_epsi, labels_win + labels_epsi, loc='center left')

        plt.show()
        break


def calc_avg_win(dir, files):
    tot_win = []
    epochs = []
    first_file = True

    for file in files:

        with open(join(dir, file), 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            for row in data:
                # Parse data
                epoch = int(row[0])
                win = int(row[2])

                if first_file:
                    epochs.append(epoch)
                    tot_win.append(win)
                else:
                    tot_win[epoch - 1] += win

        first_file = False

    # Calc avg
    avg_win = list(map(lambda x: float(x) / len(files), tot_win))
    return epochs, avg_win


# Plot one plot with all runs avg
def plot_frozen_res_avg(dir):
    # Constants
    nr_to_highlight = 5
    color = ['b', 'g', 'r', 'c', 'm']
    path_to_res = "../result/frozen/"

    # Read files from dir and group them
    all_runs = __group_by_run([f for f in listdir(dir) if isfile(join(dir, f))])
    best_runs = {}

    # Loop through all files one time to find the best result
    for run_params, files in all_runs.items():
        epochs, avg_win = calc_avg_win(dir, files)
        final_score = avg_win[-1]

        # Find the best results
        if len(best_runs) < nr_to_highlight:
            best_runs[run_params] = final_score
        else:
            min_run = min(best_runs, key=best_runs.get)
            min_value = best_runs[min_run]

            if min_value < final_score:
                best_runs.pop(min_run)
                best_runs[run_params] = final_score

    # Plot the results
    for run_params, files in all_runs.items():
        epochs, avg_win = calc_avg_win(dir, files)

        if run_params in best_runs.keys():
            plt.plot(epochs, avg_win, label=run_params, color=color.pop())
        else:
            plt.plot(epochs, avg_win, alpha=.2)

    plt.title("All runs")
    plt.xlabel('epoch')
    plt.ylabel('tot win')
    plt.legend()

    # Save fig to disk
    if not isdir(path_to_res):
        makedirs(path_to_res)
    plt.savefig(join(path_to_res, "All-runs.png"))

    plt.show()


plot_frozen_res_avg("../Data/Frozen/train")
# plot_frozen_res("../Data/Frozen/train")
