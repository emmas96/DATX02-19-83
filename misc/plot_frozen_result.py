import matplotlib.pyplot as plt
import csv
from os import listdir, makedirs
from os.path import isfile, join, isdir
import pandas as pd
import seaborn as sns
import pprint


def __group_by_run(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]
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
    all_runs = __group_by_run(dir)

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

    translations = {}
    test_nr = 1

    # Read files from dir and group them
    all_runs = __group_by_run(dir)
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
            label = f"Test {test_nr}"
            test_nr += 1
            translations[label] = run_params
            plt.plot(epochs, avg_win, label=label, color=color.pop())
        else:
            plt.plot(epochs, avg_win, alpha=0)

    pprint.pprint(translations)

    plt.title("All runs")
    plt.xlabel('epoch')
    plt.ylabel('tot win')
    plt.legend()

    # Save fig to disk
    if not isdir(path_to_res):
        makedirs(path_to_res)
    plt.savefig(join(path_to_res, "All-runs.png"))

    plt.show()


def plot_frozen_heatmap(dir):
    # Constants
    track_parmas = ['Mb', 'G']  # First x, Second y
    track_as_float = ['G', 'Et']
    nr_chars_in_name = 18  # How many chars before parmas start

    all_runs = __group_by_run(dir)

    all_values = {}

    for run_params, files in all_runs.items():
        run_params = run_params[nr_chars_in_name:]
        print(f"Run params: {run_params}")

        params = run_params.split('_')

        not_tracked = []
        axis_values = [0, 0]

        for param in params:
            # Clear all digits
            param_type = ''.join([str(i) for i in param if not i.isdigit() and i != '.'])

            if param_type not in track_parmas:
                not_tracked.append(str(param))
            else:
                index = track_parmas.index(param_type)

                # Remove type
                axis_values[index] = param[len(param_type):]

                if param_type in track_as_float:
                    axis_values[index] = float(axis_values[index])
                else:
                    axis_values[index] = int(axis_values[index])

        print(f"Axis_values {axis_values}")

        not_tracked_key = "_".join(not_tracked)
        print(f'Not tracked key: {not_tracked_key}')

        _, avg_win = calc_avg_win(dir, files)

        if not_tracked_key in all_values:
            values = all_values[not_tracked_key]

            if axis_values[0] in values:
                values[axis_values[0]][axis_values[1]] = avg_win[-1]
            else:
                values[axis_values[0]] = {axis_values[1]: avg_win[-1]}
            print(f"Got here axis_values {axis_values} avg_win {avg_win[-1]}")
            pprint.pprint(values)
        else:
            values = {axis_values[0]: {axis_values[-1]: avg_win[-1]}}
            print(f"axis_values {axis_values} avg_win {avg_win[-1]}")
            pprint.pprint(values)

            # values[][axis_values[1]] = avg_win[-1]

        all_values[not_tracked_key] = values
        print()

    print("Created data")
    for not_tracked, values in all_values.items():
        frame = pd.DataFrame.from_dict(values)
        frame.sort_index(axis=1, inplace=True)
        frame.sort_index(axis=0, ascending=False, inplace=True)
        # frame = frame.reindex(sorted(frame.columns), axis=1)
        pprint.pprint(values)
        print(frame)
        sns.heatmap(frame, annot=True, vmax=85)
        plt.title(not_tracked)
        plt.xlabel(track_parmas[0])
        plt.ylabel(track_parmas[1])
        plt.show()


plot_frozen_res_avg("../Data/Frozen/train")
# plot_frozen_res("../Data/Frozen/train")
# plot_frozen_heatmap("../Data/Frozen/train")
