import matplotlib.pyplot as plt
import csv
from os import listdir, makedirs
from os.path import isfile, join, isdir
import pandas as pd
import seaborn as sns
import pprint


def save_fig(name, folder):
    path_to_res = join("../result/frozen/", folder)

    # Save fig to disk
    if not isdir(path_to_res):
        makedirs(path_to_res)
    plt.savefig(join(path_to_res, name))


def use_tex():
    # Set up for use of tex in plt
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')


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
def plot_frozen_imi_res(dir):
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
            ax_win.plot(curr_epochs, curr_wins, alpha=0.2) # , label=f"run: {file[-5:-4]}")

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


# Returns how many epochs of imi the run used
def how_much_imi(file_name):
    nr_chars_in_name = 18  # How many chars before parmas start
    divide_by = 6

    cut_name = file_name[nr_chars_in_name:]
    split = cut_name.split('_')
    imi = split[3]

    nr_imi = int(imi[3:])

    return int(nr_imi / divide_by)


def calc_avg_win(dir, files, no_imi=False):
    tot_win = []
    epochs = []
    first_file = True

    for file in files:

        if no_imi:
            skipp_epochs = 0
        else:
            skipp_epochs = how_much_imi(file)

        score_when_skipped = 0

        with open(join(dir, file), 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            for row in data:
                # Parse data
                epoch = int(row[0]) - skipp_epochs
                win = int(row[2])

                if epoch <= 0:
                    score_when_skipped = win
                    continue

                # Compensate for skipped score
                win = win - score_when_skipped

                if first_file:
                    epochs.append(epoch)
                    tot_win.append(win)
                else:
                    tot_win[epoch - 1] += win

        first_file = False

    # Padd with wins
    while len(tot_win) != 200:
        tot_win.append(tot_win[-1] + len(files))
        epochs.append(epochs[-1] + 1)

    # Calc avg
    avg_win = list(map(lambda x: float(x) / (len(files)), tot_win))
    return epochs, avg_win


def add_random_data(dir):
    # Plot random data
    random_runs = __group_by_run(dir)

    for run_params, files in random_runs.items():
        epochs, avg_win = calc_avg_win(dir, files, no_imi=True)
        plt.plot(epochs, avg_win, label="Random Action-Policy", color="k")


# Plot one plot with all runs avg
def plot_frozen_imi_res_avg(dir):
    # Constants
    color = ['#ff9d00', 'c', 'm', 'g', '#ff0000']

    # Read files from dir and group them
    all_runs = __group_by_run(dir)

    ordered_runs = []

    # Order by run
    for run_params, files in all_runs.items():
        ordered_runs.append([run_params, files, how_much_imi(run_params)])

    ordered_runs.sort(key=lambda x: x[2])

    # Plot the results
    for params in ordered_runs:
        run_params = params[0]
        files = params[1]

        print(run_params)

        epochs, avg_win = calc_avg_win(dir, files)

        imi_epochs = how_much_imi(run_params)
        if imi_epochs == 0:
            label = "No Prior Imitation Learning"
        else:
            label = f"{imi_epochs} Epochs of Prior Imitation Learning"

        plt.plot(epochs, avg_win, label=label, color=color.pop())

    add_random_data("../Data/frozen/random")

    # plt.title("All runs")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Accumulated Wins')
    plt.legend()

    save_fig("imi_best_param.png", "imi")

    plt.show()


def plot_frozen_imi_heatmap(dir):
    use_tex()

    # Constants
    track_parmas = ['Et', 'G']  # First x, Second y
    track_as_float = ['G', 'Et']
    nr_chars_in_name = 18  # How many chars before parmas start

    all_runs = __group_by_run(dir)

    all_values = {}

    # Read and group data
    for run_params, files in all_runs.items():
        run_params = run_params[nr_chars_in_name:]

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

        not_tracked_key = "_".join(not_tracked)

        _, avg_win = calc_avg_win(dir, files)

        if not_tracked_key in all_values:
            values = all_values[not_tracked_key]

            if axis_values[0] in values:
                values[axis_values[0]][axis_values[1]] = avg_win[-1]
            else:
                values[axis_values[0]] = {axis_values[1]: avg_win[-1]}
        else:
            values = {axis_values[0]: {axis_values[-1]: avg_win[-1]}}

        all_values[not_tracked_key] = values

    print("Created data")
    for not_tracked, values in all_values.items():
        frame = pd.DataFrame.from_dict(values)

        # Sort fame to like
        frame.sort_index(axis=1, inplace=True)
        frame.sort_index(axis=0, ascending=False, inplace=True)
        sns.heatmap(frame, annot=True, vmax=10)

        print(f"Not tracked {not_tracked}")
        epsilon_val = ''.join([str(i) for i in not_tracked[:-5] if i.isdigit() or i == '.'])
        titel = f"Mini batch size: {epsilon_val}" #"$\\varepsilon = {epsilon_val}$"

        # Setup figure
        plt.title(titel, fontsize=16)
        plt.ylabel("$\gamma$")
        plt.xlabel("$\\varepsilon$")
        save_fig(f"valid_heatmap_{track_parmas[0]}_{track_parmas[1]}_{not_tracked}.png", "heatmap")
        plt.show()


plot_frozen_imi_res_avg("../Data/Frozen/best_params_imi")
# plot_frozen_imi_res("../Data/Frozen/train")
# plot_frozen_imi_heatmap("../Data/Frozen/valid")
