import matplotlib.pyplot as plt
import csv
from os import listdir, makedirs
from os.path import isfile, join, isdir
import pandas as pd
import seaborn as sns
import pprint


def save_fig(name, folder):
    path_to_res = join("../result/mtb/", folder)

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


# Returns how many epochs of imi the run used
def how_much_imi(file_name):
    return 0
    nr_chars_in_name = 23 # 18 # 23  # How many chars before parmas start
    divide_by = 8

    cut_name = file_name[nr_chars_in_name:]
    split = cut_name.split('_')
    imi = split[3]

    nr_imi = int(imi[3:])

    return int(nr_imi / divide_by)


def add_random_data(dir):
    # Plot random data
    # "../Data/frozen/random"
    random_runs = __group_by_run(dir)

    for run_params, files in random_runs.items():
        epochs, avg_win = calc_avg_win_per_epoch(dir, files, no_imi=True)
        plt.plot(epochs, avg_win, label="Random Action-Policy", color="k")


def calc_avg_win_per_epoch(dir, files, no_imi=False):
    tot_win = []
    epochs = []
    first_file = True

    max_wins_per_round = 8

    for file in files:

        if no_imi:
            skipp_epochs = 0
        else:
            skipp_epochs = how_much_imi(file)

        with open(join(dir, file), 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            for row in data:
                # Parse data
                epoch = int(row[0]) - skipp_epochs
                win = float(row[2]) / max_wins_per_round

                if epoch < 0:
                    continue

                if first_file:
                    epochs.append(epoch)
                    tot_win.append(win)
                else:
                    tot_win[epoch] += win

        first_file = False

    print(tot_win)

    # Calc avg
    avg_win = list(map(lambda x: float(x) / (len(files)), tot_win))

    #if len(avg_win) > 2000:
        #avg_win = avg_win[:2000]

    print(len(avg_win))

    #Calc rolling mean
    N = 2000
    avg_win = pd.Series(avg_win).rolling(window=N).mean().iloc[N - 1:].values
    return epochs[:len(avg_win)], avg_win


def calc_avg_win(dir, files):
    tot_win = []
    epochs = []
    first_file = True

    for file in files:

        accumulator = 0

        with open(join(dir, file), 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            for row in data:
                # Parse data
                epoch = int(row[0])

                win = float(row[2])
                accumulator += win
                win_per_round = float(row[3])

                if first_file:
                    epochs.append(epoch)
                    tot_win.append(accumulator)
                else:
                    tot_win[epoch] += accumulator

        first_file = False

    # Calc avg
    avg_win = list(map(lambda x: float(x) / (len(files)), tot_win))
    return epochs, avg_win


# Plot one plot per run
def plot_mtb_res(dir):
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
                    epsilon = 0

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

            N = 100
            curr_wins = pd.Series(curr_wins).rolling(window=N).mean().iloc[N - 1:].values
            ax_win.plot(curr_epochs[:len(curr_wins)], curr_wins, alpha=0.1)  # , label=f"run: {file[-5:-4]}")

        print(f"Len tot_win: {len(tot_win)}, len epochs: {len(epochs)}\n")

        # Calc avg
        avg_win = list(map(lambda x: float(x)/len(files), tot_win))
        avg_epsilon = list(map(lambda x: float(x)/len(files), tot_epsilon))

        # Calc rolling mean
        N = 100
        avg_win = pd.Series(avg_win).rolling(window=N).mean().iloc[N - 1:].values
        avg_epsilon = pd.Series(avg_epsilon).rolling(window=N).mean().iloc[N - 1:].values


        # Plot avg graph
        ax_win.plot(epochs[:len(avg_win)], avg_win, label="avg")
        ax_epsi.plot(epochs[:len(avg_epsilon)], avg_epsilon, label="epsilon")
        plt.title(run_params)

        # Add one common legend
        lines_win, labels_win = ax_win.get_legend_handles_labels()
        lines_epsi, labels_epsi = ax_epsi.get_legend_handles_labels()
        ax_win.legend(lines_win + lines_epsi, labels_win + labels_epsi, loc='center left')

        plt.show()


# Plot one plot with all runs avg
def plot_mtb_res_avg(dir):
    # Constants
    color = ['b', 'g', 'r', 'c', 'm']

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

        epochs, avg_win = calc_avg_win_per_epoch(dir, files)

        imi_epochs = how_much_imi(run_params)
        if imi_epochs == 0:
            label = "No Prior Imitation Learning"
        else:
            label = f"{imi_epochs} Epochs of Prior Imitation Learning"

        plt.plot(epochs, avg_win, label=label, color=color.pop())

    #add_random_data("../Data/MTB/random")

    # plt.title("All runs")
    plt.xlabel('Number of Epochs')
    plt.ylabel('Rolling Mean of Score per Epoch [ratio of max]')
    plt.legend()

    save_fig("test.png", "imi")

    plt.show()


def plot_mtb_heatmap(dir):
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

        print(f"avg_win {avg_win[-1]}")
        avg_win[-1] = avg_win[-1] / 800
        print(f"avg_win {avg_win[-1]}")

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
        sns.heatmap(frame, annot=True, vmax=0.25)

        print(f"Not tracked {not_tracked}")
        epsilon_val = ''.join([str(i) for i in not_tracked[:-5] if i.isdigit() or i == '.'])
        titel = f"Mini batch size: {epsilon_val}" #"$\\varepsilon = {epsilon_val}$"

        # Setup figure
        plt.title(titel, fontsize=16)
        plt.ylabel("$\gamma$")
        plt.xlabel("$\\varepsilon$")
        save_fig(f"valid_heatmap_{track_parmas[0]}_{track_parmas[1]}_{not_tracked}.png", "heatmap")
        plt.show()


plot_mtb_res_avg("../Data/MTB/test")
# plot_mtb_res("../Data/MTB/not_best_param/train")
# plot_mtb_heatmap("../Data/MTB/valid")
