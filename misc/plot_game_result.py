import matplotlib.pyplot as plt
import csv
from os import listdir, makedirs
from os.path import isfile, join, isdir
import pandas as pd
import seaborn as sns
import pprint


def save_fig(name, folder):
    path_to_res = join("../result/game/", folder)

    # Save fig to disk
    if not isdir(path_to_res):
        makedirs(path_to_res)

    print(f"Saveing to {join(path_to_res, name)}")
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
    avg_win = list(map(lambda x: float(x) / (len(files)), tot_win))
    return epochs, avg_win


def parse_score_per_round(last_round, curr_round, accumulator=0):
    diff = curr_round - last_round
    last_round = curr_round

    if diff == 1:
        accumulator += 1

    print(f"last round {last_round} curr_round {curr_round} diff {diff} accumulator {accumulator}")
    return accumulator, last_round


def was_round_a_win(last_round, curr_round):
    diff = curr_round - last_round
    return diff, curr_round


def mark_epochs(won_epochs, avg_win, color):

    filtered = []

    for epoch in won_epochs:
        filtered.append(avg_win[epoch])

    plt.plot(won_epochs, filtered, 'ro', color=color, alpha=.7)


def add_random_data(dir):
    # Plot random data
    # "../Data/frozen/random"
    print("Adding controll data")
    random_runs = __group_by_run(dir)

    for run_params, files in random_runs.items():
        epochs, avg_win = calc_avg_win_per_epoch(dir, files, window=10)
        plt.plot(epochs, avg_win, label="Random Action-Policy", color="k")

        print(f"Len epoch {len(epochs)} len avg win {len(avg_win)}")

        mark_won_and_drawn_epochs(dir, files, avg_win)


def calc_won_epochs(dir, files):

    for file in files:

        with open(join(dir, file), 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            won_epochs = []

            last_round = 0

            for row in data:
                # Parse data
                epoch = int(row[0])
                curr_round = float(row[1])

                diff, last_round = was_round_a_win(last_round, curr_round)

                if diff == 1:
                    won_epochs.append(epoch)

    print(f"Won {len(won_epochs)} epochs: {won_epochs} ")

    return won_epochs


def calc_drawn_epochs(dir, files):

    for file in files:

        with open(join(dir, file), 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            drawn_epochs = []

            last_round = 0

            for row in data:
                # Parse data
                epoch = int(row[0])
                curr_round = float(row[1])

                diff, last_round = was_round_a_win(last_round, curr_round)

                if diff == 0:
                    drawn_epochs.append(epoch)

    print(f"Drawn {len(drawn_epochs)} epochs: {drawn_epochs} ")

    return drawn_epochs



def calc_avg_win_per_epoch(dir, files, window=1):
    tot_win = []
    epochs = []
    first_file = True

    for file in files:

        with open(join(dir, file), 'r') as csv_file:
            data = csv.reader(csv_file, delimiter=',')

            last_round = 0
            accumulator = 0

            for row in data:
                # Parse data
                epoch = int(row[0])

                win = float(row[2])

                # To plot acc score
                # accumulator, last_round = parse_score_per_round(last_round, float(row[1]), accumulator)
                # win = accumulator

                if first_file:
                    epochs.append(epoch)
                    tot_win.append(win)
                else:
                    tot_win[epoch] += win

        first_file = False

    # Calc avg
    avg_win = list(map(lambda x: float(x) / (len(files)), tot_win))

    print(len(avg_win))

    # Calc rolling mean
    N = window
    avg_win = pd.Series(avg_win).rolling(window=N).mean().iloc[N - 1:].values
    return epochs[:len(avg_win)], avg_win


def mark_won_and_drawn_epochs(dir, files, avg_win):
    won_epochs = calc_won_epochs(dir, files)
    mark_epochs(won_epochs, avg_win, color='g')
    drawn_epochs = calc_drawn_epochs(dir, files)
    mark_epochs(drawn_epochs, avg_win, color='y')


# Plot one plot with all runs avg
def plot_game_res(dir, name, ylabel, xlabel='Number of Epochs'):
    # Constants
    nr_to_highlight = 5
    color = ['b', 'g', 'r', 'c', 'm']

    translations = {}
    test_nr = 1

    # Read files from dir and group them
    all_runs = __group_by_run(dir)

    # Plot the results
    for run_params, files in all_runs.items():
        epochs, avg_win = calc_avg_win_per_epoch(dir, files, window=10)

        mark_won_and_drawn_epochs(dir, files, avg_win)

        label = f"DQN Agent (In-Game Score)"
        test_nr += 1
        translations[label] = run_params
        plt.plot(epochs, avg_win, label=label, color=color.pop())


    add_random_data("../Data/Game/random/train")

    plt.plot([], [], 'ro', color='g', alpha=.7, label="Round Won")
    plt.plot([], [], 'ro', color='y', alpha=.7, label="Round Drawn")

    # plt.title("All runs")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    #plt.axvline(x=67, label="End of Imitation Learning", linestyle='--')
    plt.ylim(4500, 11500)
    plt.xlim(-10, 320)
    plt.legend()
    #plt.title("67 Epochs of Prior Imitation Learning")

    save_fig(name + ".png", "")

    plt.show()


# Acc wins
# plot_game_res("../Data/game/sc2_score/train", ylabel='Accumulated wins', name='accumulated_wins')

# Sc2 score
plot_game_res("../Data/game/sc2_score/imi", ylabel='Rolling Mean of Score per Epoch',
              name='sc2_score_window_100_long_axis')
