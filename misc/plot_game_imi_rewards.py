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

    # print(f"last round {last_round} curr_round {curr_round} diff {diff} accumulator {accumulator}")
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

            skipp_epochs = 100 #how_much_imi(file)
            score_when_skipped = 0

            for row in data:
                # Parse data
                epoch = int(row[0]) - skipp_epochs

                #win = float(row[2])

                # To plot acc score
                accumulator, last_round = parse_score_per_round(last_round, float(row[1]), accumulator)
                win = accumulator - score_when_skipped

                if epoch < 0:
                    score_when_skipped = accumulator
                    continue

                if first_file:
                    epochs.append(epoch)
                    tot_win.append(win)
                else:
                    tot_win[epoch] += win

        first_file = False

    # Calc avg
    avg_win = list(map(lambda x: float(x) / (len(files)), tot_win))

    # Calc rolling mean
    N = 1
    avg_win = pd.Series(avg_win).rolling(window=N).mean().iloc[N - 1:].values
    return epochs[:len(avg_win)], avg_win


# Returns how many epochs of imi the run used
def how_much_imi(file_name):
    nr_chars_in_name = 23 # 18 # 23  # How many chars before parmas start
    divide_by = 8

    cut_name = file_name[nr_chars_in_name:]
    split = cut_name.split('_')
    imi = split[3]

    nr_imi = int(imi[3:])

    return int(nr_imi / divide_by)


def manual_2_color_line():
    colors = ['k', 'c']
    x_val = []
    for i in range(0, 210, 10):
        x_val.append(i)

    for i, _ in enumerate(x_val):
        if i > len(x_val) -2:
            continue
        plt.plot([x_val[i], x_val[i+1]], [0, 0], color=colors[i % 2])


def mark_won_and_drawn_epochs(dir, files, avg_win):
    won_epochs = calc_won_epochs(dir, files)
    mark_epochs(won_epochs, avg_win, color='g')
    drawn_epochs = calc_drawn_epochs(dir, files)
    mark_epochs(drawn_epochs, avg_win, color='y')


# Plot one plot with all runs avg
def plot_game_res(dir, name, ylabel, xlabel='Number of Epochs'):
    # Constants
    nr_to_highlight = 5
    color = ['b', 'g', 'c', 'r', 'm']

    translations = {}
    test_nr = 1

    # Read files from dir and group them
    all_runs = __group_by_run(dir)

    #add_random_data("../Data/Game/random/train")

    manual_2_color_line()

    # Plot the results
    for run_params, files in all_runs.items():
        epochs, avg_win = calc_avg_win_per_epoch(dir, files, window=10)

        # mark_won_and_drawn_epochs(dir, files, avg_win)

        if test_nr == 1:
            label = f"In-Game Score +2000 on Win"
        if test_nr == 2:
            label = f"In-Game Score"

        print(f"Agent {test_nr}, file: {run_params}")



        test_nr += 1
        translations[label] = run_params
        plt.plot(epochs, avg_win, label=label, color=color.pop())

    if False:
        # Plot the results
        for run_params, files in all_runs.items():
            epochs, avg_win = calc_avg_win_per_epoch(dir, files, window=100)

            won_epochs = calc_won_epochs(dir, files)
            mark_epochs(won_epochs, avg_win)

            label = f"Agent {test_nr} 100"
            test_nr += 1
            translations[label] = run_params
            plt.plot(epochs, avg_win, label=label, color=color.pop())

    plt.plot([], [], label="Sparse Reward (1/0)", color='c')
    plt.plot([], [], label="Random Action-Policy", color='k')

    # plt.title("All runs")
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)

    # plt.axvline(x=67, label="End of imitation learning", linestyle='--')
    plt.legend()
    # plt.figlegend()
    save_fig(name + ".png", "imi")

    plt.show()


# Acc wins
# plot_game_res("../Data/game/sc2_score/train", ylabel='Accumulated wins', name='accumulated_wins')

# Sc2 score
plot_game_res("../Data/game/rewards/imi", ylabel='Accumulated Wins',
              name='rewards_acc_score_imi')
