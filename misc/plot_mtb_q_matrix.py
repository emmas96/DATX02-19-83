import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import csv
from os import listdir, makedirs
from os.path import isfile, join, isdir
import pandas as pd
import seaborn as sns
import pprint

Q = 'Q'
state = 'state'


def save_fig(name, folder):
    path_to_res = join("../result/mtb/", folder)

    # Save fig to disk
    if not isdir(path_to_res):
        makedirs(path_to_res)
    plt.savefig(join(path_to_res, name))


def group_by_run(dir):
    files = [f for f in listdir(dir) if isfile(join(dir, f))]

    container = {}

    for f in files:
        run_params = f.split('_')
        type = str(run_params[1])
        id = '_'.join(run_params[2:])

        if id in container:
            container[id][type] = f
        else:
            container[id] = {type: f}

    return container


def parse_data(dir, file):
    with open(join(dir, file), 'r') as csv_file:
        data = csv.reader(csv_file, delimiter=' ')

        matrix = []

        for row in data:
            parsed = list(map(lambda x: float(x), row))
            matrix.append(parsed)

        df = pd.DataFrame(matrix)
        df.sort_index(axis=1, inplace=True)
        df.sort_index(axis=0, ascending=True, inplace=True)

    return df, matrix


def get_xy_coordinates(matrix):
    highlighted = []

    for y, row in enumerate(matrix):
        for x, value in enumerate(row):
            if value == 1:
                highlighted.append((x, y))

    return highlighted


def mark_cords(ax, cords):
    for (x, y) in cords:
        ax.add_patch(Rectangle((x, y), 1, 1, fill=False, edgecolor='green', lw=1))



def plot_q_matrix(dir):

    all_runs = group_by_run(dir)

    for id, files in all_runs.items():
        q_df, _ = parse_data(dir, files[Q])
        state_df, state_matrix = parse_data(dir, files[state])

        ax = sns.heatmap(q_df, xticklabels=2, yticklabels=2)
        beacon_cords = get_xy_coordinates(state_matrix)

        mark_cords(ax, beacon_cords)

        save_fig(id[:-4] + ".png", "q_matrix")
        plt.show()





plot_q_matrix('../Data/mtb/state/test')
