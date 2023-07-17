import numpy as np
import pickle
import time
import matplotlib.pyplot as plt
import argparse


def plot_bif_diagram(point_array, save_file):
    rows, row_pos = np.unique(point_array[:, 0], return_inverse=True)
    cols, col_pos = np.unique(point_array[:, 1], return_inverse=True)

    pivot_table = np.zeros((len(cols), len(rows)), dtype=point_array.dtype)
    pivot_table[col_pos, row_pos] = point_array[:, 2].astype(int)
    x, y = np.meshgrid(rows, cols)
    plt.pcolormesh(x, y, pivot_table)  # now just plug the data into pcolormesh, it's that easy!
    plt.colorbar()  # need a colorbar to show the intensity scale
    plt.xlabel("Resistance (R)", fontsize=14)
    plt.ylabel("Capacitance (C)", fontsize=14)
    plt.savefig(f'{save_file}.png')
    plt.close()


def post_processing():
    original_untrimmed = np.loadtxt(args.labels_data,delimiter=',')
    cr_point_list = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in original_untrimmed], dtype=np.double)

    for i in range(2, 24):
        with open(f'{args.model_labels}_v{i}.pickle', 'rb') as predict_pickle:
            labels = pickle.load(predict_pickle)
        labels = np.expand_dims(labels, axis=1)
        predict = np.concatenate((cr_point_list[:,0:2], labels), axis=1)
        plot_bif_diagram(predict, f'{args.out_fig}_v{i}')


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_data", type=str, default="labels_data.txt")
    p.add_argument("--model_labels", type=str, default="model_labels")
    p.add_argument("--out_fig", type=str, default="out_fig")
    args = p.parse_args()
    start = time.time()
    post_processing()
    end = time.time()
    print(str(end-start))