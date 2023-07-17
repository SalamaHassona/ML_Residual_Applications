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


def post_processing_classification():
    np.random.seed(0)
    original_untrimmed = np.loadtxt(args.labels_data, delimiter=',')
    cr_point_list = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in original_untrimmed], dtype=np.double)

    with open(args.model_labels, 'rb') as predict_pickle:
        labels = pickle.load(predict_pickle)
    new_labels = np.copy(labels)
    for i in range(0, 24):
        indexes = np.array(np.where(labels == i))[0]
        if indexes.size > 0:
            choice = np.random.choice(indexes, 5)
            cr_points_mean = np.mean(cr_point_list[choice][:,2])
            if cr_points_mean > 0.5:
                new_labels[indexes] = 1
            else:
                new_labels[indexes] = 0

    new_labels = np.expand_dims(new_labels, axis=1)
    predict = np.concatenate((cr_point_list[:, 0:2], new_labels), axis=1)
    plot_bif_diagram(predict, args.out_fig)

    original= cr_point_list
    net_v2 = predict

    correct = 0
    true_chaotic = 0
    false_chaotic = 0
    true_periodic = 0
    false_periodic = 0
    new_array = list()
    for i in range(len(original)):
        if original[i][2] == 0 and net_v2[i][2] == 0:
            true_periodic += 1
        if original[i][2] == 1 and net_v2[i][2] == 1:
            true_chaotic += 1
        if original[i][2] == 1 and net_v2[i][2] == 0:
            false_periodic += 1
        if original[i][2] == 0 and net_v2[i][2] == 1:
            false_chaotic += 1
        new_array.append([original[i][0],original[i][1], original[i][2]-net_v2[i][2]])
    print(true_periodic)
    print(true_chaotic)
    print(false_periodic)
    print(false_chaotic)
    accuracy = (true_periodic+true_chaotic)/(true_periodic+true_chaotic+false_periodic+false_chaotic)
    precision = (true_chaotic)/(true_chaotic+false_chaotic)
    recall = (true_chaotic)/(true_chaotic+false_periodic)
    f1score = 2*precision*recall/(precision+recall)
    print('Accuracy = %.4f' % accuracy)
    print('Precision = %.4f' % precision)
    print('Recall = %.4f' % recall)
    print('F1score = %.4f' % f1score)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_data", type=str, default="labels_data.txt")
    p.add_argument("--model_labels", type=str, default="model_labels")
    p.add_argument("--out_fig", type=str, default="out_fig")
    args = p.parse_args()
    start = time.time()
    post_processing_classification()
    end = time.time()
    print(str(end-start))