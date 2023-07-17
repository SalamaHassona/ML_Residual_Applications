import numpy as np
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import auc
import argparse
import time


def plot_bif_diagram(point_array):
    rows, row_pos = np.unique(point_array[:, 0], return_inverse=True)
    cols, col_pos = np.unique(point_array[:, 1], return_inverse=True)

    pivot_table = np.zeros((len(cols), len(rows)), dtype=point_array.dtype)
    pivot_table[col_pos, row_pos] = point_array[:, 2].astype(int)
    x, y = np.meshgrid(rows, cols)
    plt.pcolormesh(x, y, pivot_table)  # now just plug the data into pcolormesh, it's that easy!
    plt.colorbar()  # need a colorbar to show the intensity scale
    # plt.savefig('net_image.png')
    plt.xlabel("Resistance (R)", fontsize=14)
    plt.ylabel("Capacitance (C)", fontsize=14)
    plt.show()


def evaluate():
    original_untrimmed = np.loadtxt(args.labels_data, delimiter=',')
    plot_bif_diagram(original_untrimmed)
    cr_point_list = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in original_untrimmed],
                             dtype=np.double)
    R = cr_point_list[0:180000, 0]
    C = cr_point_list[0:180000, 1]
    with open(args.predicted_prob_labels, 'rb') as predict:
        data = pickle.load(predict)
    gen_point = np.array([[R[i], C[i], data[i][0], data[i][1]] for i in range(0, 180000)])
    gen_point_list = np.array([[R[i], C[i], 0 if data[i][4] < .5 else 1] for i in range(0, 180000)])

    original = cr_point_list
    net_v2 = gen_point_list

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
        new_array.append([original[i][0], original[i][1], original[i][2] - net_v2[i][2]])
    print(true_periodic)
    print(true_chaotic)
    print(false_periodic)
    print(false_chaotic)
    accuracy = (true_periodic + true_chaotic) / (true_periodic + true_chaotic + false_periodic + false_chaotic)
    precision = (true_chaotic) / (true_chaotic + false_chaotic)
    recall = (true_chaotic) / (true_chaotic + false_periodic)
    f1score = 2 * precision * recall / (precision + recall)
    print('Accuracy = %.4f' % accuracy)
    print('Precision = %.4f' % precision)
    print('Recall = %.4f' % recall)
    print('F1score = %.4f' % f1score)

    plot_bif_diagram(net_v2)

    new_array = list()
    for i in range(len(original)):
        new_array.append([original[i][0], original[i][1], original[i][2] - net_v2[i, 2]])

    point_array = np.array(new_array)
    rows, row_pos = np.unique(point_array[:, 0], return_inverse=True)
    cols, col_pos = np.unique(point_array[:, 1], return_inverse=True)

    pivot_table = np.zeros((len(cols), len(rows)), dtype=point_array.dtype)
    pivot_table[col_pos, row_pos] = point_array[:, 2]
    x, y = np.meshgrid(rows, cols)
    plt.pcolormesh(x, y, pivot_table)  # now just plug the data into pcolormesh, it's that easy!
    plt.colorbar()  # need a colorbar to show the intensity scale
    plt.show()  # boom

    y_true = original[:, 2]
    y_probas = gen_point[:, 3]
    yn_probas = gen_point[:, 2]
    lr_fpr, lr_tpr, _ = roc_curve(y_true, y_probas)
    lr_auc = roc_auc_score(y_true, y_probas)
    print('ROC AUC=%.4f' % (lr_auc))
    plt.plot([0, 1], [0, 1], linestyle='--', label="No Skill")
    plt.plot(lr_fpr, lr_tpr, marker='.', label='MLSTM', color="b")
    plt.show()

    precision, recall, _ = precision_recall_curve(y_true, y_probas)
    auc_score = auc(recall, precision)
    print('PR AUC: %.4f' % auc_score)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--labels_data", type=str, default="labels_data.txt")
    p.add_argument("--predicted_prob_labels", type=str, default="predicted_prob_labels.pickle")
    args = p.parse_args()
    start = time.time()
    evaluate()
    end = time.time()
    print(str(end-start))