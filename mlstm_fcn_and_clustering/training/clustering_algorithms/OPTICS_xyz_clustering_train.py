import numpy as np
import pickle
import time
from sklearn.cluster import OPTICS
import argparse


def training():
    with open(args.residual_x_var_mean, 'rb') as residual_pickle:
        residual_x = pickle.load(residual_pickle)
    with open(args.residual_y_var_mean, 'rb') as residual_pickle:
        residual_y = pickle.load(residual_pickle)
    with open(args.residual_z_var_mean, 'rb') as residual_pickle:
        residual_z = pickle.load(residual_pickle)

    residual_xyz = np.hstack((residual_x, residual_y, residual_z))
    optics = OPTICS(n_jobs=args.n_jobs).fit(residual_xyz)

    with open(args.out_labels, 'wb') as labels_pickle:
        pickle.dump(optics.labels_, labels_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    with open(args.out_model, 'wb') as model_pickle:
        pickle.dump(optics, model_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=20)
    p.add_argument("--residual_x_var_mean", type=str, default="residual_x_var_mean.pickle")
    p.add_argument("--residual_y_var_mean", type=str, default="residual_y_var_mean.pickle")
    p.add_argument("--residual_z_var_mean", type=str, default="residual_z_var_mean.pickle")
    p.add_argument("--out_labels", type=str, default="out_labels.pickle")
    p.add_argument("--out_model", type=str, default="out_model.pickle")
    args = p.parse_args()
    start = time.time()
    training()
    end = time.time()
    print(str(end-start))