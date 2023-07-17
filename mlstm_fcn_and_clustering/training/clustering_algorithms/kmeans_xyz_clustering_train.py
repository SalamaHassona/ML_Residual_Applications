import numpy as np
import pickle
import time
from sklearn.cluster import KMeans
import argparse


def training():
    with open(args.residual_x_var_mean, 'rb') as residual_pickle:
        residual_x = pickle.load(residual_pickle)
    with open(args.residual_y_var_mean, 'rb') as residual_pickle:
        residual_y = pickle.load(residual_pickle)
    with open(args.residual_z_var_mean, 'rb') as residual_pickle:
        residual_z = pickle.load(residual_pickle)

    residual_xyz = np.hstack((residual_x, residual_y, residual_z))

    for i in range(2, args.n_clusters+1):
        kmeans = KMeans(n_clusters=args.n_clusters,
                        random_state=args.random_state,
                        max_iter=args.max_iter).fit(residual_xyz)
        with open(f'{args.out_labels}_v{i}.pickle', 'wb') as labels_pickle:
            pickle.dump(kmeans.labels_, labels_pickle, protocol=pickle.HIGHEST_PROTOCOL)
        with open(f'{args.out_model}_v{i}.pickle', 'wb') as model_pickle:
            pickle.dump(kmeans, model_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--n_clusters", type=int, default=23)
    p.add_argument("--max_iter", type=int, default=1000)
    p.add_argument("--random_state", type=int, default=0)
    p.add_argument("--residual_x_var_mean", type=str, default="residual_x_var_mean.pickle")
    p.add_argument("--residual_y_var_mean", type=str, default="residual_y_var_mean.pickle")
    p.add_argument("--residual_z_var_mean", type=str, default="residual_z_var_mean.pickle")
    p.add_argument("--out_labels", type=str, default="out_labels")
    p.add_argument("--out_model", type=str, default="out_model")
    args = p.parse_args()
    start = time.time()
    training()
    end = time.time()
    print(str(end-start))