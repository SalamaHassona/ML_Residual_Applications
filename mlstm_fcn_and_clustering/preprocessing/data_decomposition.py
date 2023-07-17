import numpy as np
import pickle
from seasonal import fit_seasons, adjust_seasons
import time
import argparse


def standardization(x):
    x_mean = np.mean(x, axis=1)
    x_std = np.std(x, axis=1)
    x = x-x_mean[..., np.newaxis]
    x = x/ x_std[..., np.newaxis]
    return x


def residual_standardization(x):
    x_mean = np.mean(x)
    x_std = np.std(x)
    x = x-x_mean
    x = x/ x_std
    return x


def main():
    with open(args.data, 'rb') as data_pickle:
        data = pickle.load(data_pickle)
    data = standardization(data)
    residual_x = np.zeros((data.shape[0], data.shape[1]))
    for i in range(0, data.shape[0]):
        seasons, trend = fit_seasons(data[i, :])
        if seasons is None:
            residual = data[i, :] - trend
            residual_x[i, :] = residual
        else:
            residual_x[i, :] = adjust_seasons(data[i, :], seasons=seasons) - trend

    with open(args.out, 'wb') as residual_x_pickle:
        pickle.dump(residual_x, residual_x_pickle, protocol=pickle.HIGHEST_PROTOCOL)
    res_mean = residual_standardization(np.mean(residual_x, axis=1))
    res_var = residual_standardization(np.var(residual_x, axis=1))
    residual_x_var_mean = np.concatenate((np.expand_dims(res_var, axis=1),np.expand_dims(res_mean, axis=1)), axis=1)
    with open(args.out_var_mean, 'wb') as residual_var_mean_pickle:
        pickle.dump(residual_x_var_mean, residual_var_mean_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", type=str, default="x_data.pickle")
    p.add_argument("--out", type=str, default="residual_x.pickle")
    p.add_argument("--out_var_mean", type=str, default="residual_x_var_mean.pickle")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))