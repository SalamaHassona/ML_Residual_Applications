import numpy as np
import torch
import pickle
import time
import argparse


def generate_x_data(R, C, x0, y0, z0, data_range, L, dt, kd, kdmin):
    zp = torch.from_numpy(np.ones((data_range, kd-kdmin), dtype=np.double)).double().cpu()
    zp[:, 0] = z0
    xp = torch.from_numpy(np.ones(data_range, dtype=np.double)).double().cpu() * x0
    yp = torch.from_numpy(np.ones(data_range, dtype=np.double)).double().cpu() * y0
    for k in range(kdmin, kd - 1):
        xx = xp
        yy = yp
        zz = zp[:, k - kdmin]
        kx1 = (yy - xx * pow(zz, -2 / 3)) / L
        ky1 = (R + 1 - yy - R * xx) / (R * C)
        kz1 = xx * xx - zz
        x1 = xx + (dt * kx1) / 2
        y1 = yy + (dt * ky1) / 2
        z1 = zz + (dt * kz1) / 2
        kx2 = (y1 - x1 * pow(z1, -2 / 3)) / L
        ky2 = (R + 1 - y1 - R * x1) / (R * C)
        kz2 = x1 * x1 - z1
        del x1, y1, z1
        x2 = xx + (dt * kx2) / 2
        y2 = yy + (dt * ky2) / 2
        z2 = zz + (dt * kz2) / 2
        kx3 = (y2 - x2 * pow(z2, -2 / 3)) / L
        ky3 = (R + 1 - y2 - R * x2) / (R * C)
        kz3 = x2 * x2 - z2
        del x2, y2, z2
        x3 = xx + (dt * kx3)
        y3 = yy + (dt * ky3)
        z3 = zz + (dt * kz3)
        kx4 = (y3 - x3 * pow(z3, -2 / 3)) / L
        ky4 = (R + 1 - y3 - R * x3) / (R * C)
        kz4 = x3 * x3 - z3
        del x3, y3, z3
        xp = xx + ((kx1 + 2 * kx2 + 2 * kx3 + kx4) / 6) * dt
        yp = yy + ((ky1 + 2 * ky2 + 2 * ky3 + ky4) / 6) * dt
        zp[:, k - kdmin+1] = zz + ((kz1 + 2 * kz2 + 2 * kz3 + kz4) / 6) * dt
        del kx1, kx2, kx3, ky1, ky2, ky3, kz1, kz2, kz3
    del xp, yp
    zp = zp.numpy()[:, ::8]
    return zp


def main():
    xyz_points_untrimmed = np.loadtxt(args.X_0_points, delimiter=',')
    xyz_points = np.array([[item[0], item[1], item[2]] for item in xyz_points_untrimmed], dtype=np.double)
    original_untrimmed = np.loadtxt(args.labels_data, delimiter=',')
    original = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in original_untrimmed], dtype=np.double)
    cr_point_list = np.concatenate((original, xyz_points), axis=1)

    data_len = cr_point_list.shape[0]
    R = torch.from_numpy(cr_point_list[0:data_len, 0]).double().cpu()
    C = torch.from_numpy(cr_point_list[0:data_len, 1]).double().cpu()
    x0 = torch.from_numpy(cr_point_list[0:data_len, 3]).double().cpu()
    y0 = torch.from_numpy(cr_point_list[0:data_len, 4]).double().cpu()
    z0 = torch.from_numpy(cr_point_list[0:data_len, 5]).double().cpu()

    data = generate_x_data(R, C, x0, y0, z0, data_len, args.L,
                                  args.dt, int(args.tmax/args.dt)+1, int(args.tmin/args.dt))

    with open(args.out, 'wb') as data_pickle:
        pickle.dump(data, data_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--L", type=float, default=1.00)
    p.add_argument("--dt", type=float, default=0.01)
    p.add_argument("--tmin", type=int, default=1800)
    p.add_argument("--tmax", type=int, default=2000)
    p.add_argument("--labels_data", type=str, default="labels_data.txt")
    p.add_argument("--X_0_points", type=str, default="X_0_points.txt")
    p.add_argument("--out", type=str, default="z_data.pickle")
    args = p.parse_args()
    start = time.time()
    main()
    end = time.time()
    print(str(end-start))