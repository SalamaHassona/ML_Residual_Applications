from sklearn.model_selection import train_test_split
import torch
import os
import numpy as np
import pickle

data_dir = '/mnt/data/selfsupervised/sadecki/arc/l1'
output_dir = '/mnt/data/selfsupervised/iclr/arc/l1'

data_label_file_name = os.path.join(data_dir, "labeled_data_arc.txt")
random_state = 2

with open(os.path.join(data_dir, "x_data.pickle"), 'rb') as data_pickle:
    x_data = pickle.load(data_pickle)
with open(os.path.join(data_dir, "y_data.pickle"), 'rb') as data_pickle:
    y_data = pickle.load(data_pickle)
with open(os.path.join(data_dir, "z_data.pickle"), 'rb') as data_pickle:
    z_data = pickle.load(data_pickle)
with open(os.path.join(data_dir, "residual_x_data.pickle"), 'rb') as data_pickle:
    residual_x_data = pickle.load(data_pickle)
with open(os.path.join(data_dir, "residual_y_data.pickle"), 'rb') as data_pickle:
    residual_y_data = pickle.load(data_pickle)
with open(os.path.join(data_dir, "residual_z_data.pickle"), 'rb') as data_pickle:
    residual_z_data = pickle.load(data_pickle)

xyz_data_shape = x_data.shape[1]
train_labels_untrimmed = np.loadtxt(data_label_file_name, delimiter=',')
train_labels = np.array([0 if item[2] < 23 else 1 for item in train_labels_untrimmed], dtype=np.double)
train_data = np.dstack((x_data, y_data, z_data, residual_x_data, residual_y_data, residual_z_data))

X_train, X_val_test, y_train, y_val_test = train_test_split(train_data, train_labels, test_size=0.3, random_state=random_state)
X_val, X_test, y_val, y_test = train_test_split(X_val_test, y_val_test, test_size=0.5, random_state=random_state)

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_train)
dat_dict["labels"] = torch.from_numpy(y_train)
torch.save(dat_dict, os.path.join(output_dir, "train.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_val)
dat_dict["labels"] = torch.from_numpy(y_val)
torch.save(dat_dict, os.path.join(output_dir, "val.pt"))

dat_dict = dict()
dat_dict["samples"] = torch.from_numpy(X_test)
dat_dict["labels"] = torch.from_numpy(y_test)
torch.save(dat_dict, os.path.join(output_dir, "test.pt"))