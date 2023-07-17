import numpy as np
import pickle
import gc


def get_data(x_data_file_name, y_data_file_name, z_data_file_name, data_label_file_name, batch_size):
    with open(x_data_file_name, 'rb') as data_pickle:
        xout = pickle.load(data_pickle)
    with open(y_data_file_name, 'rb') as data_pickle:
        yout = pickle.load(data_pickle)
    with open(z_data_file_name, 'rb') as data_pickle:
        zout = pickle.load(data_pickle)

    xyz_data_shape = xout.shape[1]
    original_untrimmed = np.loadtxt(data_label_file_name, delimiter=',')
    original = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in original_untrimmed], dtype=np.double)
    original = np.concatenate((original, xout), axis=1)
    del xout
    gc.collect()
    original = np.concatenate((original, yout), axis=1)
    del yout
    gc.collect()
    original = np.concatenate((original, zout), axis=1)
    del zout
    gc.collect()
    idx_class_0 = np.where(original[:, 2] == 0)
    idx_class_1 = np.where(original[:, 2] == 1)
    len_class_0 = np.count_nonzero(original[:, 2] == 0)
    len_class_1 = np.count_nonzero(original[:, 2] == 1)
    class_0_points = original[idx_class_0]
    class_1_points = original[idx_class_1]
    np.random.shuffle(class_0_points)
    np.random.shuffle(class_1_points)

    if len_class_0 >= len_class_1:
        cr_point_list = np.concatenate((class_0_points[0:len_class_1, :], class_1_points), axis=0)
    else:
        cr_point_list = np.concatenate((class_0_points, class_1_points[0:len_class_0, :]), axis=0)

    np.random.shuffle(cr_point_list)

    data_len = cr_point_list.shape[0]
    training_len = (int(data_len * 0.7)//batch_size)*batch_size
    validation_len = (int(data_len * 0.15)//batch_size)*batch_size
    test_len = (int(data_len - training_len - validation_len)//batch_size)*batch_size

    # data = cr_point_list[:,2:]
    labels = cr_point_list[:, 2] #data[:,0]
    features = cr_point_list[:,3:].reshape((data_len, 3, xyz_data_shape))
    trainin_features = features[0:training_len,:]
    validation_features = features[training_len:training_len+validation_len,:]
    test_features = features[training_len+validation_len:training_len+validation_len+test_len,:]
    del features
    gc.collect()
    trainin_labels = labels[0:training_len]
    validation_labels = labels[training_len:training_len+validation_len]
    test_labels = labels[training_len+validation_len:training_len+validation_len+test_len]
    return trainin_features, validation_features, test_features, trainin_labels, validation_labels, test_labels


def get_data(x_data_file_name, y_data_file_name, z_data_file_name, data_label_file_name):
    with open(x_data_file_name, 'rb') as data_pickle:
        xout = pickle.load(data_pickle)
    with open(y_data_file_name, 'rb') as data_pickle:
        yout = pickle.load(data_pickle)
    with open(z_data_file_name, 'rb') as data_pickle:
        zout = pickle.load(data_pickle)
    xyz_data_shape = xout.shape[1]
    original = np.concatenate((xout, yout), axis=1)
    del xout, yout
    gc.collect()
    original = np.concatenate((original, zout), axis=1)
    del zout
    gc.collect()
    original = original.reshape((-1, 3, xyz_data_shape))
    original_untrimmed = np.loadtxt(data_label_file_name, delimiter=',')
    cr_point_list = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in original_untrimmed], dtype=np.double)
    return cr_point_list, original