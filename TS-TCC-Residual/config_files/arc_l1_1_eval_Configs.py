import torch
import os
import numpy as np
import pickle


class Config(object):
    def __init__(self):
        # model configs
        self.input_channels = 3
        self.kernel_size = 16
        self.stride = 1
        self.final_out_channels = 128
        self.model_output_dim = 315

        self.num_classes = 2
        self.dropout = 0.35
        self.features_len = 7503

        # training configs
        self.num_epoch = 100

        
        # optimizer parameters
        self.beta1 = 0.9
        self.beta2 = 0.99
        self.lr = 3e-4

        # data parameters
        self.drop_last = True
        self.batch_size = 128


        self.Context_Cont = Context_Cont_configs()
        self.TC = TC()
        self.augmentation = augmentations()
        self.data = Data()


class augmentations(object):
    def __init__(self):
        self.jitter_scale_ratio = 0.001
        self.jitter_ratio = 0.001
        self.max_seg = 5


class Context_Cont_configs(object):
    def __init__(self):
        self.temperature = 0.2
        self.use_cosine_similarity = True


class TC(object):
    def __init__(self):
        self.hidden_dim = 100
        self.timesteps = 10


class Data(object):
    def __init__(self):
        self.data_path = "/mnt/data/selfsupervised/sadecki/arc/l1_1"
        self.data_label_file_name = os.path.join(self.data_path, "labeled_data_arc.txt")

    def get_data(self):
        with open(os.path.join(self.data_path, "x_data.pickle"), 'rb') as data_pickle:
            x_data = pickle.load(data_pickle)
        with open(os.path.join(self.data_path, "y_data.pickle"), 'rb') as data_pickle:
            y_data = pickle.load(data_pickle)
        with open(os.path.join(self.data_path, "z_data.pickle"), 'rb') as data_pickle:
            z_data = pickle.load(data_pickle)

        eval_labels_untrimmed = np.loadtxt(self.data_label_file_name, delimiter=',')
        eval_labels = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in eval_labels_untrimmed], dtype=np.double)
        eval_data = np.dstack((x_data, y_data, z_data))

        return torch.from_numpy(eval_data), torch.from_numpy(eval_labels)
