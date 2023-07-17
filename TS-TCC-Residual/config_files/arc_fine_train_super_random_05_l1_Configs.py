import torch
import os
from sklearn.model_selection import train_test_split


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
        self.data_path = "/mnt/data/selfsupervised/iclr/arc/l1"

    def get_datasets(self, seed):
        train_dataset = torch.load(os.path.join(self.data_path, "train.pt"))
        valid_dataset = torch.load(os.path.join(self.data_path, "val.pt"))
        test_dataset = torch.load(os.path.join(self.data_path, "test.pt"))

        X_train = train_dataset["samples"].numpy()
        y_train = train_dataset["labels"].numpy()
        _, X_fine_tune, _, y_fine_tune = train_test_split(X_train, y_train, test_size=0.05, random_state=seed)

        train_dataset = dict()
        train_dataset["samples"] = torch.from_numpy(X_fine_tune)
        train_dataset["labels"] = torch.from_numpy(y_fine_tune)

        return train_dataset, valid_dataset, test_dataset
