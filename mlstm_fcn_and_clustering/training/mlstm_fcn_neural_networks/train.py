from model_mlstm_fcn import Network, train
from torch import optim
from torch import nn
from DynamicDataset import DynamicDataset
from load_data import get_data
import torch
import time
import argparse


def training():
    batch_size = args.batch_size
    training_features, validation_features, test_features, \
    training_labels, validation_labels, test_labels = get_data(x_data_file_name=args.x_data,
                                                               y_data_file_name=args.y_data,
                                                               z_data_file_name=args.z_data,
                                                               data_label_file_name=args.labels_data, batch_size=batch_size)
    trainset = DynamicDataset(training_features, training_labels, shuffle=True, batch_size=batch_size)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=1, shuffle=False)
    validset = DynamicDataset(validation_features, validation_labels, shuffle=True, batch_size=batch_size)
    validloader = torch.utils.data.DataLoader(validset, batch_size=1, shuffle=False)
    testset = DynamicDataset(test_features, test_labels, shuffle=True, batch_size=batch_size)
    testloader = torch.utils.data.DataLoader(testset, batch_size=1, shuffle=False)

    device = torch.device("cuda:0")
    model = Network(NumClassesOut=2, N_time=training_features.shape[2], N_Features=3)
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    model.to(device)
    try:
        train(model, trainloader, validloader, testloader, criterion, optimizer, device,
              epochs=args.epochs, start_epochs=0, save=True,
              save_file_name=args.save_file_name)
    except Exception as ex:
        print(ex)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=0.0007)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--labels_data", type=str, default="labels_data.txt")
    p.add_argument("--x_data", type=str, default="x_data.pickle")
    p.add_argument("--y_data", type=str, default="y_data.pickle")
    p.add_argument("--z_data", type=str, default="z_data.pickle")
    p.add_argument("--save_file_name", type=str, default="checkpoint")
    args = p.parse_args()
    start = time.time()
    training()
    end = time.time()
    print(str(end-start))