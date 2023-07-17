import torch
from torch import nn
import torch.nn.functional as F
import time
from torch.utils.tensorboard import SummaryWriter

class SELayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class Network(nn.Module):

    def __init__(self, NumClassesOut, N_time, N_Features, N_LSTM_Out=128, N_LSTM_layers=1,
                 Conv1_NF=128, Conv2_NF=256, Conv3_NF=128, lstmDropP = 0.8, FC_DropP = 0.3):
        super().__init__()

        self.N_time = N_time
        self.N_Features = N_Features
        self.NumClassesOut = NumClassesOut
        self.N_LSTM_Out = N_LSTM_Out
        self.N_LSTM_layers = N_LSTM_layers
        self.Conv1_NF = Conv1_NF
        self.Conv2_NF = Conv2_NF
        self.Conv3_NF = Conv3_NF
        self.lstm = nn.LSTM(N_Features, N_LSTM_Out, N_LSTM_layers)
        self.C1 = nn.Conv1d(N_Features, Conv1_NF, 8)
        self.C2 = nn.Conv1d(Conv1_NF, Conv2_NF, 5)
        self.C3 = nn.Conv1d(Conv2_NF, Conv3_NF, 3)
        self.BN1 = nn.BatchNorm1d(Conv1_NF)
        self.BN2 = nn.BatchNorm1d(Conv2_NF)
        self.BN3 = nn.BatchNorm1d(Conv3_NF)
        self.se1 = SELayer(self.Conv1_NF)
        self.se2 = SELayer(self.Conv2_NF)
        self.relu = nn.ReLU()
        self.lstmDrop = nn.Dropout(lstmDropP)
        self.ConvDrop = nn.Dropout(FC_DropP)
        self.FC = nn.Linear(Conv3_NF + N_LSTM_Out, NumClassesOut)

    def forward(self, x, device):
        h0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(device)
        c0 = torch.zeros(self.N_LSTM_layers, self.N_time, self.N_LSTM_Out).to(device)
        x = x.reshape((list(x.size())[0], list(x.size())[2], list(x.size())[1]))

        x1, (ht, ct) = self.lstm(x, (h0, c0))
        x1 = x1[:, -1, :]

        x2 = x.transpose(2, 1)
        x2 = self.ConvDrop(self.relu(self.BN1(self.C1(x2))))
        x2 = self.se1(x2)
        x2 = self.ConvDrop(self.relu(self.BN2(self.C2(x2))))
        x2 = self.se2(x2)
        x2 = self.ConvDrop(self.relu(self.BN3(self.C3(x2))))
        x2 = torch.mean(x2, 2)

        x_all = torch.cat((x1, x2), dim=1)
        x_out = self.FC(x_all)
        x_out_2 = F.log_softmax(x_out, dim=1)
        return x_out_2


def validation(model, loader, criterion, device):
    accuracy = 0
    loss = 0
    start = time.time()
    for features, labels in loader:
        features, labels = features.squeeze().to(device), labels.squeeze().to(device)
        output = model.forward(features, device)
        loss += criterion(output, labels).item()

        ps = torch.exp(output)
        equality = (labels == ps.max(1)[1])
        accuracy += equality.type_as(torch.FloatTensor()).mean().item()
    end = time.time()
    return loss, accuracy, end-start



def train(model, train_loader, validation_loader, test_loader, criterion, optimizer, device, epochs=5, start_epochs=0,
          save=False, save_file_name=None):
    writer = SummaryWriter()

    for epoch in range(start_epochs, epochs):
        training_loss = 0
        training_accuracy = 0
        # Model in training mode, dropout is on
        start = time.time()
        model.train()
        for features, labels in train_loader:
            features, labels = features.squeeze().to(device), labels.squeeze().to(device)
            optimizer.zero_grad()

            output = model.forward(features, device)
            loss = criterion(output, labels)
            loss.backward()
            optimizer.step()

            training_loss += loss.item()

            ps = torch.exp(output)
            equality = (labels == ps.max(1)[1])
            training_accuracy += equality.type_as(torch.FloatTensor()).mean().item()
        model.eval()
        end = time.time()
        writer.add_scalar("Loss: train", training_loss / len(train_loader), epoch)
        writer.add_scalar("Accuracy: train", training_accuracy / len(train_loader), epoch)
        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Training Loss: {:.3f} ".format(training_loss / len(train_loader)),
              "Training Accuracy: {:.3f} ".format(training_accuracy / len(train_loader)),
              "Training Time: {time}".format(time=str(end-start)))

        with torch.no_grad():
            valid_loss, valid_accuracy, valid_time = validation(model, validation_loader, criterion, device)
        writer.add_scalar("Loss: validation", valid_loss / len(validation_loader), epoch)
        writer.add_scalar("Accuracy: validation", valid_accuracy / len(validation_loader), epoch)
        print("Epoch: {}/{}.. ".format(epoch + 1, epochs),
              "Validation Loss: {:.3f} ".format(valid_loss / len(validation_loader)),
              "Validation Accuracy: {:.3f}".format(valid_accuracy / len(validation_loader)),
              "Validation Time: {time}".format(time=str(valid_time)))

        if save:
            torch.save({'epoch':epoch,
                        'model':model.state_dict(),
                        'optimizer':optimizer.state_dict()},
                       '{name}_trainloss_{trainloss}_trainaccuracy_{trainaccuracy}_validloss_{validloss}_validaccuracy_{validaccuracy}_epoch_{epoch}.pth'
                       .format(name=save_file_name,
                               trainloss=str(round(training_loss / len(train_loader), 3)),
                               trainaccuracy=str(round(training_accuracy / len(train_loader), 3)),
                               validloss=str(round(valid_loss / len(validation_loader), 3)),
                               validaccuracy=str(round(valid_accuracy / len(validation_loader), 3)),
                               epoch=str(epoch)))
            with open("{save_file_name}.txt".format(save_file_name=save_file_name), "a") as file_object:
                file_object.write("{0:.6f},".format(training_loss / len(train_loader))+
                                  "{0:.6f},".format(training_accuracy / len(train_loader))+
                                  "{0:.6f},".format(valid_loss / len(validation_loader)) +
                                  "{0:.6f},".format(valid_accuracy / len(validation_loader)) +
                                  "{epoch}\n".format(epoch=epoch))
        model.train()
    model.eval()
    with torch.no_grad():
        test_loss, test_accuracy, test_time = validation(model, test_loader, criterion, device)
    writer.add_scalar("Loss: test", test_loss / len(test_loader))
    writer.add_scalar("Accuracy: test", test_accuracy / len(test_loader))
    writer.flush()
    writer.close()
    print("Test Loss: {:.3f} ".format(test_loss / len(test_loader)),
          "Test Accuracy: {:.3f} ".format(test_accuracy / len(test_loader)),
          "Test Time: {time}".format(time=str(test_time)))