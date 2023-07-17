import torch
import numpy as np
from training.mlstm_fcn_neural_networks.model_mlstm_fcn import Network
import pickle
import time
import gc
from training.mlstm_fcn_neural_networks.load_data import get_data
import argparse


def predict_prob():
    loop_range = args.loop_range
    data_range = args.data_range

    device_0 = torch.device("cuda:0")

    def standardization(_x):
        x = (_x - _x.mean(dim=2).view(_x.shape[0], _x.shape[1], 1)) / _x.std(dim=2).view(_x.shape[0], _x.shape[1], 1)
        return x

    model = Network(NumClassesOut=2, N_time=2501, N_Features=3)
    model.to(device_0)
    state_dict = torch.load(
        args.checkpoint,
        map_location=device_0)
    model.load_state_dict(state_dict['model'])
    model.eval()
    ps_list = np.array([], dtype=np.int64).reshape(0, 5)

    cr_period_list, data = get_data(x_data_file_name=args.x_data,
                                    y_data_file_name=args.y_data,
                                    z_data_file_name=args.z_data,
                                    data_label_file_name=args.labels_data)

    for i in range(0, loop_range):
        x = standardization(torch.from_numpy(data[i * data_range:(i + 1) * data_range, :])
                            .type(torch.FloatTensor)).squeeze().to(device_0)
        output = model.forward(x, device_0)
        ps = torch.exp(output).double().detach().cpu().numpy()
        ps_list = np.vstack(
            (ps_list, np.hstack((cr_period_list[i * data_range:(i + 1) * data_range, 0].reshape((data_range, 1)),
                                 cr_period_list[i * data_range:(i + 1) * data_range, 1].reshape((data_range, 1)),
                                 cr_period_list[i * data_range:(i + 1) * data_range, 2].reshape((data_range, 1)),
                                 ps))))
        print("i: " + str(i))
        del ps, output, x
        gc.collect()
        with torch.cuda.device('cuda:0'):
            torch.cuda.empty_cache()

    with open(args.predicted_prob_labels, 'wb') as ps_list_pickle:
        pickle.dump(ps_list, ps_list_pickle, protocol=pickle.HIGHEST_PROTOCOL)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--loop_range", type=int, default=3600)
    p.add_argument("--data_range", type=int, default=50)
    p.add_argument("--labels_data", type=str, default="labels_data.txt")
    p.add_argument("--x_data", type=str, default="x_data.pickle")
    p.add_argument("--y_data", type=str, default="y_data.pickle")
    p.add_argument("--z_data", type=str, default="z_data.pickle")
    p.add_argument("--checkpoint", type=str, default="checkpoint")
    p.add_argument("--predicted_prob_labels", type=str, default="predicted_prob_labels")
    args = p.parse_args()
    start = time.time()
    predict_prob()
    end = time.time()
    print(str(end-start))