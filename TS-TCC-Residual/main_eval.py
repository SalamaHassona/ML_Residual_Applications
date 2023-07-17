import torch

import gc
import time
import pickle
import os
import numpy as np
from datetime import datetime
import argparse
from utils import _logger
from models.TC import TC
from models.model import base_Model
import torch.nn.functional as F

# Args selections
start_time = datetime.now()

start = time.time()

parser = argparse.ArgumentParser()

######################## Model parameters ########################
home_dir = os.getcwd()
parser.add_argument('--experiment_description', default='exp1', type=str,
                    help='Experiment Description')
parser.add_argument('--run_description', default='run1', type=str,
                    help='Experiment Description')
parser.add_argument('--seed', default=0, type=int,
                    help='seed value')
parser.add_argument('--training_mode', default='eval', type=str,
                    help='Modes of choice: random_init, supervised, self_supervised, fine_tune, train_linear, eval')
parser.add_argument('--selected_dataset', default='arc', type=str,
                    help='Dataset of choice: arc')
parser.add_argument('--logs_save_dir', default='logs_arc', type=str,
                    help='saving directory')
parser.add_argument('--device', default='cuda', type=str,
                    help='cpu or cuda')
parser.add_argument('--home_path', default='./', type=str,
                    help='Project home directory')
parser.add_argument('--loop_range', default=1800, type=int,
                    help='loop range')
parser.add_argument('--data_range', default=100, type=int,
                    help='data range')
args = parser.parse_args()

device = torch.device(args.device)
experiment_description = args.experiment_description
data_type = args.selected_dataset
method = 'TS-TCC'
training_mode = args.training_mode
run_description = args.run_description
logs_save_dir = args.logs_save_dir
data_range = args.data_range
loop_range = args.loop_range
os.makedirs(logs_save_dir, exist_ok=True)

exec(f'from config_files.{data_type}_Configs import Config as Configs')
configs = Configs()

# ##### fix random seeds for reproducibility ########
SEED = args.seed
#####################################################

experiment_log_dir = os.path.join(logs_save_dir, experiment_description, run_description,
                                  training_mode + f"_seed_{SEED}")
os.makedirs(experiment_log_dir, exist_ok=True)

# loop through domains
counter = 0
src_counter = 0

# Logging
log_file_name = os.path.join(experiment_log_dir, f"logs_{datetime.now().strftime('%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Dataset: {data_type}')
logger.debug(f'Method:  {method}')
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# Load datasets
X_eval, y_eval = configs.data.get_data()

if len(X_eval.shape) < 3:
    X_eval = X_eval.unsqueeze(2)

if X_eval.shape.index(min(X_eval.shape)) != 1:  # make sure the Channels in second dim
    X_eval = X_eval.permute(0, 2, 1)

cr_period_list = y_eval.double().numpy()
# cr_period_list = np.array([[item[0], item[1], 0 if item[2] < 23 else 1] for item in y_eval], dtype=np.double)
ps_list = np.array([], dtype=np.int64).reshape(0, 5)
logger.debug("Data loaded ...")

# Load Model
model = base_Model(configs).to(device)
temporal_contr_model = TC(configs, device).to(device)

load_from = os.path.join(
    os.path.join(logs_save_dir, experiment_description, run_description, f"train_linear_seed_{SEED}", "saved_models"))
chkpoint = torch.load(os.path.join(load_from, "ckp_last.pt"), map_location=device)
# chkpoint = torch.load(os.path.join('./', "ckp_last.pt"), map_location=device)
model.load_state_dict(chkpoint["model_state_dict"])
model.eval()
temporal_contr_model.load_state_dict(chkpoint["temporal_contr_model_state_dict"])
temporal_contr_model.eval()

logger.debug("Evaluation started ....")

for i in range(0, loop_range):
    data = X_eval[i * data_range:(i + 1) * data_range, :]
    data = data.float().to(device)
    output, _ = model(data)
    ps = torch.exp(F.log_softmax(output, dim=1)).double().detach().cpu().numpy()
    # output = model.forward(x, device)
    # ps = torch.exp(output).double().detach().cpu().numpy()
    ps_list = np.vstack(
        (ps_list, np.hstack((cr_period_list[i * data_range:(i + 1) * data_range, 0].reshape((data_range, 1)),
                             cr_period_list[i * data_range:(i + 1) * data_range, 1].reshape((data_range, 1)),
                             cr_period_list[i * data_range:(i + 1) * data_range, 2].reshape((data_range, 1)),
                             ps))))
    if i % 200 == 0:
        logger.debug("\ni: " + str(i))
    del ps, output, data
    gc.collect()
    with torch.cuda.device(device):
        torch.cuda.empty_cache()

with open(os.path.join(experiment_log_dir, 'ps_eval.pickle'), 'wb') as ps_list_pickle:
    pickle.dump(ps_list, ps_list_pickle, protocol=pickle.HIGHEST_PROTOCOL)

logger.debug("\n################## Evaluation is Done! #########################")

logger.debug(f"Evaluation time is : {datetime.now() - start_time}")

end = time.time()
print(str(end - start))
