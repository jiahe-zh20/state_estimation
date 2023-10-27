import argparse
import os
import sys
from datetime import datetime

import numpy as np
import torch
from data.dataloder import data_generator
from model import BertNet
from trainer import Trainer, model_evaluate
from utils import _logger
from Config import Config

start_time = datetime.now()
configs = Config()
parser = argparse.ArgumentParser()
device = 'cuda' if torch.cuda.is_available() else 'cpu'

######################## Model parameters ########################
home_dir = os.getcwd()

parser.add_argument('--ex_description',  default='test',             type=str,   help='Experiment Description')
parser.add_argument('--seed',            default=0,                  type=int,   help='seed value')
parser.add_argument('--training_mode',   default='self_supervised',  type=str,   help='Modes of choice')
parser.add_argument('--data_path',       default=r'data/saved_data',           type=str,   help='Path containing dataset')
parser.add_argument('--logs_save_dir',   default='ex_log',           type=str,   help='saving directory')
parser.add_argument('--home_path',       default=home_dir,           type=str,   help='Project home directory')
args = parser.parse_args()

ex_description = args.ex_description
training_mode = args.training_mode
log_save_dir = args.log_save_dir
data_path = args.data_path
SEED = args.seed

os.makedirs(log_save_dir, exist_ok=True)
ex_log_dir = os.path.join(log_save_dir, ex_description, training_mode + f'seed_{SEED}')
os.makedirs(ex_log_dir, exist_ok=True)

############## fix random seeds for reproducibility ##############
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
##################################################################

# logging
log_file_name = os.path.join(ex_log_dir, f"logs_{datetime.now().strftime(f'%d_%m_%Y_%H_%M_%S')}.log")
logger = _logger(log_file_name)
logger.debug("=" * 45)
logger.debug(f'Mode:    {training_mode}')
logger.debug("=" * 45)

# load data
train_dl, valid_dl, test_dl = data_generator(data_path, configs, training_mode)
logger.debug('Data loaded ...')

# load model
model = BertNet(configs)
model_optimizer = torch.optim.Adam(model.parameters(), lr=configs.lr, weight_decay=3e-4)
Trainer(model, model_optimizer, train_dl, valid_dl, test_dl,  device, logger, configs,
        ex_log_dir, training_mode)
logger.debug(f"Training time is : {datetime.now() - start_time}")

