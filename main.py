import torch
from torch.utils.data import DataLoader

import os, datetime, warnings, logging
from pathlib import Path
# for evaluation
warnings.filterwarnings('ignore')
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from Models.diffusion import Diffusion
from train import Trainer
from utils import writelog, BaseDataset, SineDataset, fMRIDataset, MuJoCoDataset


def parse_args():
    import argparse
    parser = argparse.ArgumentParser(description='Frequency-based diffusion model')
    parser.add_argument('-g', '--gpu', type=str, default='0')
    parser.add_argument('-seed', '--seed', type=int, default=12345)
    parser.add_argument('-datasets', '--datasets', type=str, default='sine') # sine, stock, energy ...
    parser.add_argument('-path', '--path', type=str, default='./diffusion_datasets/')
    parser.add_argument('-save_dir', '--save_dir', type=str, default='./Output/')
    # training param.
    parser.add_argument('-batch_size', '--batch_size', type=int, default=128)
    parser.add_argument('-epochs', '--epochs', type=int, default=12000)
    parser.add_argument('-lr', '--base_lr', type=float, default=1e-5)
    parser.add_argument('--gradient_accumulate_every', type=int, default=2)
    parser.add_argument('--ema_decay', type=float, default=0.995)
    parser.add_argument('--ema_update_interval', type=int, default=10)
    parser.add_argument('--patience', type=int, default=3000)
    # model param.
    parser.add_argument('--timesteps', type=int, default=500)
    parser.add_argument('--sampling_timesteps', type=int, default=500)
    args = parser.parse_args()
    return args

def main():
    args = parse_args()
    # gpu setting
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    torch.manual_seed(args.seed)
    # log
    dir = args.save_dir + '{}/{}/{}'.format(args.datasets,
                                            str(datetime.datetime.now().strftime('%m%d')),
                                            str(datetime.datetime.now().strftime('%H%M%S')))
    Path(dir).mkdir(parents=True, exist_ok=True)
    f = open(dir + '/log.txt', 'a')
    writelog(f, '-'*20)
    writelog(f, 'Experiment settings')
    writelog(f, 'Datasets: ' + str(args.datasets)) # sine, stock, ETTh, Energy, fMRI, ...
    writelog(f, 'Batch size: ' + str(args.batch_size))
    writelog(f, 'Epochs: ' + str(args.epochs))
    writelog(f, 'Learning rate: ' + str(args.base_lr))
    writelog(f, 'Timesteps/Sampling_timesteps: {}/{}'.format(args.timesteps, args.sampling_timesteps))


    # load datasets
    if args.datasets in ['stock', 'energy', 'etth']:
        datasets = BaseDataset(args.datasets, args.path)
        _, len, dim = datasets.samples.shape
    elif args.datasets == 'fMRI':
        datasets = fMRIDataset(proportion=1.0, name=args.datasets, data_root=args.path)
        _, len, dim = datasets.samples.shape
    elif args.datasets == 'sine':
        datasets = SineDataset(num=10000, dim=5, window=24)
        _, len, dim = 10000, 24, 5
    elif args.datasets == 'mujoco':
        datasets = MuJoCoDataset(window=24, num=10000, dim=14)
        _, len, dim = 10000, 24, 14

    dataloader = DataLoader(datasets, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, sampler=None, drop_last=True)


    # define model
    model = Diffusion(seq_length=len,
                          feature_size=dim,
                          n_layer_enc=1,
                          n_layer_dec=2,
                          d_model=64,
                          n_head=4,
                          mlp_hid_times=4,
                          attn_pd=.0,
                          resid_pd=.0,
                          timesteps=500,
                          sampling_timesteps=500,
                          loss_type='l1',
                          beta_schedule='cosine',
                          reg_weight=None)
    # training
    train_op = Trainer(args, model, dataloader, dir, device)
    train_op.train()

    f.close()

if __name__ == "__main__":
    main()
