import torch
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_

import numpy as np

from tqdm import tqdm
from ema_pytorch import EMA
from lr_scheduler import ReduceLROnPlateauWithWarmup

from utils import get_freq_repr, normalize_to_neg_one_to_one

def cycle(dl):
    while True:
        for data in dl:
            yield data

class Trainer(object):
    def __init__(self, args, model, dataloader, dir, device):
        super(Trainer, self).__init__()
        self.args = args
        self.model = model
        self.save_dir = dir
        self.device = device
        self.dl = cycle(dataloader)

        self.train_num_step = args.epochs
        self.gradient_accumulate_every = args.gradient_accumulate_every
        start_lr = args.base_lr
        ema_decay = args.ema_decay
        ema_update_every = args.ema_update_interval

        self.opt = optim.Adam(filter(lambda p:p.requires_grad, self.model.parameters()), lr=start_lr, betas=[.9, .96])
        self.ema = EMA(self.model, beta=ema_decay, update_every=ema_update_every).to(self.device)

        self.scheduler = ReduceLROnPlateauWithWarmup(optimizer=self.opt,
                                                     factor=.5,
                                                     patience=args.patience,
                                                     min_lr=1e-5,
                                                     threshold=1e-1,
                                                     threshold_mode='rel',
                                                     warmup_lr=8e-4,
                                                     warmup=500,
                                                     verbose=False)

    def save(self, step, samples=None):
        data = {
            'step': step,
            'model': self.model.state_dict(),
            'ema': self.ema.state_dict(),
            'opt': self.opt.state_dict()
        }
        torch.save(data, self.save_dir + "/checkpoint_{}.pt".format(step))
        if samples is not None:
            np.save(self.save_dir+'/checkpoint_{}_samples.npy'.format(step), samples)

    def load(self, step):
        data = torch.load(self.save_dir + '/checkpoint_{}.pt'.format(step), map_location=self.device)
        self.model.load_state_dict(data['model'])
        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        self.ema.load_state_dict(data['ema'])

    def train(self):
        device = self.device
        step = 0
        with tqdm(initial=step, total=self.train_num_step) as pbar:
            while step < self.train_num_step:
                total_loss = 0.
                for _ in range(self.gradient_accumulate_every):
                    data = next(self.dl).to(device)
                    loss = self.model(data)
                    loss = loss / self.gradient_accumulate_every
                    loss.backward()
                    total_loss += loss.item()
                pbar.set_description('loss: {:.6f}'.format(total_loss))
                clip_grad_norm_(self.model.parameters(), 1.0)
                self.opt.step()
                self.scheduler.step(total_loss)
                self.opt.zero_grad()
                step += 1
                self.ema.update()

                with torch.no_grad():
                    if step % (self.train_num_step // 10) == 0:
                        samples = self.sample(num=10000, size_every=2001, shape=(data.shape[1], data.shape[2]))
                        self.save(step, samples)
                pbar.update(1)
        print("training complete")

    def sample(self, num, size_every, shape=None):

        samples = np.empty([0, shape[0], shape[1]])
        num_cycle = int(num // size_every) + 1

        freq_size = 0
        l_freqs = []
        h_freqs = []
        while True:
            l_freq, h_freq = get_freq_repr(next(self.dl).to(self.device))
            l_freqs.append(l_freq)
            h_freqs.append(h_freq)
            freq_size += l_freq.shape[0]
            if freq_size > num:
                break

        for _ in range(num_cycle):
            sample = self.ema.ema_model.generate_mts([torch.cat(l_freqs)[_*size_every:(_+1)*size_every], torch.cat(h_freqs)[_*size_every:(_+1)*size_every]], batch_size=size_every)
            samples = np.row_stack([samples, sample.detach().cpu().numpy()])
            torch.cuda.empty_cache()
        return samples
