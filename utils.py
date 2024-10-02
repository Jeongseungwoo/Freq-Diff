import torch
from torch.utils.data import Dataset
from torch.fft import rfft


import numpy as np
import pandas as pd
import os, math

from scipy import io
from sklearn.preprocessing import MinMaxScaler
from pathlib import Path
from tqdm import tqdm

def writelog(f, line):
    f.write(line + "\n")
    print(line)

def normalize_to_neg_one_to_one(x):
    return x * 2 - 1

def unnormalize_to_zero_to_one(x):
    return (x + 1) * 0.5

# stocks, energy, etth
class BaseDataset(Dataset):
    def __init__(self, name, data_root, window=24, proportion=1, train=True, output_dir='./Output', seed=1234):
        super(BaseDataset, self).__init__()
        self.name = name
        if name == 'etth':
            data_root = data_root + "ETTh.csv"
        elif name == 'fMRI':
            data_root = data_root + 'fMRI/'
        else:
            data_root = data_root + name + "_data.csv"
        self.data_root = data_root
        self.window = window
        self.proportion = proportion
        self.train = train
        self.auto_norm = True  # neg_one_to_one
        self.save2npy = True  # save gt

        self.dir = os.path.join(output_dir, 'samples')
        Path(self.dir).mkdir(parents=True, exist_ok=True)

        self.rawdata, self.scaler = self.read_data(data_root, self.name)
        self.sample_num_total = max(self.rawdata.shape[0] - self.window + 1, 0)
        self.var_num = self.rawdata.shape[-1]  # number of features
        self.data = self.__normalize(self.rawdata)
        train_data, inference_data = self.__getsamples(self.data, proportion, seed=seed)
        self.samples = train_data if self.train else inference_data

    def __getitem__(self, item):
        return torch.from_numpy(self.samples[item, :, :]).float()

    def __len__(self):
        return len(self.samples)

    def __getsamples(self, data, proportion, seed):
        x = np.zeros((self.sample_num_total, self.window, self.var_num))
        for i in range(self.sample_num_total):
            start = i
            end = i + self.window
            x[i, :, :] = data[start:end, :]

        train_data, test_data = self.divide(x, proportion, seed)

        if self.save2npy:
            if 1 - proportion > 0:
                np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_test.npy"),
                        self.unnormalize(test_data))
            np.save(os.path.join(self.dir, f"{self.name}_ground_truth_{self.window}_train.npy"),
                    self.unnormalize(train_data))
            if self.auto_norm:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"),
                            unnormalize_to_zero_to_one(test_data))
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"),
                        unnormalize_to_zero_to_one(train_data))
            else:
                if 1 - proportion > 0:
                    np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_test.npy"), test_data)
                np.save(os.path.join(self.dir, f"{self.name}_norm_truth_{self.window}_train.npy"), train_data)
        return train_data, test_data

    @staticmethod
    def divide(data, ratio, seed=2023):
        size = data.shape[0]
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        regular_train_num = int(np.ceil(size * ratio))
        id_rdm = np.random.permutation(size)
        regular_train_id = id_rdm[:regular_train_num]
        irregular_train_id = id_rdm[regular_train_num:]

        regular_data = data[regular_train_id, :]
        irregular_data = data[irregular_train_id, :]

        # Restore RNG.
        np.random.set_state(st0)
        return regular_data, irregular_data

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        df = pd.read_csv(filepath, header=0)
        if name == 'etth':
            df.drop(df.columns[0], axis=1, inplace=True)
        data = df.values
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

    def normalize(self, sq):
        d = sq.reshape(-1, self.var_num)
        d = self.scaler.transform(d)
        if self.auto_norm:
            d = normalize_to_neg_one_to_one(d)
        return d.reshape(-1, self.window, self.var_num)

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)


class SineDataset(Dataset):
    def __init__(self, window=128, num=30000, dim=12, train=True, output_dir='./Output', seed=1234):
        super(SineDataset, self).__init__()
        self.window = window
        self.num = num
        self.dim =dim
        self.train = train

        self.save2npy = True  # save gt

        self.dir = os.path.join(output_dir, 'samples')
        Path(self.dir).mkdir(parents=True, exist_ok=True)

        self.rawdata = self.sine_data_generation(no=num, seq_len=window, dim=dim, save2npy=self.save2npy,
                                                 seed=seed, dir=self.dir, period=self.train)

        self.auto_norm = True
        self.samples = self.normalize(self.rawdata)
        self.var_num = dim
        self.sample_num = self.samples.shape[0]
        self.window = window

    def __getitem__(self, item):
        return torch.from_numpy(self.samples[item, :, :]).float()

    def __len__(self):
        return len(self.samples)

    def normalize(self, rawdata):
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(rawdata)
        return data

    def unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        return data

    @staticmethod
    def sine_data_generation(no, seq_len, dim, save2npy=True, seed=123, dir="./", period=True):
        """Sine data generation.

        Args:
           - no: the number of samples
           - seq_len: sequence length of the time-series
           - dim: feature dimensions

        Returns:
           - data: generated data
        """
        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        period = 'train' if period == True else 'test'

        # Initialize the output
        data = list()
        # Generate sine data
        for i in tqdm(range(0, no), total=no, desc="Sampling sine-dataset"):
            # Initialize each time-series
            temp = list()
            # For each feature
            for k in range(dim):
                # Randomly drawn frequency and phase
                freq = np.random.uniform(0, 0.1)
                phase = np.random.uniform(0, 0.1)

                # Generate sine signal based on the drawn frequency and phase
                temp_data = [np.sin(freq * j + phase) for j in range(seq_len)]
                temp.append(temp_data)

            # Align row/column
            temp = np.transpose(np.asarray(temp))
            # Normalize to [0,1]
            temp = (temp + 1) * 0.5
            # Stack the generated data
            data.append(temp)

        # Restore RNG.
        np.random.set_state(st0)
        data = np.array(data)
        if save2npy:
            np.save(os.path.join(dir, f"sine_ground_truth_{seq_len}_{period}.npy"), data)

        return data


class fMRIDataset(BaseDataset):
    def __init__(
        self,
        proportion=1.,
        **kwargs
    ):
        super().__init__(proportion=proportion, **kwargs)

    @staticmethod
    def read_data(filepath, name=''):
        """Reads a single .csv
        """
        data = io.loadmat(filepath + '/sim4.mat')['ts']
        scaler = MinMaxScaler()
        scaler = scaler.fit(data)
        return data, scaler

class MuJoCoDataset(Dataset):
    def __init__(self, window=128, num=30000, dim=12, train=True, output_dir='./Output', seed=1234):
        super(MuJoCoDataset, self).__init__()
        self.window = window
        self.num = num
        self.dim = dim
        self.train = train
        self.var_num = dim
        self.period = 'train'

        self.save2npy = True  # save gt

        self.dir = os.path.join(output_dir, 'samples')
        Path(self.dir).mkdir(parents=True, exist_ok=True)
        self.rawdata, self.scaler = self._generate_random_trajectories(n_samples=num, seed=seed)

        self.auto_norm = True
        self.samples = self.normalize(self.rawdata)
        self.sample_num = self.samples.shape[0]
        self.window = window

    def __getitem__(self, ind):
        # if self.period == 'test':
        #     x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        #     m = self.masking[ind, :, :]  # (seq_length, feat_dim) boolean array
        #     return torch.from_numpy(x).float(), torch.from_numpy(m)
        x = self.samples[ind, :, :]  # (seq_length, feat_dim) array
        return torch.from_numpy(x).float()

    def __len__(self):
        return self.sample_num

    def _generate_random_trajectories(self, n_samples, seed=123):
        try:
            from dm_control import suite  # noqa: F401
        except ImportError as e:
            raise Exception('Deepmind Control Suite is required to generate the dataset.') from e

        env = suite.load('hopper', 'stand')
        physics = env.physics

        # Store the state of the RNG to restore later.
        st0 = np.random.get_state()
        np.random.seed(seed)

        data = np.zeros((n_samples, self.window, self.var_num))
        for i in range(n_samples):
            with physics.reset_context():
                # x and z positions of the hopper. We want z > 0 for the hopper to stay above ground.
                physics.data.qpos[:2] = np.random.uniform(0, 0.5, size=2)
                physics.data.qpos[2:] = np.random.uniform(-2, 2, size=physics.data.qpos[2:].shape)
                physics.data.qvel[:] = np.random.uniform(-5, 5, size=physics.data.qvel.shape)

            for t in range(self.window):
                data[i, t, :self.var_num // 2] = physics.data.qpos
                data[i, t, self.var_num // 2:] = physics.data.qvel
                physics.step()

            # Restore RNG.
        np.random.set_state(st0)

        scaler = MinMaxScaler()
        scaler = scaler.fit(data.reshape(-1, self.var_num))
        return data, scaler

    def normalize(self, sq):
        d = self.__normalize(sq.reshape(-1, self.var_num))
        data = d.reshape(-1, self.window, self.var_num)
        if self.save2npy:
            np.save(os.path.join(self.dir, f"mujoco_ground_truth_{self.window}_{self.period}.npy"), sq)

            if self.auto_norm:
                np.save(os.path.join(self.dir, f"mujoco_norm_truth_{self.window}_{self.period}.npy"), unnormalize_to_zero_to_one(data))
            else:
                np.save(os.path.join(self.dir, f"mujoco_norm_truth_{self.window}_{self.period}.npy"), data)

        return data

    def __normalize(self, rawdata):
        data = self.scaler.transform(rawdata)
        if self.auto_norm:
            data = normalize_to_neg_one_to_one(data)
        return data

    def unnormalize(self, sq):
        d = self.__unnormalize(sq.reshape(-1, self.var_num))
        return d.reshape(-1, self.window, self.var_num)

    def __unnormalize(self, data):
        if self.auto_norm:
            data = unnormalize_to_zero_to_one(data)
        x = data
        return self.scaler.inverse_transform(x)


def dft(x: torch.Tensor) -> torch.Tensor:
    """Compute the DFT of the input time series by keeping only the non-redundant components.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).

    Returns:
        torch.Tensor: DFT of x with the same size (batch_size, max_len, n_channels).
    """

    max_len = x.size(1)

    # Compute the FFT until the Nyquist frequency
    dft_full = rfft(x, dim=1, norm="ortho")
    dft_re = torch.real(dft_full)
    dft_im = torch.imag(dft_full)

    # The first harmonic corresponds to the mean, which is always real
    zero_padding = torch.zeros_like(dft_im[:, 0, :], device=x.device)
    assert torch.allclose(
        dft_im[:, 0, :], zero_padding
    ), f"The first harmonic of a real time series should be real, yet got imaginary part {dft_im[:, 0, :]}."
    dft_im = dft_im[:, 1:]

    # If max_len is even, the last component is always zero
    if max_len % 2 == 0:
        assert torch.allclose(
            dft_im[:, -1, :], zero_padding
        ), f"Got an even {max_len}, which should be real at the Nyquist frequency, yet got imaginary part {dft_im[:, -1, :]}."
        dft_im = dft_im[:, :-1]

    # Concatenate real and imaginary parts
    x_tilde = torch.cat((dft_re, dft_im), dim=1)
    assert (
        x_tilde.size() == x.size()
    ), f"The DFT and the input should have the same size. Got {x_tilde.size()} and {x.size()} instead."

    return x_tilde.detach()

def spectral_density(x: torch.Tensor, apply_dft: bool = True) -> torch.Tensor:
    """Compute the spectral density of the input time series.

    Args:
        x (torch.Tensor): Time series of shape (batch_size, max_len, n_channels).
        apply_dft (bool, optional): Whether to apply the DFT to the input. Defaults to True.

    Returns:
        torch.Tensor: Spectral density of x with the size (batch_size, n_frequencies, n_channels).
    """

    max_len = x.size(1)
    x = dft(x) if apply_dft else x

    # Extract real and imaginary parts
    n_real = math.ceil((max_len + 1) / 2)
    x_re = x[:, :n_real, :]
    x_im = x[:, n_real:, :]

    # Create imaginary tensor
    zero_padding = torch.zeros(size=(x.size(0), 1, x.size(2)), device=x.device)
    x_im = torch.cat((zero_padding, x_im), dim=1)

    # If number of time steps is even, put the null imaginary part
    if max_len % 2 == 0:
        x_im = torch.cat((x_im, zero_padding), dim=1)

    assert (
        x_im.size() == x_re.size()
    ), f"The real and imaginary parts should have the same shape, got {x_re.size()} and {x_im.size()} instead."

    # Compute the spectral density
    x_dens = x_re**2 + x_im**2
    assert isinstance(x_dens, torch.Tensor)
    return x_dens


def get_freq_repr(x, threshold=.8, eps=1e-15):
    X_spec = spectral_density(x)
    X_spec_norm = X_spec.sum(dim=2, keepdim=True) / (eps + X_spec.sum(dim=(1, 2), keepdim=True))
    X_spec_norm_mean = torch.mean(X_spec_norm, dim=(0, 2), )
    X_spec_norm_se = torch.std(X_spec_norm, dim=(0, 2), ) / math.sqrt(len(X_spec))

    # Compute normalized frequency
    freq_norm = [k / (X_spec.shape[1] - 1) for k in range(X_spec.shape[1])]

    l_spec = torch.zeros_like(X_spec)
    h_spec = torch.zeros_like(X_spec)
    for i in range(len(X_spec_norm_mean)):
        if X_spec_norm_mean[:i].sum() > threshold:
            break
    # scaling
    X_spec_scale = (X_spec - X_spec.mean(dim=(0, 2), keepdim=True)) / X_spec.std(dim=(0, 2), keepdim=True)
    l_spec[:, :i, :] = X_spec_scale[:, :i, :]
    h_spec[:, i:, :] = X_spec_scale[:, i:, :]

    return l_spec, h_spec
