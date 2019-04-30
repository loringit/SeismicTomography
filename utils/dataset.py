import numpy as np
from torch.utils.data import Dataset
from torchvision import transforms
from utils import NMO

import os
import torch

use_gpu = torch.cuda.is_available()
device = torch.device("cuda" if use_gpu else "cpu")


class SeismicDataset(Dataset):
    """Seismic dataset."""

    def __init__(self, seismo_dir, velocity_dir, sparsity=1, transform=None, clean_seismo_path='data/seismogramm.txt'):
        """
        Args:
            seismo_dir (string): Directory with all the seismograms.
            velocity_dir (string): Directory with all the c1 velocity models.
            transform (callable, optional): Optional transform to be applied on a sample.
        """

        self.seismo_dir = seismo_dir
        self.velocity_dir = velocity_dir
        self.transform = transform
        self.sparsity = sparsity
        self.clean_seismo = np.loadtxt(clean_seismo_path)[:, 1::2]

        self.seismo_filenames = sorted(os.listdir(seismo_dir))
        self.velocity_filenames = sorted(os.listdir(velocity_dir))

    def __len__(self):
        return len(os.listdir(self.seismo_dir))

    def __getitem__(self, idx):
        seismogram = np.loadtxt(self.seismo_dir + self.seismo_filenames[idx])[:, 1::2] - self.clean_seismo
        velocity = np.fromfile(self.velocity_dir + self.velocity_filenames[idx], dtype='f')

        sample = {'seismogram': seismogram[::self.sparsity, ::self.sparsity].astype(np.float32),
                  'velocity': velocity.reshape(250, 500)[::self.sparsity, ::self.sparsity].astype(np.float32)}

        if self.transform:
            sample = self.transform(sample)

        return sample


class NormalMoveout(object):
    """Do normal moveout for seismogram."""

    def __init__(self, offsets, velocities, dt):
        self.offsets = offsets
        self.velocities = velocities
        self.dt = dt

    def __call__(self, sample):
        seismogram = sample['seismogram']
        corrected_seismogram = NMO.nmo_correction(seismogram, dt, offsets, velocities)

        return {'seismogram': corrected_seismogram, 'velocity': sample['velocity']}


class ToTensor(object):
    """Convert seismogram to tensor."""

    def __call__(self, sample):
        seismogram = sample['seismogram']
        tensor_seismogram = torch.from_numpy(seismogram)

        return {'seismogram': tensor_seismogram, 'velocity': sample['velocity']}


if __name__ == '__main__':
    dt = 0.001

    offsets = []
    for i in range(50, 2050, 10):
        offsets.append(np.absolute(1250 - i))

    velocities = []
    for i in range(2000):
        velocities.append(1500)

    nmo_transform = NormalMoveout(offsets=offsets, velocities=velocities, dt=dt)
    dataset = SeismicDataset(seismo_dir='train/raw/',
                             velocity_dir='train/outputs/',
                             transform=transforms.Compose([ToTensor()]))