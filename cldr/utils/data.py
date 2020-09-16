import h5py
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

import torch
from torchvision import transforms


class Shapes3D(torch.utils.data.Dataset):
    size = 480000
    mean = [0.5026, 0.5788, 0.6034]
    std  = [0.3412, 0.3529, 0.3536]
    num_factors = 6
    factor_names = ['floor_hue', 'wall_hue', 'object_hue', 'scale', 'shape',
                    'orientation']
    num_factor_values = {'floor_hue': 10, 'wall_hue': 10, 'object_hue': 10, 
                         'scale': 8, 'shape': 4, 'orientation': 15}
    
    @staticmethod
    def extract_h5(h5_path, base_path='./data/Shapes3D', N=10000):
        h5 = h5py.File(h5_path, 'r')
        Path(base_path + '/images').mkdir(parents=True, exist_ok=True)
        Path(base_path + '/factors').mkdir(parents=True, exist_ok=True)
        images_path = Path(base_path + '/images')
        factors_path = Path(base_path +'/factors')
        images = h5['images']
        factors = h5['labels']
        for i in range(images.len()//N):
            img = images[i*N:(i+1)*N]
            fac = factors[i*N:(i+1)*N]
            for j in range(N):
                np.save(images_path/'{}.npy'.format(i*N+j), img[j])
                np.save(factors_path/'{}.npy'.format(i*N+j), fac[j])
    
    def __init__(self, path, normalize=False):
        base_path = Path(path)
        self.images_path = base_path/'images'
        self.factors_path = base_path/'factors'
        self.transform = transforms.Compose(
            [transforms.ToTensor()] +
            ([transforms.Normalize(self.mean, self.std)] if normalize else [])
        )

    def __len__(self):
        return self.size
    
    def __getitem__(self, i):
        image = np.load(self.images_path/'{}.npy'.format(i))
        factor = np.load(self.factors_path/'{}.npy'.format(i))
        factor = torch.Tensor(factor)
        image = self.transform(image)
        return image, factor
    
    def get_index(self, factors):
        indices = 0
        base = 1
        for factor, name in reversed(list(enumerate(self.factor_names))):
            indices += factors[factor] * base
            base *= self.num_factor_values[name]
        return indices

    def sample_batch(self, batch_size, fixed_factor, fixed_factor_value):
        factors = np.zeros([self.num_factors, batch_size],
                           dtype=np.int32)
        for factor, name in enumerate(self.factor_names):
            num_choices = self.num_factor_values[name]
            factors[factor] = np.random.choice(num_choices, batch_size)
        factors[fixed_factor] = fixed_factor_value
        indices = self.get_index(factors)
        ims = []
        facs = []
        for ind in indices:
            im, fac = self.__getitem__(ind)
            ims.append(im)
            facs.append(fac)
        ims = torch.stack(ims, dim=0)
        facs = torch.stack(facs, dim=0)
        return ims, facs


class BatchTransform:
    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return torch.stack([self.transform(xi) for xi in x])
