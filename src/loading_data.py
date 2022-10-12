import os
import torch
import torch.utils.data as td
import torchvision as tv
import pandas as pd
import numpy as np
from PIL import Image
import scipy.io
from matplotlib import cm

class NoisyBSDSDataset(td.Dataset):

    def __init__(self, root_dir, mode='train', image_size=(180, 180), sigma=30):
        super(NoisyBSDSDataset, self).__init__()
        self.mode = mode
        self.image_size = image_size
        self.sigma = sigma
        self.images_dir = os.path.join(root_dir, mode)
        self.files = os.listdir(self.images_dir)

    def __len__(self):
        return len(self.files)

    def __repr__(self):
        return "NoisyBSDSDataset(mode={}, image_size={}, sigma={})". \
            format(self.mode, self.image_size, self.sigma)

    def __getitem__(self, idx):
        img_path = os.path.join(self.images_dir, self.files[idx])
        name = self.files[idx]
        clean = Image.open(img_path).convert('RGB')
        degraded_data=  scipy.io.loadmat(self.images_dir+'_noise_'+str(self.sigma)+'/'+os.path.splitext(self.files[idx])[0]+'.mat')
        noisy = degraded_data['noisy']
        i = (clean.size[0] - self.image_size[0])//2
        j = (clean.size[1] - self.image_size[1])//2
        # print(i,j)
        clean = clean.crop([i, j, i+self.image_size[0], j+self.image_size[1]])

        # print(noisy.size())
        noisy = noisy[j:j+self.image_size[1],i:i+self.image_size[0],:]
        noisy = np.transpose(noisy,(2,0,1))
        noisy = torch.from_numpy(noisy).float()/255.
        transform = tv.transforms.Compose([
            # convert it to a tensor
            tv.transforms.ToTensor(),
            # normalize it to the range [−1, 1]
            # tv.transforms.Normalize((.5, .5, .5), (.5, .5, .5))
        ])
        clean = transform(clean)
        # noisy = transform(noisy)
        # print(noisy.max())
        # print(noisy.min())
        # noisy = clean +   self.sigma/255 * torch.randn(clean.shape)
        return noisy, clean,name
