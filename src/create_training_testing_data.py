import torch
import matplotlib.pyplot as plt
from data import NoisyBSDSDataset
from argument import parse
from model_10_jan import *
import nntools as nt
from utils import DenoisingStatsManager, plot,plot_compare
from prettytable import PrettyTable
import pickle
import numpy as np
import scipy.io
import scipy as scp

def imshow(image, ax=plt):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    # image = (image + 1) / 2
    # image[image < 0] = 0
    # image[image > 1] = 1
    h = ax.imshow(image)
    ax.axis('off')
    return h

root_dir = '../dataset/BSDS500_300/images'
mode = 'train'
images_dir= os.path.join(root_dir, mode)
files = os.listdir(images_dir)

# sig = 25

for idx in range(len(files)):
    img_path = os.path.join(images_dir, files[idx])
    clean = np.array(Image.open(img_path).convert('RGB'))

    noisy = clean + sig * np.random.normal(0,1,clean.shape)
    scipy.io.savemat('../dataset/BSDS500_300/images/train_noise_'+str(sig)+'/'+os.path.splitext(files[idx])[0]+'.mat', dict(noisy=noisy,clean=clean))

root_dir = '../dataset/BSDS500_300/images'
mode = 'test'
images_dir= os.path.join(root_dir, mode)
files = os.listdir(images_dir)
for idx in range(len(files)):
    img_path = os.path.join(images_dir, files[idx])
    print(files[idx])
    clean = np.array(Image.open(img_path).convert('RGB'))
    noisy = clean + sig * np.random.normal(0,1,clean.shape)
    # print(noisy)
    scipy.io.savemat('../dataset/BSDS500_300/images/test_noise_'+str(sig)+'/'+os.path.splitext(files[idx])[0]+'.mat', dict(noisy=noisy,clean=clean))


#--- Test Code ---
# train_set = NoisyBSDSDataset('../dataset/BSDS300/images/', image_size=(180,180), sigma=sig)
# test_set = NoisyBSDSDataset('../dataset/BSDS300/images/', mode='test', image_size=(180,180), sigma=sig)
# fig, axes = plt.subplots(ncols=2, nrows=2, figsize=(20, 20))
# imshow(test_set[55][0],ax=axes[0,1])
# imshow(test_set[55][1],ax=axes[0,0])
# fig.savefig('out/outtest.png')
# plt.tight_layout()
# fig.canvas.draw()
#
