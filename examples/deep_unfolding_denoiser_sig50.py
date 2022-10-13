import torch
from PIL import Image
import torchvision as tv
import os
from prettytable import PrettyTable
import sys
sys.path.insert(0, '../src')
from utils import count_parameters
from argument import parse
from model import *
import nntools as nt
from utils import DenoisingStatsManager

import matplotlib.pyplot as plt

noisy = Image.open("10081.jpg").convert('RGB')
noisy = np.array(noisy)
noisy_np= np.transpose(noisy,(2,0,1))
noisy_torch = torch.from_numpy(noisy_np).float()/255.

def run(args):
    device= 'cuda' if  torch.cuda.is_available() else 'cpu'
    print(device)
    if args.model == 'DnCNN':
        print('Train DnCNN  ---- K= ', args.K,', F=',args.F,'-----')
        net = DnCNN(args.K, F=args.F).to(device)

    elif args.model == 'unfolded_ISTA':
        net = unfolded_ISTA(args.K, F=args.F).to(device)
    elif args.model == 'unfolded_FISTA':
        net = unfolded_FISTA(args.K, F=args.F).to(device)
    elif args.model == 'unfolded_CP_v2':
        net = unfolded_CP_v2(args.K, F=args.F).to(device)
    elif args.model == 'unfolded_CP_v3':
        net = unfolded_CP_v3(args.K, F=args.F).to(device)

    stats_manager = DenoisingStatsManager()
    countparam = count_parameters(net,print_table=False,namenet=args.model)
    output_dir = args.output_dir+args.model+'/'+args.model+"_F"\
                 +str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)\
                 +'_sigma_'+str(args.sigma)+'_param_'+str(countparam)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
    checkpoint = torch.load(checkpoint_path,map_location=device)
    net.load_state_dict(checkpoint['Net'])
    denoised= net(noisy_torch[None].to(device))[0]
    denoised = denoised.to('cpu').detach().numpy()
    denoised = np.moveaxis(denoised, [0, 1, 2], [2, 0, 1])
    print(denoised)
    plt.figure(figsize=(20,10))
    plt.box(False)
    plt.subplot(121)
    plt.imshow(noisy)
    plt.axis('off')
    plt.subplot(122)
    plt.imshow(denoised)
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    args = parse()
    run(args)
