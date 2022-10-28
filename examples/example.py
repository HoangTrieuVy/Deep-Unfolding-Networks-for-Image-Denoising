import torch
from PIL import Image
import torchvision as tv
import os
from prettytable import PrettyTable
import sys
sys.path.insert(0, '../src')
from utils import *
from argument import *
from model import *
import nntools as nt
from utils import DenoisingStatsManager

import matplotlib.pyplot as plt


def run(args):
    if args.i is not None:
        clean =  np.array(Image.open(args.i).convert('RGB'))/255.
    noisy = np.array(Image.open(args.n).convert('RGB'))
    noisy_np= np.transpose(noisy,(2,0,1))
    noisy_torch = torch.from_numpy(noisy_np).float()/255.

    device= 'cuda' if  torch.cuda.is_available() else 'cpu'
    #device = 'cpu'
    if args.model == 'DnCNN':
        net = DnCNN(K=9, F=13).to(device)
    elif args.model == 'unfolded_ISTA':
        net = unfolded_ISTA(K=13, F=21).to(device)
    elif args.model == 'unfolded_FISTA':
        net = unfolded_FISTA(K=13, F=21).to(device)
    elif args.model == 'unfolded_ScCP':
        net = unfolded_ScCP(K=13, F=21).to(device)
    elif args.model == 'unfolded_CP':
        net = unfolded_CP(K=13, F=21).to(device)
    else:
        print('Error chosen model name')

    sigma = 50
    batch_size = 10

    countparam = count_parameters(net,print_table=False,namenet=args.model)
    output_dir = '../checkpoints/'+args.model+'/'+args.model+"_F"\
                 +str(net.F)+"_K"+str(net.K)+"_batchsize"+str(batch_size)\
                 +'_sigma_'+str(sigma)+'_param_'+str(countparam)
    checkpoint_path = os.path.join(output_dir, "checkpoint.pth.tar")
    checkpoint = torch.load(checkpoint_path,map_location=device)
    net.load_state_dict(checkpoint['Net'])
    denoised= net(noisy_torch[None].to(device))[0]
    denoised = denoised.to('cpu').detach().numpy()
    denoised = np.moveaxis(denoised, [0, 1, 2], [2, 0, 1])
    sf=plt.figure(figsize=(10,5))
    if args.i is None:
        
        ax1=plt.subplot(121)
        plt.imshow(noisy)
        plt.title('Noisy')
        plt.axis('off')     
        ax2=plt.subplot(122)
        plt.imshow(denoised)
        plt.axis('off')
        plt.title(args.model+' denoiser')
        
        plt.show()
    else:
        plt.subplot(131)
        plt.imshow(clean)
        plt.axis('off')
        plt.title('Clean')
        ax1=plt.subplot(132)
        plt.imshow(noisy)
        plt.title('Noisy')
        plt.axis('off')
        ax1.text(0.5,-0.1, 'PSNR: '+  str(format(PSNR_np(noisy/255.,clean), '.2f')), size=12, ha="center", 
            transform=ax1.transAxes)
        ax2=plt.subplot(133)
        plt.imshow(denoised)
        plt.axis('off')
        ax2.text(0.5,-0.1, 'PSNR: '+  str(format(PSNR_np(denoised,clean), '.2f')), size=12, ha="center", 
            transform=ax2.transAxes)
        plt.title(args.model+' Denoised')
        plt.show()
    if args.saveresults is True:
        sf.savefig('results_'+args.model+'_'+args.n,bbox_inches='tight',pad_inches = 0)
if __name__ == '__main__':
    args = parse_for_test()
    run(args)
