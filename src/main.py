import argparse
from GPUtil import showUtilization as gpu_usage
import numpy as np
import torch
from numba import cuda
import gc
import matplotlib.pyplot as plt
# from data import NoisyBSDSDataset
from argument import parse
from model import *
import nntools as nt
from utils import DenoisingStatsManager,save_compare
from prettytable import PrettyTable
from skimage.metrics import structural_similarity as ssim
from loading_data import *
from torch.nn import functional as F

def PSNR(img1, img2):
    img1 = img1.to('cpu').numpy()
    img1 = np.moveaxis(img1, [0, 1, 2], [2, 0, 1])
    img2 = img2.to('cpu').numpy()
    img2 = np.moveaxis(img2, [0, 1, 2], [2, 0, 1])
    return 10*np.log10(np.max(img1)/(np.linalg.norm(img1-img2)**2))

def SSIM(img1,img2):
    img1 = img1.to('cpu').numpy()
    img1 = np.moveaxis(img1, [0, 1, 2], [2, 0, 1])
    img2 = img2.to('cpu').numpy()
    img2 = np.moveaxis(img2, [0, 1, 2], [2, 0, 1])
    return ssim(img1, img2,multichannel=True)

def count_parameters(model,print_table=True,namenet=''):
    table = PrettyTable(["Modules", "Parameters"])
    total_params = 0
    for name, parameter in model.named_parameters():
        if not parameter.requires_grad: continue
        param = parameter.numel()
        table.add_row([name, param])
        total_params+=param
    if print_table == True:
        print(table)
    print(namenet+" has total Trainable Params: ",total_params)
    return total_params


def op_W_k_FISTA(D1k,D2k,x1,x2,alpha,k):
    u1 = x2
    D2k_x1 = F.conv2d(input=x1, weight=D2k,padding=1)
    D1k_D2k_x1 = F.conv2d(input=D2k_x1, weight=D1k,padding=1)
    D2k_x2 = F.conv2d(input=x2, weight=D2k,padding=1)
    D1k_D2k_x2 = F.conv2d(input=D2k_x2, weight=D1k,padding=1)
    u2 = -alpha[k]*x1+alpha[k]*D1k_D2k_x1+ (1+alpha[k])*x2-(1+alpha[k])*D1k_D2k_x2
    return u1,u2

def op_W_k_T_FISTA(D1kT,D2kT,u1,u2,alpha,k):
    D1kT_u2 = F.conv2d(input=u1, weight=D1kT,padding=1)
    D2kT_D1kT_u2 = F.conv2d(input=D1kT_u2, weight=D2kT,padding=1)
    x1 = -alpha[k]*u2+ alpha[k]*D2kT_D1kT_u2
    x2 =  u1 + (1+alpha[k])*u2- (1+alpha[k])*D2kT_D1kT_u2
    return x1,x2

def op_W_k_ISTA(D1k,D2k,x1,x2):
    u1 = x2
    D2k_x1 = F.conv2d(input=x1, weight=D2k,padding=1)
    D1k_D2k_x1 = F.conv2d(input=D2k_x1, weight=D1k,padding=1)
    D2k_x2 = F.conv2d(input=x2, weight=D2k,padding=1)
    D1k_D2k_x2 = F.conv2d(input=D2k_x2, weight=D1k,padding=1)
    u2 =  x2-D1k_D2k_x2
    return u1,u2

def op_W_k_T_ISTA(D1kT,D2kT,u1,u2):
    D1kT_u2 = F.conv2d(input=u1, weight=D1kT,padding=1)
    D2kT_D1kT_u2 = F.conv2d(input=D1kT_u2, weight=D2kT,padding=1)
    x1 = torch.zeros_like(u2)
    x2 =  u1 + u2- D2kT_D1kT_u2
    return x1,x2

def op_W_k_CPv2(D1k,D2k_prev,x1,x2,alpha,sigma,k):
    D2k_prev_x2 = F.conv2d(input=x2, weight=D2k_prev,padding=1)
    D1k_x1 = F.conv2d(input=x1, weight=D1k,padding=1)
    D1k_D2k_prev_x2 = F.conv2d(input=D2k_prev_x2, weight=D1k,padding=1)
    u1 = (1/(1+sigma[k-1]))*x1-sigma[k-1]/(1+sigma[k-1])*D2k_prev_x2
    u2 = (1+alpha[k])/(1+sigma[k-1])*D1k_x1-alpha[k]*D1k_x1 + x2 - (1+alpha[k])*sigma[k-1]/(1+sigma[k-1])*D1k_D2k_prev_x2
    return u1,u2

def op_WT_k_CPv2(D1kT,D2kT_prev,u1,u2,alpha,sigma,k):
    D1kT_u2  = F.conv2d(input=u2, weight=D1kT,padding=1)
    D2kT_prev_u1 = F.conv2d(input=u1, weight=D2kT_prev,padding=1)
    D2kT_prev_D1kT_u2  = F.conv2d(input=D1kT_u2, weight=D2kT_prev,padding=1)
    x1 = (1/(1+sigma[k-1]))*u1+ (1+alpha[k])/(1+sigma[k-1])*D1kT_u2-alpha[k]*D1kT_u2
    x2 = -(sigma[k-1])/(1+sigma[k-1]+1)*D2kT_prev_u1+u2-(1+alpha[k])*sigma[k-1]/(1+sigma[k-1])*D2kT_prev_D1kT_u2
    return x1,x2

def op_W_k_CPv3(D1k,D2k_prev,x1,x2,sigma,k):
    D2k_prev_x2 = F.conv2d(input=x2, weight=D2k_prev,padding=1)
    D1k_x1 = F.conv2d(input=x1, weight=D1k,padding=1)
    D1k_D2k_prev_x2 = F.conv2d(input=D2k_prev_x2, weight=D1k,padding=1)
    u1 = (1/(1+sigma[k]))*x1-sigma[k-1]/(1+sigma[k-1])*D2k_prev_x2
    u2 = (1)/(1+sigma[k-1])*D1k_x1- x2 - (1)*sigma[k-1]/(1+sigma[k-1])*D1k_D2k_prev_x2
    return u1,u2

def op_WT_k_CPv3(D1kT,D2kT_prev,u1,u2,sigma,k):
    D1kT_u2  = F.conv2d(input=u2, weight=D1kT,padding=1)
    D2kT_prev_u1 = F.conv2d(input=u1, weight=D2kT_prev,padding=1)
    D2kT_prev_D1kT_u2  = F.conv2d(input=D1kT_u2, weight=D2kT_prev,padding=1)
    x1 = (1/(1+sigma[k]))*u1+ (1)/(1+sigma[k-1])*D1kT_u2
    x2 = -(sigma[k-1])/(1+sigma[k-1]+1)*D2kT_prev_u1+u2-(1)*sigma[k-1]/(1+sigma[k-1])*D2kT_prev_D1kT_u2
    return x1,x2


def calul_norm_ISTA(net,k,image):
    # W_k = recupere_poids(net,k)
    D1k= None
    D2k= None
    D1kT= None
    D2kT= None
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad: continue
        if name =='conv.'+str(2*k)+'.weight':
            D1k = parameter
            D1kT= torch.flip(D1k,[2,3])
            D1kT  = torch.transpose(D1kT,0,1)

        if name =='conv.'+str(2*k+1)+'.weight':
            D2k = parameter
            # D2kT = torch.transpose(D2k, 0,1)
            D2kT= torch.flip(D2k,[2,3])
            D2kT  = torch.transpose(D2kT,0,1)
    with torch.no_grad():
        #Init
        image = torch.rand(1,3,320,320).cuda()
        xn1 = F.conv2d(input=image, weight=D1k,padding=1)
        xn2 = F.conv2d(input=image, weight=D1k,padding=1)
        rhon = 1+1e-2
        rhok = 1
        iter=0

        ### Check D1k,D1kT and D2k,D2kT
        # a= F.conv2d(input=xn1, weight=D2k,padding=1)
        # b= torch.rand(a.shape).cuda()
        # print(torch.sum(a*b))
        # c = F.conv2d(input=b, weight=D2kT,padding=1)
        # print(torch.sum(c*xn1   ))

        while abs(rhok - rhon)/abs(rhok) >= 1e-6 and iter<10000:

            x  = torch.cat((xn1,xn2),0)
            x1 = xn1/torch.norm(x,p='fro')
            x2 = xn2/torch.norm(x,p='fro')
        #     [un1,un2] = op_W_k(W_k,[x1,x2])       = [                     x2              ,  x2-D1k(D2k(x2)) ]
        #     [xn1,xn2] = op_W_k_T(W_k_T,[un1,un2]) = [                      0              ,         un1 + un2- *(D2kT(D1kT(un2)))              ]
            un1,un2 = op_W_k_ISTA(D1k,D2k,x1,x2)
            xn1,xn2 = op_W_k_T_ISTA(D1kT,D2kT,un1,un2)
            rhon = rhok
            iter+=1
            rhok = torch.norm(torch.cat((xn1,xn2),0),p='fro')
        gc.collect()
        torch.cuda.empty_cache()
        return torch.sqrt(rhok)

def calul_norm_FISTA(net,k,image):
    # W_k = recupere_poids(net,k)
    alpha = None
    D1k= None
    D2k= None
    D1kT= None
    D2kT= None
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad: continue
        if  name =='multip':
            alpha = parameter
        if name =='conv.'+str(2*k)+'.weight':
            D1k = parameter
            D1kT= torch.flip(D1k,[2,3])
            D1kT  = torch.transpose(D1kT,0,1)

        if name =='conv.'+str(2*k+1)+'.weight':
            D2k = parameter
            # D2kT = torch.transpose(D2k, 0,1)
            D2kT= torch.flip(D2k,[2,3])
            D2kT  = torch.transpose(D2kT,0,1)

    with torch.no_grad():
        #Init
        image = torch.rand(1,3,320,320).cuda()
        xn1 = F.conv2d(input=image, weight=D1k,padding=1)
        xn2 = F.conv2d(input=image, weight=D1k,padding=1)
        rhon = 1+1e-2
        rhok = 1
        iter=0

        ### Check D1k,D1kT and D2k,D2kT
        # a= F.conv2d(input=xn1, weight=D2k,padding=1)
        # b= torch.rand(a.shape).cuda()
        # print(torch.sum(a*b))
        # c = F.conv2d(input=b, weight=D2kT,padding=1)
        # print(torch.sum(c*xn1   ))

        while abs(rhok - rhon)/abs(rhok) >= 1e-6 and iter<10000:

            x  = torch.cat((xn1,xn2),0)
            x1 = xn1/torch.norm(x,p='fro')
            x2 = xn2/torch.norm(x,p='fro')
        #     [un1,un2] = op_W_k(W_k,[x1,x2])       = [                     x2              , -alpha_k*x1+alpha_k*D1k(D2k(x1))+ (1+alpha_k)*x2-(1+alpha_k)*D1k(D2k(x2)) ]
        #     [xn1,xn2] = op_W_k_T(W_k_T,[un1,un2]) = [-alpha_k*un2+ alpha_k*D2kT(D1kT(un2)),         un1 + (1+alpha_k)*un2- (1+alpha_k)*(D2kT(D1kT(un2)))              ]
            un1,un2 = op_W_k_FISTA(D1k,D2k,x1,x2,alpha,k-1)
            xn1,xn2 = op_W_k_T_FISTA(D1kT,D2kT,un1,un2,alpha,k-1)
            rhon = rhok
            iter+=1
            rhok = torch.norm(torch.cat((xn1,xn2),0),p='fro')
        gc.collect()
        torch.cuda.empty_cache()
        return torch.sqrt(rhok)

def calul_norm_CPv3(net,k,image):
    # W_k = recupere_poids(net,k)
    alpha = None
    sigma = None
    D1k= None
    D2k_prev= None
    D1kT= None
    D2kT_prev= None
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad: continue
        if  name =='sigma':
            sigma = parameter
        if name =='conv.'+str(2*k)+'.weight':
            D1k = parameter
            D1kT= torch.flip(D1k,[2,3])
            D1kT  = torch.transpose(D1kT,0,1)

        if name =='conv.'+str(2*(k-1)+1)+'.weight':
            D2k_prev = parameter
            D2kT_prev= torch.flip(D2k_prev,[2,3])
            D2kT_prev  = torch.transpose(D2kT_prev,0,1)

    with torch.no_grad():
        #Init
        image = torch.rand(1,3,320,320).cuda()
        xn1 = image
        xn2 = F.conv2d(input=image, weight=D1k,padding=1)
        #torch.rand(xn1.shape).cuda()
        rhon = 1+1e-2
        rhok = 1
        iter=0

        ### Check D1k,D1kT and D2k,D2kT
        # a= F.conv2d(input=xn1, weight=D2k,padding=1)
        # b= torch.rand(a.shape).cuda()
        # print(torch.sum(a*b))
        # c = F.conv2d(input=b, weight=D2kT,padding=1)
        # print(torch.sum(c*xn1   ))

        while abs(rhok - rhon)/abs(rhok) >= 1e-6 and iter<10000:

            # x  = torch.cat((xn1,xn2),0)
            norm_x = torch.sqrt(torch.norm(xn1,p='fro')+torch.norm(xn2,p='fro'))
            x1 = xn1/norm_x#torch.norm(x,p='fro')
            x2 = xn2/norm_x#torch.norm(x,p='fro')
            un1,un2 = op_W_k_CPv3(D1k,D2k_prev,x1,x2,sigma,k-1)
            xn1,xn2 = op_WT_k_CPv3(D1kT,D2kT_prev,un1,un2,sigma,k-1)
            rhon = rhok
            iter+=1
            rhok = torch.sqrt(torch.norm(xn1,p='fro')**2+torch.norm(xn2,p='fro')**2)#torch.norm(torch.cat((xn1,xn2),0),p='fro')
        gc.collect()
        torch.cuda.empty_cache()
        return torch.sqrt(rhok)

def calul_norm_CPv2(net,k,image):
    # W_k = recupere_poids(net,k)
    alpha = None
    sigma = None
    D1k= None
    D2k_prev= None
    D1kT= None
    D2kT_prev= None
    for name, parameter in net.named_parameters():
        if not parameter.requires_grad: continue
        if  name =='alpha':
            alpha = parameter
        if  name =='sigma':
            sigma = parameter
        if name =='conv.'+str(2*k)+'.weight':
            D1k = parameter
            D1kT= torch.flip(D1k,[2,3])
            D1kT  = torch.transpose(D1kT,0,1)

        if name =='conv.'+str(2*(k-1)+1)+'.weight':
            D2k_prev = parameter
            D2kT_prev= torch.flip(D2k_prev,[2,3])
            D2kT_prev  = torch.transpose(D2kT_prev,0,1)

    with torch.no_grad():
        #Init
        image = torch.rand(1,3,320,320).cuda()
        xn1 = image
        xn2 = F.conv2d(input=image, weight=D1k,padding=1)
        #torch.rand(xn1.shape).cuda()
        rhon = 1+1e-2
        rhok = 1
        iter=0

        ### Check D1k,D1kT and D2k,D2kT
        # a= F.conv2d(input=xn1, weight=D2k,padding=1)
        # b= torch.rand(a.shape).cuda()
        # print(torch.sum(a*b))
        # c = F.conv2d(input=b, weight=D2kT,padding=1)
        # print(torch.sum(c*xn1   ))

        while abs(rhok - rhon)/abs(rhok) >= 1e-6 and iter<10000:

            # x  = torch.cat((xn1,xn2),0)
            norm_x = torch.sqrt(torch.norm(xn1,p='fro')**2+torch.norm(xn2,p='fro')**2)
            x1 = xn1/norm_x#torch.norm(x,p='fro')
            x2 = xn2/norm_x#torch.norm(x,p='fro')
            un1,un2 = op_W_k_CPv2(D1k,D2k_prev,x1,x2,alpha,sigma,k-1)
            xn1,xn2 = op_WT_k_CPv2(D1kT,D2kT_prev,un1,un2,alpha,sigma,k-1)
            rhon = rhok
            iter+=1
            rhok = torch.sqrt(torch.norm(xn1,p='fro')**2+torch.norm(xn2,p='fro')**2)#torch.norm(torch.cat((xn1,xn2),0),p='fro')
        gc.collect()
        torch.cuda.empty_cache()
        return torch.sqrt(rhok)



def run(args):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_set = NoisyBSDSDataset(args.root_dir, image_size=args.image_size, sigma=args.sigma)
    test_set = NoisyBSDSDataset(args.root_dir, mode='test', image_size=args.test_image_size, sigma=args.sigma)

    if args.model == 'DnCNN':
        print('Train DnCNN  ---- K= ', args.K,', F=',args.F,'-----')
        net = DnCNN(args.K, F=args.F).to(device)

    elif args.model == 'unfolded_ISTA':
        net = unfolded_ISTA(args.K, F=args.F).to(device)
    elif args.model == 'unfolded_FISTA':
        net = unfolded_FISTA(args.K, F=args.F).to(device)
    elif args.model == 'unfolded_ScCP':
        net = unfolded_ScCP(args.K, F=args.F).to(device)
    elif args.model == 'unfolded_CP':
        net = unfolded_CP(args.K, F=args.F).to(device)
    elif args.model == 'unfolded_CP':
        net = unfolded_CP(args.K, F=args.F).to(device)
    elif args.model == 'compare_diff_K':
        # Comparison different number of layer K
        net_DnCNN = DnCNN(K=args.K, F=args.F).to(device)
        dncnn_param = count_parameters(net_DnCNN,print_table=False)

        net_unfolded_ISTA = unfolded_ISTA(K=int(dncnn_param/54/args.F), F=args.F).to(device)
        # ista_param=count_parameters(net_unfolded_ISTA,print_table=False)
        K_ISTA = int(dncnn_param/54/args.F)

        net_unfolded_FISTA = unfolded_FISTA(K=int(dncnn_param/(54*args.F)), F=args.F).to(device)
        # fista_param=count_parameters(net_unfolded_FISTA,print_table=False)
        K_FISTA = int(dncnn_param/(54*args.F))

        net_unfolded_CP = unfolded_CP(K=int(dncnn_param/(54*args.F+2)), F=args.F).to(device)
        # cp_param=count_parameters(net_unfolded_CP,print_table=False)
        K_CP = int(dncnn_param/(54*args.K+2))

        net_unfolded_ScCP = unfolded_ScCP(K=int(dncnn_param/(54*args.F+2)), F=args.F).to(device)
        # cpv2_param=count_parameters(net_unfolded_ScCP,print_table=False)

        net_unfolded_CP = unfolded_CP(K=int(dncnn_param/(54*args.F+2)), F=args.F).to(device)
        # cpv3_param=count_parameters(net_unfolded_CP,print_table=False)

    elif args.model=='compare_grid':
        grid_ISTA =[]
        grid_FISTA=[]
        grid_CP   =[]
        grid_ScCP=[]
        grid_CP=[]
        for K in [5,13,21,29,37]:
            for F in [13,21,29,37,45]:
                grid_ISTA += [unfolded_ISTA(K=K,F=F).to(device)]
                grid_FISTA+= [unfolded_FISTA(K=K,F=F).to(device)]
                grid_CP += [unfolded_CP(K=K,F=F).to(device)]
                grid_ScCP+= [unfolded_ScCP(K=K,F=F).to(device)]
                grid_CP += [unfolded_CP(K=K,F=F).to(device)]

    elif args.model == 'compare_same_param':
        # This choice serves to compare models having same number of params but we permute K and F
        net_DnCNN = DnCNN(K=9, F=13).to(device)
        # dncnn_param = count_parameters(net_DnCNN,print_table=False)

        net_unfolded_ISTA = unfolded_ISTA(K=13, F=21).to(device)
        # ista_param=count_parameters(net_unfolded_ISTA,print_table=False)

        net_unfolded_FISTA = unfolded_FISTA(K=13, F=21).to(device)
        # fista_param=count_parameters(net_unfolded_FISTA,print_table=False)

        net_unfolded_CP = unfolded_CP(K=13, F=21).to(device)
        # cp_param=count_parameters(net_unfolded_CP,print_table=False)

        net_unfolded_ScCP = unfolded_ScCP(K=13, F=21).to(device)
        # cp_param=count_parameters(net_unfolded_ScCP,print_table=False)

        net_unfolded_CP = unfolded_CP(K=13, F=21).to(device)

        # net_unfolded_ISTA_bis = unfolded_ISTA(K=21, F=13).to(device)
        # net_unfolded_FISTA_bis = unfolded_FISTA(K=21, F=13).to(device)
        # net_unfolded_CP_bis = unfolded_CP(K=21, F=13).to(device)
        # net_unfolded_ScCP_bis = unfolded_ScCP(K=21, F=13).to(device)
        # net_unfolded_CP_bis = unfolded_CP(K=21, F=13).to(device)

    elif args.model == 'compare_diff_F':
        net_DnCNN = DnCNN(K=args.D, F=int(args.F)).to(device)
        dncnn_param = count_parameters(net_DnCNN,print_table=False)
        # net_DnCNN = DnCNN(args.D, F=(-63+np.sqrt(63**2-4*9*args.D*(3+12*args.D)))//(2*9*args.D)).to(device)

        net_unfolded_ISTA = unfolded_ISTA(K=args.D, F=int(dncnn_param/54/args.D)).to(device)
        # ista_param=count_parameters(net_unfolded_ISTA,print_table=False)

        net_unfolded_FISTA = unfolded_FISTA(K=args.D, F=int((dncnn_param-args.D)/54/args.D)).to(device)
        # fista_param=count_parameters(net_unfolded_FISTA,print_table=False)

        net_unfolded_CP = unfolded_CP(K=args.D, F=int((dncnn_param-2*args.D)/(54*args.D))).to(device)
        # cp_param=count_parameters(net_unfolded_CP,print_table=False)

        net_unfolded_ScCP = unfolded_ScCP(K=args.D, F=int((dncnn_param-2*args.D)/(54*args.D))).to(device)
        # cp_param=count_parameters(net_unfolded_ScCP,print_table=False)
    elif args.model == 'compare_custom':
        net_DnCNN = DnCNN(args.K, F=args.F).to(device)
        net_unfolded_ISTA = unfolded_ISTA(args.K, F=args.F).to(device)
        net_unfolded_FISTA = unfolded_FISTA(args.K, F=args.F).to(device)
        count_parameters(net_DnCNN,print_table=False)
        count_parameters(net_unfolded_ISTA,print_table=False)
        count_parameters(net_unfolded_FISTA,print_table=False)
    else:
        raise NameError('Please enter correct choice: dncnn, unfolded_ISTA, unfolded_FISTA,'
                        ,'compare_custom', 'compare_diff_K','compare_diff_F', 'compare_grid', 'compare_discussion_25_jan' )

    # Run one model only
    if args.model !='compare_custom' and args.model !='compare_diff_K' and args.model !='compare_diff_F' and args.model!='compare_same_param'and args.model!='compare_grid':
        countparam = count_parameters(net,print_table=False,namenet=args.model)
        # optimizer
        adam = torch.optim.Adam(net.parameters(), lr=args.lr)
        # stats manager
        stats_manager = DenoisingStatsManager()
        # experiment
        exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                            output_dir=args.output_dir+args.model+'/'+args.model+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+'_sigma_'+str(args.sigma)+'_param_'+str(countparam), perform_validation_during_training=True)
        # run
        exp.run(num_epochs=args.num_epochs)
        for ind_rand in range(199): # Tester sur base de donnee test (1->299)
            print('ID:',ind_rand)
            noise_im = test_set[ind_rand][0]
            org_im   = test_set[ind_rand][1]
            name_im = test_set[ind_rand][2]
            namefile= name_im+'_'+args.model+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+"_epochs_"+str(args.num_epochs)
            save_compare(exp,noisy=noise_im,original=org_im,model=args.model,namefile=namefile)

    elif args.model=='compare_same_param':
        # net_name = ["DnCNN","unfolded_CP","unfolded_ScCP","unfolded_FISTA","unfolded_ISTA","unfolded_CP",
        #             "unfolded_CP_bis","unfolded_ScCP_bis","unfolded_FISTA_bis","unfolded_ISTA_bis","unfolded_CP_bis"]
        # table_net=[net_DnCNN,net_unfolded_CP,net_unfolded_ScCP,net_unfolded_FISTA,net_unfolded_ISTA,net_unfolded_CP,
        #            net_unfolded_CP_bis,net_unfolded_ScCP_bis,net_unfolded_FISTA_bis,net_unfolded_ISTA_bis,net_unfolded_CP_bis]

        # net_name = ["DnCNN","unfolded_CP","unfolded_ScCP"]
        # table_net=[net_DnCNN,net_unfolded_CP,net_unfolded_ScCP]

        net_name = ["DnCNN","unfolded_ScCP","unfolded_FISTA","unfolded_ISTA","unfolded_CP"]
        table_net=[net_DnCNN,net_unfolded_ScCP,net_unfolded_FISTA,net_unfolded_ISTA,net_unfolded_CP]

        for ind_rand in range(1):
            print('ID:',ind_rand)
            noise_im = test_set[ind_rand][0]
            org_im   = test_set[ind_rand][1]
            name_im = test_set[ind_rand][2]
            for i,net in enumerate(table_net):
                countparam = count_parameters(net,print_table=False,namenet=net_name[i])
                adam = torch.optim.Adam(net.parameters(), lr=args.lr)
                stats_manager = DenoisingStatsManager()
                exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                                output_dir=args.output_dir+args.model+'/'+net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+'_sigma_'+str(args.sigma)+'_param_'+str(countparam), perform_validation_during_training=True)

                namefile= name_im+'_'+net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+"_epochs_"+str(args.num_epochs)
                exp.run(num_epochs=args.num_epochs)

                # if net_name[i]=='unfolded_FISTA' :
                #     theta=1
                #     for k in range(1,exp.net.K):
                #         norm = calul_norm_FISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         theta *= norm
                #     print('Theta of FISTA: ', theta)
                #     net.norm =theta
                # elif net_name[i] =='unfolded_ISTA':
                #     # norm = calul_norm_ISTA(net=net,k=1,image=noise_im[None].to(exp.net.device))
                #     theta=1
                #     for k in range(1,exp.net.K):
                #         norm = calul_norm_ISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #     print('Theta of ISTA: ', theta)
                #     net.norm =theta
                # elif net_name[i]=='unfolded_ScCP':
                #      theta=1
                #      for k in range(1,exp.net.K):
                #         norm = calul_norm_CPv2(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #      print('Theta of ScCP: ', theta)
                #      net.norm =theta
                # elif net_name[i]=='unfolded_CP':
                #      theta=1
                #      for k in range(1,exp.net.K):
                #         norm = calul_norm_CPv3(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #      print('Theta of CP: ', theta)
                #      net.norm =theta
                # #         gc.collect()
                # #         torch.cuda.empty_cache()
                save_compare(exp,noisy=noise_im,original=org_im,model=args.model,namefile=namefile)
    elif args.model =='compare_diff_K':
        net_name = ["DnCNN","unfolded_ScCP","unfolded_FISTA","unfolded_ISTA","unfolded_CP","unfolded_CP"]
        table_net=[net_DnCNN,net_unfolded_ScCP,net_unfolded_FISTA,net_unfolded_ISTA,net_unfolded_CP,net_unfolded_CP]

        for ind_rand in range(199):
            print('ID:',ind_rand)
            noise_im = test_set[ind_rand][0]
            org_im   = test_set[ind_rand][1]
            name_im = test_set[ind_rand][2]
            for i,net in enumerate(table_net):
                countparam = count_parameters(net,print_table=False,namenet=net_name[i])
                adam = torch.optim.Adam(net.parameters(), lr=args.lr)
                stats_manager = DenoisingStatsManager()
                exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                                output_dir=args.output_dir+args.model+'/'+net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+'_sigma_'+str(args.sigma)+'_param_'+str(countparam), perform_validation_during_training=True)

                namefile= name_im+'_'+net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+"_epochs_"+str(args.num_epochs)
                exp.run(num_epochs=args.num_epochs)

                # if net_name[i]=='unfolded_FISTA' :
                #     theta=1
                #     for k in range(1,exp.net.K):
                #         norm = calul_norm_FISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         theta *= norm
                #     print('Theta of FISTA: ', theta)
                #     net.norm =theta
                # elif net_name[i] =='unfolded_ISTA':
                #     # norm = calul_norm_ISTA(net=net,k=1,image=noise_im[None].to(exp.net.device))
                #     theta=1
                #     for k in range(1,exp.net.K):
                #         norm = calul_norm_ISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #     print('Theta of ISTA: ', theta)
                #     net.norm =theta
                # elif net_name[i]=='unfolded_ScCP':
                #      theta=1
                #      for k in range(1,exp.net.K):
                #         norm = calul_norm_CPv2(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #      print('Theta of ScCP: ', theta)
                #      net.norm =theta
                # elif net_name[i]=='unfolded_CP':
                #      theta=1
                #      for k in range(1,exp.net.K):
                #         norm = calul_norm_CPv3(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #      print('Theta of CP: ', theta)
                #      net.norm =theta
                # #         gc.collect()
                # #         torch.cuda.empty_cache()
                save_compare(exp,noisy=noise_im,original=org_im,model=args.model,namefile=namefile)
    elif args.model =='compare_diff_F':
        net_name = ["DnCNN","unfolded_ScCP","unfolded_FISTA","unfolded_ISTA","unfolded_CP","unfolded_CP"]
        table_net=[net_DnCNN,net_unfolded_ScCP,net_unfolded_FISTA,net_unfolded_ISTA,net_unfolded_CP,net_unfolded_CP]

        for ind_rand in range(199):
            print('ID:',ind_rand)
            noise_im = test_set[ind_rand][0]
            org_im   = test_set[ind_rand][1]
            name_im = test_set[ind_rand][2]
            for i,net in enumerate(table_net):
                countparam = count_parameters(net,print_table=False,namenet=net_name[i])
                adam = torch.optim.Adam(net.parameters(), lr=args.lr)
                stats_manager = DenoisingStatsManager()
                exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                                output_dir=args.output_dir+args.model+'/'+net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+'_sigma_'+str(args.sigma)+'_param_'+str(countparam), perform_validation_during_training=True)

                namefile= name_im+'_'+net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+"_epochs_"+str(args.num_epochs)
                exp.run(num_epochs=args.num_epochs)

                # if net_name[i]=='unfolded_FISTA' :
                #     theta=1
                #     for k in range(1,exp.net.K):
                #         norm = calul_norm_FISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         theta *= norm
                #     print('Theta of FISTA: ', theta)
                #     net.norm =theta
                # elif net_name[i] =='unfolded_ISTA':
                #     # norm = calul_norm_ISTA(net=net,k=1,image=noise_im[None].to(exp.net.device))
                #     theta=1
                #     for k in range(1,exp.net.K):
                #         norm = calul_norm_ISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #     print('Theta of ISTA: ', theta)
                #     net.norm =theta
                # elif net_name[i]=='unfolded_ScCP':
                #      theta=1
                #      for k in range(1,exp.net.K):
                #         norm = calul_norm_CPv2(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #      print('Theta of ScCP: ', theta)
                #      net.norm =theta
                # elif net_name[i]=='unfolded_CP':
                #      theta=1
                #      for k in range(1,exp.net.K):
                #         norm = calul_norm_CPv3(net=net,k=k,image=noise_im[None].to(exp.net.device))
                #         # print('NORM of W ',k,': ',norm)
                #         theta *= norm
                #      print('Theta of CP: ', theta)
                #      net.norm =theta
                # #         gc.collect()
                # #         torch.cuda.empty_cache()
                save_compare(exp,noisy=noise_im,original=org_im,model=args.model,namefile=namefile)

    else:
        print('Compare grid')
        net_name = ["unfolded_FISTA","unfolded_ISTA","unfolded_CP","unfolded_ScCP","unfolded_CP"]
        # net_name = ["unfoled_FISTA"]
        # net_name = ["unfolded_ISTA"]
        # net_name = ["unfolded_CP"]
        # net_name = ["unfolded_ScCP"]
        # net_name = ["unfolded_CP"]
        ind_rand = 44
        print('ID:',ind_rand)
        noise_im = test_set[ind_rand][0]
        org_im   = test_set[ind_rand][1]

        # for i,grid in enumerate([grid_FISTA]):
        # for i,grid in enumerate([grid_ISTA]):
        # for i,grid in enumerate([grid_CP]):
        # for i,grid in enumerate([grid_ScCP]):
        # for i,grid in enumerate([grid_CP]):

        for i, grid in enumerate([grid_FISTA,grid_ISTA,grid_CP,grid_ScCP,grid_CP]):
            for net in grid:
                countparam = count_parameters(net,print_table=False,namenet=net_name[i]+'_F_'+str(net.F)+'_K_'+str(net.K))#
                adam = torch.optim.Adam(net.parameters(), lr=args.lr)
                stats_manager = DenoisingStatsManager()
                exp = nt.Experiment(net, train_set, test_set, adam, stats_manager, batch_size=args.batch_size,
                                output_dir=args.output_dir+args.model+'/'+net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+'_sigma_'+str(args.sigma)+'_param_'+str(countparam), perform_validation_during_training=True)

                namefile= net_name[i]+"_F"+str(net.F)+"_K"+str(net.K)+"_batchsize"+str(args.batch_size)+"_epochs_"+str(args.num_epochs)
                exp.run(num_epochs=args.num_epochs)
                if net_name[i]=='unfolded_FISTA' :
                #     countparam = count_parameters(net,pr=True)
                #     norm = calul_norm_FISTA(net=net,k=1,image=noise_im[None].to(exp.net.device))
                    theta=1
                    for k in range(1,exp.net.K):
                        norm = calul_norm_FISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                        # print('NORM of W ',k,': ',norm)
                        theta *= norm
                    print('Theta of FISTA: ', theta.cpu().numpy())
                    net.norm =theta.cpu().numpy()

                elif net_name[i] =='unfolded_ISTA':
                    # norm = calul_norm_ISTA(net=net,k=1,image=noise_im[None].to(exp.net.device))
                    theta=1
                    for k in range(1,exp.net.K):
                        norm = calul_norm_ISTA(net=net,k=k,image=noise_im[None].to(exp.net.device))
                        # print('NORM of W ',k,': ',norm)
                        theta *= norm
                    print('Theta of ISTA: ', theta.cpu().numpy())
                    net.norm =theta.cpu().numpy()
                elif net_name[i]=='unfolded_ScCP':
                     theta=1
                     for k in range(1,exp.net.K):
                        norm = calul_norm_CPv2(net=net,k=k,image=noise_im[None].to(exp.net.device))
                        # print('NORM of W ',k,': ',norm)
                        theta *= norm
                     print('Theta of ScCP: ', theta.cpu().numpy())
                     net.norm =theta.cpu().numpy()
                elif net_name[i]=='unfolded_CP':
                     theta=1
                     for k in range(1,exp.net.K):
                        norm = calul_norm_CPv3(net=net,k=k,image=noise_im[None].to(exp.net.device))
                        # print('NORM of W ',k,': ',norm)
                        theta *= norm
                     print('Theta of CP: ', theta.cpu().numpy())
                     net.norm =theta.cpu().numpy()
                #         gc.collect()
                #         torch.cuda.empty_cache()
                save_compare(exp,noisy=noise_im,model=net_name[i],original=org_im,namefile=namefile)
if __name__ == '__main__':

    args = parse()
    run(args)
