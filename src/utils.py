import numpy as np
import matplotlib.pyplot as plt
import nntools as nt
import torch
from torch import nn
from skimage.metrics import structural_similarity as ssim
import scipy.io
import os
from prettytable import PrettyTable

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
    # print(namenet+" has total Trainable Params: ",total_params)
    return total_params

def PSNR_np(I,Iref):
    temp=I.ravel()
    tempref=Iref.ravel()
    NbP=I.size
    EQM=np.sum((temp-tempref)**2)/NbP
    b=np.max(np.abs(tempref))**2
    return 10*np.log10(b/EQM)

def PSNR(img1, img2):
    #mse = torch.mean((img1 - img2) ** 2)
    # return 10 * torch.log10(255.0 / torch.sqrt(mse))
    img1 = img1.to('cpu').numpy()
    img1 = np.moveaxis(img1, [0, 1, 2], [2, 0, 1])
    img2 = img2.to('cpu').numpy()
    img2 = np.moveaxis(img2, [0, 1, 2], [2, 0, 1])
    # n = img1.shape[0] * img1.shape[1] * img1.shape[2]
    # print(10*np.log10(np.max(img1)**2/(np.mean(np.abs(img1-img2)**2))))
    return 10*np.log10(np.max(img1)**2/(np.mean(np.abs(img1-img2)**2)))

def SSIM(img1,img2):
    img1 = img1.to('cpu').numpy()
    img1 = np.moveaxis(img1, [0, 1, 2], [2, 0, 1])
    img2 = img2.to('cpu').numpy()
    img2 = np.moveaxis(img2, [0, 1, 2], [2, 0, 1])
    return ssim(img1, img2,multichannel=True)


def save_im(image,namefile=''):
    image = image.to('cpu').numpy()
    image = np.moveaxis(image, [0, 1, 2], [2, 0, 1])
    scipy.io.savemat('out/'+namefile+'.mat', dict(u_rec=image))

def save_compare(exp, noisy,original,model,namefile=''):

    with torch.no_grad():
        denoised = exp.net(noisy[None].to(exp.net.device))[0]
        
    save_im(noisy,namefile='noisy_'+namefile)
    save_im(denoised, namefile='denoised_'+namefile)
    save_im(original, namefile='clean_'+namefile)

    scipy.io.savemat('out/'+model+namefile+'.mat',
                     dict(eval_loss=[exp.history[k][1]['loss']
                     for k in range(exp.epoch)],
                          val_loss=[exp.history[k][0]['loss']
                     for k in range(exp.epoch)],
                          PSNR_train=[exp.history[k][0]['PSNR']
                     for k in range(exp.epoch)],
                          PSNR_test=[exp.history[k][1]['PSNR']
                     for k in range(exp.epoch)],
                          CT=[exp.history[k][2]
                     for k in range(exp.epoch)],
                          nb_param=count_parameters(exp.net,print_table=False),
                          K=exp.net.K,
                          F=exp.net.F,
                          norm_net=exp.net.norm_net))

class NNRegressor(nt.NeuralNetwork):

    def __init__(self):
        super(NNRegressor, self).__init__()
        self.mse = nn.MSELoss()

    def criterion(self, y, d):
        return self.mse(y, d)

class DenoisingStatsManager(nt.StatsManager):

    def __init__(self):
        super(DenoisingStatsManager, self).__init__()

    def init(self):
        super(DenoisingStatsManager, self).init()
        self.running_psnr = 0
        self.running_ssim = 0

    def accumulate(self, loss, x, y, d):
        super(DenoisingStatsManager, self).accumulate(loss, x, y, d)
        # n = x.shape[0] * x.shape[1] * x.shape[2] * x.shape[3]
        # self.running_psnr += 10*torch.log10(4*n/(torch.norm(y-d)**2))
        # self.running_ssim += SSIM(y,d)
        self.running_psnr= PSNR(d,y)

    def summarize(self):
        loss = super(DenoisingStatsManager, self).summarize()
        # psnr = self.running_psnr / self.number_update
        psnr = self.running_psnr
        # ssim = self.running_ssim / self.number_update
        # return {'loss': loss, 'PSNR': psnr.cpu(),'SSIM': ssim.cpu()}
        # return {'loss': loss, 'PSNR': psnr.cpu()}
        return {'loss': loss, 'PSNR': psnr}
