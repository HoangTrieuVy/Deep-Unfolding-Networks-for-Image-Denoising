clear all;
clc;
% addpath('cbrewer')
close all;


% -- Noise level 50: Comparison different layer but similar number of param for all method

colormap(gray);
iter=1;
files = dir('../dataset/BSDS500/data/images/test_noise_50/*.mat');

for file = files'
if strcmp('10081.mat',file.name) || strcmp('103029.mat',file.name) ||strcmp('106005.mat',file.name) || strcmp('106024.mat',file.name) ||strcmp('106047.mat',file.name)
     disp(file.name);
 
    if strcmp('10081.mat',file.name)
        zoomx=135;
        zoomy=225;
    elseif strcmp('103029.mat',file.name)
        zoomx=100;
        zoomy=195;
    elseif strcmp('106005.mat',file.name)
        zoomx=100;
        zoomy=160;
    elseif strcmp('106024.mat',file.name)
        zoomx=100;
        zoomy=160;
    elseif strcmp('106047.mat',file.name)
        zoomx=100;
        zoomy=160;
    else
       zoomx=160;
        zoomy=160; 
    end 
    
name_im= strrep(file.name,'.mat','');
name_model = 'unfolded_FISTA';
% name_model = 'unfolded_ISTA';
% name_model = 'unfolded_CP_v2';

F= 21;
K= 13;

% name_model = 'DnCNN';
% F= 13;
% K= 9;

std_noise = 50;
batchsize=10;
epoch=500;


load(['../src/out/',name_model,name_im,'.jpg','_',name_model,'_F',num2str(F),'_K',num2str(K),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
nb = nb_param;
load(['../src/out/','clean_',name_im,'.jpg_',name_model,'_F',num2str(F),'_K',num2str(K),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
clean= u_rec;
[r,c,channel]= size(clean);

figure(iter)
subplot(3,8,1);imagesc(clean);axis image off; title 'Original';
subplot(3,8,9);imagesc(clean(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; title 'Original';


load(['../src/out/','noisy_',name_im,'.jpg_',name_model,'_F',num2str(F),'_K',num2str(K),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
noisy= double(u_rec);
subplot(3,8,2);imagesc(noisy);axis image off; title({'Degraded';['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),noisy),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),noisy),'%3.3f')]} );
subplot(3,8,10);imagesc(noisy(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; title 'Original';


load(['../src/out/','denoised_',name_im,'.jpg_',name_model,'_F',num2str(F),'_K',num2str(K),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
resim= double(u_rec);
subplot(3,8,5);imagesc(resim);axis image off; title({name_model;['batchsize:',num2str(batchsize)];['nb param:', num2str(nb),' F:',num2str(F),' K:',num2str(K)];['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),resim),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim),'%3.2f')]} );
subplot(3,8,13);imagesc(resim(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; 

iter = iter +1;

h=hsv(18);
figure(100)
load(['../src/out/',name_model,name_im,'.jpg','_',name_model,'_F',num2str(F),'_K',num2str(K),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
semilogy(PSNR_test,'Color',h(1,:));hold on;

legend({name_model},'Location','southeast')
title('PNSR test');
ylim([10,27]);
set(gca,'FontSize',17)
xlabel('Epochs')
ylabel('Average PSNR (dB)')

figure(101)
load(['../src/out/',name_model,name_im,'.jpg','_',name_model,'_F',num2str(F),'_K',num2str(K),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
plot(val_loss,'Color',h(1,:));hold on;
legend(name_model)
title('loss train');
set(gca,'FontSize',17)
xlabel('Epochs')
ylabel('MSE loss')
end
end