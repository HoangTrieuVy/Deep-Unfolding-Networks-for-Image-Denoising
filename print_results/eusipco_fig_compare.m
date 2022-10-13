clear all;
clc;
% addpath('cbrewer')
close all;


% -- Noise level 50: Comparison different layer but similar number of param for all method

colormap(gray);
iter=1;
files = dir('../dataset/BSDS500/data/images/test_noise_50/*.mat');

for file = files'
if strcmp('10081.mat',file.name) %|| strcmp('103029.mat',file.name) ||strcmp('106005.mat',file.name) || strcmp('106024.mat',file.name) ||strcmp('106047.mat',file.name)
     disp(file.name);
 
    if strcmp('10081.mat',file.name)
        zoomx=135;
        zoomy=225;
%     elseif strcmp('103029.mat',file.name)
%         zoomx=100;
%         zoomy=195;
%     elseif strcmp('106005.mat',file.name)
%         zoomx=100;
%         zoomy=160;
%     elseif strcmp('106024.mat',file.name)
%         zoomx=100;
%         zoomy=160;
%     elseif strcmp('106047.mat',file.name)
%         zoomx=100;
%         zoomy=160;
%     else
%        zoomx=160;
%         zoomy=160; 
    end
    
name_im= strrep(file.name,'.mat','');



std_noise = 50;
batchsize=10;
epoch=500;

dncnn = 'DnCNN';
F_dncnn= 13;
K_dncnn= 9;
ista = 'unfolded_ISTA';
% fista = 'unfolded_FISTA';
cpv2 = 'unfolded_CP_v2';
cpv3 = 'unfolded_CP_v3';
F_unroll= 21;
K_unroll= 13;

load(['../src/out/',dncnn,name_im,'.jpg','_',dncnn,'_F',num2str(F_dncnn),'_K',num2str(K_dncnn),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
nb_dncnn = nb_param;
PSNR_dncnn=PSNR_test;
eval_dncnn = eval_loss;
load(['../src/out/',ista,name_im,'.jpg','_',ista,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
nb_ista = nb_param;
PSNR_ista=PSNR_test;
eval_ista = eval_loss;
% load(['../src/out/',fista,name_im,'.jpg','_',fista,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
% nb_fista = nb_param;
load(['../src/out/',cpv3,name_im,'.jpg','_',cpv3,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
nb_cpv3 = nb_param;
PSNR_cpv3 = PSNR_test;
eval_cpv3 = eval_loss;

load(['../src/out/',cpv2,name_im,'.jpg','_',cpv2,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
nb_cpv2 = nb_param;
PSNR_cpv2=PSNR_test;
eval_cpv2 = eval_loss;
load(['../src/out/','clean_',name_im,'.jpg_',dncnn,'_F',num2str(F_dncnn),'_K',num2str(K_dncnn),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
clean= u_rec;
load(['../src/out/','noisy_',name_im,'.jpg_',dncnn,'_F',num2str(F_dncnn),'_K',num2str(K_dncnn),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
noisy= double(u_rec);

figure(iter)
subplot(3,8,1);imagesc(clean);axis image off; title 'Original';
subplot(3,8,9);imagesc(clean(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; title 'Original';
subplot(3,8,2);imagesc(noisy);axis image off; title({'Degraded';['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),noisy),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),noisy),'%3.3f')]} );
subplot(3,8,10);imagesc(noisy(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; title 'Original';


load(['../src/out/','denoised_',name_im,'.jpg_',dncnn,'_F',num2str(F_dncnn),'_K',num2str(K_dncnn),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
resim_dncnn= double(u_rec);


load(['../src/out/','denoised_',name_im,'.jpg_',ista,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
resim_ista= double(u_rec);
% load(['../src/out/','denoised_',name_im,'.jpg_',fista,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
% resim_fista= double(u_rec);
load(['../src/out/','denoised_',name_im,'.jpg_',cpv2,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
resim_cpv2= double(u_rec);
load(['../src/out/','denoised_',name_im,'.jpg_',cpv3,'_F',num2str(F_unroll),'_K',num2str(K_unroll),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat'])
resim_cpv3= double(u_rec);

% subplot(3,8,4);imagesc(resim_dncnn);axis image off; title({dncnn;['batchsize:',num2str(batchsize)];['nb param:', num2str(nb_dncnn),' F:',num2str(F_dncnn),' K:',num2str(K_dncnn)];['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),resim_dncnn),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_dncnn),'%3.2f')]} );
% subplot(3,8,5);imagesc(resim_ista);axis image off; title({'unfolded ISTA';['batchsize:',num2str(batchsize)];['nb param:', num2str(nb_ista),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),resim_ista),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_ista),'%3.2f')]} );
% % subplot(3,8,6);imagesc(resim_fista);axis image off; title({fista;['batchsize:',num2str(batchsize)];['nb param:', num2str(nb_fista),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),resim_fista),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_fista),'%3.2f')]} );
% subplot(3,8,7);imagesc(resim_cpv3);axis image off; title({'CP without SC';['batchsize:',num2str(batchsize)];['nb param:', num2str(nb_cpv3),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),resim_cpv3),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_cpv3),'%3.2f')]} );
% subplot(3,8,8);imagesc(resim_cpv2);axis image off; title({'CP with SC';['batchsize:',num2str(batchsize)];['nb param:', num2str(nb_cpv2),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['std:',num2str(std_noise)];['PSNR: ' num2str(plpsnr(double(clean),resim_cpv2),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_cpv2),'%3.2f')]} );

subplot(3,8,4);imagesc(resim_dncnn);axis image off; title({dncnn;['nb param:', num2str(nb_dncnn),' F:',num2str(F_dncnn),' K:',num2str(K_dncnn)];['PSNR: ' num2str(plpsnr(double(clean),resim_dncnn),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_dncnn),'%3.2f')]} );
subplot(3,8,5);imagesc(resim_ista);axis image off; title({'unfolded ISTA';['nb param:', num2str(nb_ista),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['PSNR: ' num2str(plpsnr(double(clean),resim_ista),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_ista),'%3.2f')]} );
% subplot(3,8,6);imagesc(resim_fista);axis image off; title({fista;['nb param:', num2str(nb_fista),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['PSNR: ' num2str(plpsnr(double(clean),resim_fista),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_fista),'%3.2f')]} );
subplot(3,8,7);imagesc(resim_cpv3);axis image off; title({'CP without SC';['nb param:', num2str(nb_cpv3),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['PSNR: ' num2str(plpsnr(double(clean),resim_cpv3),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_cpv3),'%3.2f')]} );
subplot(3,8,8);imagesc(resim_cpv2);axis image off; title({'CP with SC';['nb param:', num2str(nb_cpv2),' F:',num2str(F_unroll),' K:',num2str(K_unroll)];['PSNR: ' num2str(plpsnr(double(clean),resim_cpv2),'%3.2f') '/ SSIM: ' num2str(ssim(double(clean),resim_cpv2),'%3.2f')]} );

subplot(3,8,12);imagesc(resim_dncnn(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off;
subplot(3,8,13);imagesc(resim_ista(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; 
% subplot(3,8,14);imagesc(resim_fista(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; 
subplot(3,8,15);imagesc(resim_cpv3(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; 
subplot(3,8,16);imagesc(resim_cpv2(zoomx-30:zoomx+30,zoomx-30:zoomx+30,:));axis image off; 


iter = iter +1;
h=hsv(18);
figure(100)
semilogy(PSNR_dncnn,"black",'LineWidth',2);hold on;
semilogy(PSNR_ista,"--g",'LineWidth',2);hold on;
semilogy(PSNR_cpv2,"-r",'LineWidth',2);hold on;
semilogy(PSNR_cpv3,"--r",'LineWidth',2);hold on;

legend({dncnn,"unfolded ISTA","unfolded ScCP"},'Location','southeast')
title('PNSR test');
ylim([10,27]);
set(gca,'FontSize',17)
xlabel('Epochs')
ylabel('Average PSNR (dB)')

figure(101)
load(['../src/out/',dncnn,name_im,'.jpg','_',dncnn,'_F',num2str(F_dncnn),'_K',num2str(K_dncnn),'_batchsize',num2str(batchsize),'_epochs_',num2str(epoch),'.mat']);
plot(eval_dncnn,"black");hold on;
plot(eval_ista,"--g",'LineWidth',2);hold on;
plot(eval_cpv2,"-r",'LineWidth',2);hold on;
plot(eval_cpv3,"--r",'LineWidth',2);hold on;

legend({dncnn,"unfolded ISTA","unfolded ScCP"},'Location','southeast')
title('loss train');
set(gca,'FontSize',17)
xlabel('Epochs')
ylabel('MSE loss')

 end
end
 