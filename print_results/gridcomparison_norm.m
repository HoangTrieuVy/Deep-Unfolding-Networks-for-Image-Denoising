clear all;
clc;
close all;
addpath('src/cbrewer');
map = cbrewer('seq', 'BuPu', 20);
batchsize=4;
mat_norm_validation = zeros(5,5);
tabD= [5,13,21,29,37];
tabC= [13,21,29,37,45];

load(['out/compare_grid_01_fev/clean_image.mat'])
clean= u_rec;
load(['out/compare_grid_01_fev/noisy_image.mat'])

for i=1:5 
    for j=1:5 
        try
        load(['out/compare_grid_01_fev/unfolded_ISTA_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
        load(['out/compare_grid_01_fev/info_unfolded_ISTA_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
%         mat_norm_validation_ISTA(i,j) = PSNR_test(end);
        mat_norm_validation_ISTA(i,j) = log(norm_net);
        catch
           mat_norm_validation_ISTA(i,j) = NaN;
        end
end
end

figure(1)
imagesc(tabD,tabC,mat_norm_validation_ISTA,[0,50]);
ax=gca;
ax.FontSize = 20;
xticks([5,13,21,29,37])
xticklabels({'5','13','21','29','37'})
yticks([13,21,29,37,45])
yticklabels({'13','21','29','37','45'})
colormap(flipud(parula(15))); 
axis xy;
xlabel('$ K $', 'interpreter','latex','FontSize', 30);  
ylabel('$ \vert F \vert $','interpreter','latex','FontSize', 30);  


for i=1:5 
    for j=1:5 
        try
        load(['out/compare_grid_01_fev/unfolded_FISTA_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
        load(['out/compare_grid_01_fev/info_unfolded_FISTA_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
%         mat_norm_validation_FISTA(i,j) = PSNR_test(end);
          mat_norm_validation_FISTA(i,j) = log(norm_net);
        catch
           mat_norm_validation_FISTA(i,j) = NaN;
        end
end
end

figure(2)
imagesc(tabD,tabC,mat_norm_validation_FISTA,[0,50]);
ax=gca;
ax.FontSize = 20;
xticks([5,13,21,29,37])
xticklabels({'5','13','21','29','37'})
yticks([13,21,29,37,45])
yticklabels({'13','21','29','37','45'})
colormap(flipud(parula(15))); 
axis xy;
xlabel('$ K $', 'interpreter','latex','FontSize', 30);  
ylabel('$ \vert F \vert $','interpreter','latex','FontSize', 30);  

% for i=1:5 
%     for j=1:5 
%         try
%         load(['out/compare_grid_01_fev/unfolded_CP_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
%         load(['out/compare_grid_01_fev/info_unfolded_CP_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
% %         mat_norm_validation_CP(i,j) = PSNR_test(end);
% mat_norm_validation_CP(i,j) = log(norm_net);
%         catch
%            mat_norm_validation_CP(i,j) = NaN;
%         end
% end
% end

% figure(3)
% imagesc(tabD,tabC,mat_norm_validation_CP,[0,50]);
% ax=gca;
% ax.FontSize = 20;
% xticks([5,13,21,29,37])
% xticklabels({'5','13','21','29','37'})
% yticks([13,21,29,37,45])
% yticklabels({'13','21','29','37','45'})
% colormap(flipud(parula(15))); 
% 
% axis xy;
% xlabel('$K$', 'FontSize', 30); 
% ylabel('$\mid \mathbb{F}\mid $', 'interpreter','latex','FontSize', 30);  
% 
for i=1:5 
    for j=1:5 
        try
        load(['out/compare_grid_01_fev/unfolded_CP_v2_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
        load(['out/compare_grid_01_fev/info_unfolded_CP_v2_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
%         mat_norm_validation_CP_v2(i,j) = PSNR_test(end);
        mat_norm_validation_CP_v2(i,j) = log(norm_net);
        catch
           mat_norm_validation_CP_v2(i,j) = NaN;
        end
end
end

figure(4)
imagesc(tabD,tabC,mat_norm_validation_CP_v2,[0,50]);
ax=gca;
ax.FontSize = 20;
xticks([5,13,21,29,37])
xticklabels({'5','13','21','29','37'})
yticks([13,21,29,37,45])
yticklabels({'13','21','29','37','45'})
colormap(flipud(parula(15))); 
axis xy;
xlabel('$ K $', 'interpreter','latex','FontSize', 30);  
ylabel('$ \vert F \vert $','interpreter','latex','FontSize', 30);  

for i=1:5 
    for j=1:5 
        try
        load(['out/compare_grid_01_fev/unfolded_CP_v3_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
        load(['out/compare_grid_01_fev/info_unfolded_CP_v3_C',num2str(tabC(i)),'_D',num2str(tabD(j)),'_batchsize',num2str(batchsize),'.mat']);
%         mat_norm_validation_CP_v3(i,j) = PSNR_test(end);
         mat_norm_validation_CP_v3(i,j) = log(norm_net);
        catch
           mat_norm_validation_CP_v3(i,j) = NaN;
        end
end
end

figure(5)
imagesc(tabD,tabC,mat_norm_validation_CP_v3,[0,50]);
ax=gca;
ax.FontSize = 20;
xticks([5,13,21,29,37])
xticklabels({'5','13','21','29','37'})
yticks([13,21,29,37,45])
yticklabels({'13','21','29','37','45'})
colormap(flipud(parula(15))); 
axis xy;
xlabel('$ K $', 'interpreter','latex','FontSize', 30);  
ylabel('$ \vert F \vert $','interpreter','latex','FontSize', 30);  
