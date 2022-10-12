clear all
close all
clc

load param_total.mat

c = 20:1:119;
d = 2:1:29;


h = hsv(100);
% ISTA
figure(1)
for ii = 1:length(c)
subplot(241);plot(squeeze(d),squeeze(param_total(ii,:,1)),'Color',h(ii,:)); hold on;grid on; title 'ISTA'
end
%legend('20','40','50','60','70','80','90');

for ii = 1:length(d)
subplot(245);plot(squeeze(c),squeeze(param_total(:,ii,1)),'Color',h(ii,:)); hold on;grid on;
end
%legend('10','20','40','50','60','70','80','90');

% FISTA
for ii = 1:length(c)
subplot(242);plot(squeeze(d),squeeze(param_total(ii,:,2)),'Color',h(ii,:)); hold on;grid on; title 'FISTA'
end

for ii = 1:length(d)
subplot(246);plot(squeeze(c),squeeze(param_total(:,ii,2)),'Color',h(ii,:)); hold on;grid on;
end

% DnCNN
for ii = 1:length(c)
subplot(243);plot(squeeze(d),squeeze(param_total(ii,:,3)),'Color',h(ii,:)); hold on;grid on; title 'DnCNN'
end
ylim([0 1e5])

for ii = 1:length(d)
subplot(247);plot(squeeze(c),squeeze(param_total(:,ii,3)),'Color',h(ii,:)); hold on;grid on;
end
ylim([0 1e5])

% CP
for ii = 1:length(c)
subplot(244);plot(squeeze(d),squeeze(param_total(ii,:,4)),'Color',h(ii,:)); hold on;grid on; title 'CP'
end

for ii = 1:length(d)
subplot(248);plot(squeeze(c),squeeze(param_total(:,ii,4)),'Color',h(ii,:)); hold on;grid on;
end

figure(2)
subplot(121); imagesc(param_total(:,:,1),[0 2e5]);
subplot(122); imagesc(param_total(:,:,3),[0 2e5]);

% Choice of config from DnCNN
nb_param = param_total(4,8,3);
%param_total(2,2,3)
coef = 0.01;
method=1;
[ind1,ind2] = find(param_total(:,:,method)>(1-coef)*nb_param & param_total(:,:,method)<(1+coef)*nb_param);
c(ind1)
d(ind2)

