%% load bias grid data
close all
load('data/grid.mat')

%% interpolate
bs_interp = 9;
l2_interp = 5;
[Bsfine,L2fine] = meshgrid(linspace(min(Bs),max(Bs),bs_interp), ...
    linspace(min(L2),max(L2),l2_interp));
scores_fine = griddata(Bs,L2,scores,Bsfine,L2fine, 'cubic');

%% display 2D
subplot(1,2,1)
%colormap(hsv2rgb([0.6*linspace(1,0,512)'.^10 ones(512,1) ones(512,1)]));
colormap('jet')
img = full
img(find(img>1.3)) = max(img(img<=1.3))
%img(isnan(img)) = max(max(img));
imagesc(l2_set, bs_set, img');
title('Grid search')
xlim([min(L2), max(L2)]); ylim([min(Bs), max(Bs)]);
colorbar EastOutside
xlabel('L2 (bias vector regularization)');
ylabel('Standard deviation of initialization of bias vectors');

%% interpolate
bs_interp = 33;
l2_interp = 17;
[Bsfine,L2fine] = meshgrid(linspace(min(Bs),max(Bs),bs_interp), ...
    linspace(min(L2),max(L2),l2_interp));
scores(find(scores>1.3)) = NaN
scores_fine = griddata(Bs,L2,scores,Bsfine,L2fine, 'natural');

%% display 2D

subplot(1,2,2)
%colormap(hsv2rgb([0.6*linspace(1,0,512)'.^10 ones(512,1) ones(512,1)]));
colormap('jet')
img = scores_fine;
img(isnan(img)) = max(max(img));
imagesc(L2fine(:,1), Bsfine(1,:), img');
title('Interpolation by factor 4')
xlim([min(L2), max(L2)]); ylim([min(Bs), max(Bs)]);
colorbar EastOutside
xlabel('L2 (bias vector regularization)');
ylabel('Standard deviation of initialization of bias vectors');
