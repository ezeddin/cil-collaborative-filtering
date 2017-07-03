%% load bias grid data
close all
load('data/grid.mat')

%% interpolate
uvs_interp = 9;
uvm_interp = 5;
[Uvsfine,Uvmfine] = meshgrid(linspace(min(UVs),max(UVs),uvs_interp), ...
    linspace(min(UVm),max(UVm),uvm_interp));
scores_fine = griddata(UVs,UVm,scores,Uvsfine,Uvmfine, 'cubic');

%% display 2D
subplot(1,2,1)
%colormap(hsv2rgb([0.6*linspace(1,0,512)'.^10 ones(512,1) ones(512,1)]));
colormap('jet')
img = scores_fine;
img(find(img>1.3)) = NaN
img(isnan(img)) = max(max(img));
imagesc(Uvmfine(:,1), Uvsfine(1,:), img');
title('Grid search')
xlim([min(UVm), max(UVm)]); ylim([min(UVs), max(UVs)]);
colorbar EastOutside
xlabel('Mean of initialization of U and V matrix');
ylabel('Standard deviation of initialization of U and V matrix');

%% interpolate
uvs_interp = 33;
uvm_interp = 17;
[Uvsfine,Uvmfine] = meshgrid(linspace(min(UVs),max(UVs),uvs_interp), ...
    linspace(min(UVm),max(UVm),uvm_interp));
scores_fine = griddata(UVs,UVm,scores,Uvsfine,Uvmfine, 'linear');

%% display 2D

subplot(1,2,2)
colormap(hsv2rgb([0.6*linspace(1,0,512)'.^10 ones(512,1) ones(512,1)]));
%colormap('jet')
img = scores_fine;
img(find(img>1.3)) = NaN
img(isnan(img)) = max(max(img));
imagesc(Uvmfine(:,1), Uvsfine(1,:), img');
title('Interpolation by factor 4')
xlim([min(UVm), max(UVm)]); ylim([min(UVs), max(UVs)]);
colorbar EastOutside
xlabel('Mean of initialization of U and V matrix');
ylabel('Standard deviation of initialization of U and V matrix');