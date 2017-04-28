load('data/grid.mat')
close all

subplot(1,2,1)
colormap default
surf(k,l,scores')
colorbar
xlabel('K')
ylabel('L')
zlabel('score')

subplot(1,2,2)
colormap(jet)
imagesc(k, l, scores')
xlabel('K')
ylabel('L')
colorbar