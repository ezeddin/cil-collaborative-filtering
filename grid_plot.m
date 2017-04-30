%% load data
close all
load('data/grid.mat')
k = double(k);

%% preprocess
% kill point with coords 2,0.0304
[a b]=find(k==2);
[c d]=find(l==0.0304);
ind = intersect(b,d);
k(ind) = [];
l(ind) = [];
scores(ind) = [];

%% interpolate
k_interp = max(k)-min(k)+1;
l_interp = 40;
[Kfine,Lfine] = meshgrid(linspace(min(k),max(k),k_interp), ...
    linspace(min(l),max(l),l_interp));
scores_fine = griddata(k,l,scores,Kfine,Lfine, 'cubic');

%% display 2D
subplot(1,2,1)
colormap(hsv2rgb([0.6*linspace(1,0,512)'.^10 ones(512,1) ones(512,1)]));
img = scores_fine;
img(isnan(img)) = max(max(img));
imagesc(Lfine(:,1), Kfine(1,:), img');
xlim([min(l), max(l)]); ylim([min(k), max(k)]);
colorbar EastOutside
xlabel('L');
ylabel('K');

%% display 3D
subplot(1,2,2);
plot3(k,l,scores, '+r', 'MarkerSize',1, 'LineWidth',0.5);
hold on
surf(Kfine,Lfine,scores_fine);
axis vis3d;
view([124 40]);
li = light('Position',[0, 4.5, 3.5]);
lighting phong;
shading interp;
xlabel('K');
ylabel('L')
zlabel('Score');
hold off