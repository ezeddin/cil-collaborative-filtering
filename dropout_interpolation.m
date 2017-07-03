close all
x = [0.4	0.98299;
0.5	0.98051;
0.55	0.97988;
0.6	0.97968;
0.65 0.97996;
0.7	0.98073];

plot(x(:,1), x(:,2), 'x');

p = polyfit(x(:,1), x(:,2), 2);
x1 = linspace(0.4,0.7);
f1 = polyval(p,x1);

hold on;
plot(x1, f1);
plot(x1(find(f1 == min(f1))),min(f1),'o');
format long
x1(find(f1 == min(f1)))
min(f1)
format short