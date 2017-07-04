close all

% data from kaggle submissions
x = [0.4 0.98299;
0.5 0.98051;
0.55 0.97988;
0.6 0.97968;
0.65 0.97996;
0.7 0.98073];

% plot fitted curve
polynome_order = 2;
p = polyfit(x(:,1), x(:,2), polynome_order);
x_fit = linspace(min(x(:,1)), max(x(:,1)));
y_fit = polyval(p, x_fit);
h_fit_curve = plot(x_fit, y_fit, '-r', 'linewidth', 2); hold on;
h_data_points = plot(x(:,1), x(:,2), 'xb', 'linewidth', 2);

% plot original data points and interpolated expected minimum score
h_minimum = plot(x_fit(y_fit == min(y_fit)),min(y_fit),'or', 'linewidth', 2);
format long
x_fit(y_fit == min(y_fit))
min(y_fit)
format short

% graph labels
xlabel('dropout rate')
ylabel('Kaggle score')
legend([h_data_points,h_fit_curve,h_minimum], 'Kaggle score results','Quadratic fit curve','Minimum of fit curve')
title('Best score for different dropout rates')