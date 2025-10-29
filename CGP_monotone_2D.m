gpml_root = 'GPML';
run(fullfile(gpml_root, 'startup.m'));
clear; clc; close all; rng(1);

n  = 50;
x1 = rand(n,1);
x2 = rand(n,1);
y  = [x1 x2];

f_true = @(u,v) 1./(1+exp(-10*(u-0.4))) + 1./(1+exp(-10*(v-0.6)));
%f_true = @(u,v) sin(2*pi*x1) + 2*x2;
sigma   = 0.5;
epsilon = sigma * randn(n,1);
X       = f_true(x1, x2) + epsilon;

g = linspace(0,1,100);
[yt1, yt2] = meshgrid(g, g);
yt = [yt1(:) yt2(:)];
nstar = size(yt,1);

meanfunc = {@meanConst};   hyp.mean = 0;
covfunc  = {@covSEard};    hyp.cov  = [0, 0, 0];
likfunc  = {@likGauss};    hyp.lik  = -inf;

[hyp_sol, ~, ~] = minimize(hyp, @gp, -100, @infExact, meanfunc, covfunc, [], y, X);

dj = 2;
yt_s = GP_Multivariate_solCGP(hyp_sol, X, y, yt, dj);

Zval = Gibbs_MultivariateGP_CGP(hyp_sol, X, y, yt, yt_s, dj);
validCols = all(~isnan(Zval), 1);
Zval = Zval(:, validCols); 

mu_cgp = mean(Zval, 2);

true_on_grid = f_true(yt1, yt2);
mse = mean((mu_cgp - true_on_grid(:)).^2);
fprintf('Test-grid MSE = %.6f\n', mse);
Mu = reshape(mu_cgp, size(yt1));
Real = reshape(true_on_grid, size(yt1));
figure;
mesh(yt1, yt2, Mu, 'FaceColor', 'b', 'EdgeColor', 'b');
hold on;
mesh(yt1, yt2, Real, 'FaceColor', 'r', 'EdgeColor', 'r');
hold off;
legend('Predicted mean', 'True function', 'Location', 'Best');
