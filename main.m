gpml_root1 = 'GPML';
run(fullfile(gpml_root1, 'startup.m'));
gpml_root2 = 'gpstuff';
run(fullfile(gpml_root2, 'startup.m'));
clear; clc;
rng(2025, 'twister');

data_root = 'datasets_stepwise_monotone';

Ttest  = readtable(fullfile(data_root, 'test_grid.csv'));
xt     = Ttest.xt(:);
y_true = Ttest.y_true(:);

n_trials = 25;

results = nan(n_trials, 5);

for i = 1:n_trials
    f1 = fullfile(data_root, sprintf('train_run%02d.csv', i));
    if ~isfile(f1)
        warning('File not found: %s (skip run %d)', f1, i);
        continue;
    end

    try
        T = readtable(f1);
        x = T.x(:);
        y = T.y(:);

        [mse, nlpd, coverage, width, tsec] = IP(x, y, xt, y_true, 1);%IP, CGP_monotone, CGP_convex
        %[mse, nlpd, coverage, width, tsec] = CGP_monotone(x, y, xt, y_true);
        %[mse, nlpd, coverage, width, tsec] = CGP_convex(x, y, xt, y_true);
        results(i,:) = [mse, nlpd, coverage, width, tsec];
    catch ME
        warning('Run %d failed: %s', i, ME.message);
    end
end

run_id = (1:n_trials).';
Tout = table(run_id, results(:,1), results(:,2), results(:,3), results(:,4), results(:,5), ...
    'VariableNames', {'run','MSE','NLPD','Coverage','Width','Time'});

out_csv = fullfile(data_root, 'results_monotone\ip_runs_stepwise_monotone.csv');
writetable(Tout, out_csv);

fprintf('Saved results to: %s\n', out_csv);

mu  = mean(results, 1, 'omitnan');
sig = std(results, 0, 1, 'omitnan');
fprintf('\nAverage over %d runs :\n', n_trials);
fprintf('MSE: %.4f ± %.4f\n',   mu(1), sig(1));
fprintf('NLPD: %.4f ± %.4f\n',  mu(2), sig(2));
fprintf('Coverage: %.4f ± %.4f\n', mu(3), sig(3));
fprintf('Width: %.4f ± %.4f\n', mu(4), sig(4));
fprintf('Time (sec): %.4f ± %.4f\n', mu(5), sig(5));
