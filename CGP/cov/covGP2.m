function [K_1, K_2, K_3, KK_2] = covGP2(loghyper, x, z, t)
% covGP  — Squared Exponential (SE) covariance blocks with higher-order derivatives
%
% Parameterization:
%   k(x^p, x^q) = sf2 * exp( - (x^p - x^q)' * inv(P) * (x^p - x^q) / 2 )
%   where P = ell^2 * I, and sf2 is the signal variance.
%
% Hyperparameters:
%   loghyper = [ log(ell)
%                log(sqrt(sf2)) ]
%   (when using the 4-argument call, also requires log(sigma_n))
%
% Call formats:
%   Three arguments:
%       [K_1, K_2, K_3, KK_2] = covGP([log(ell); log(sf)], x, z)
%       K_1  = cov(f(x),   f(x))
%       K_2  = cov(f(x),   f''(z))
%       K_3  = cov(f''(z), f''(z))
%       KK_2 = K_2'
%
%   Four arguments:
%       [K_1, K_2, K_3, KK_2] = covGP([log(ell); log(sf); log(sigma_n)], x, z, t)
%       This version returns larger block matrices including noise and mixed
%       derivatives (kept consistent with your original implementation).
%
% Dependency: gp_dist (pairwise scaled squared distance)

    % Unpack hyperparameters
    ell = exp(loghyper(1));       % characteristic length scale
    sf2 = exp(2 * loghyper(2));   % signal variance

    if nargin == 3
        % ===================== Three-argument version =====================
        % K_1 = cov(f(x), f(x))
        K_1 = sf2 * exp( - (gp_dist(x'/ell)).^2 / 2 );

        % K_2 = cov(f(x), f''(z))
        tempK2 = gp_dist(x'/ell, z'/ell);
        K_2 = sf2 * exp(-tempK2.^2 / 2) .* ( (tempK2.^2) / ell^2 - 1 / ell^2 );

        % KK_2 = transpose of K_2
        KK_2 = K_2';

        % K_3 = cov(f''(z), f''(z))
        tempK3 = gp_dist(z'/ell).^2;
        K_3 = sf2 * exp(-tempK3 / 2) .* (1 / ell^4) .* ( tempK3.^2 - 6 * tempK3 + 3 );

    else
        % ===================== Four-argument version (with noise) =====================
        sigma2_n = exp(2 * loghyper(3));   % noise variance

        % K_1 = cov(f(t), f(t))
        K_1 = sf2 * exp( - (gp_dist(t'/ell)).^2 / 2 );

        % K_2 = [ cov(f(t), f'(z)),  cov(f(t), f(x)) ]
        tempK21 = gp_dist(t'/ell, z'/ell);
        tempK22 = gp_dist(t'/ell, x'/ell);
        K_2 = [ sf2 * exp(-tempK21.^2 / 2) .* ( -1/ell .* tempK21 ), ...
                sf2 * exp(-tempK22.^2 / 2) ];

        % KK_2 = transpose of K_2
        KK_2 = K_2';

        % Blocks of K_3
        % Top-left: cov(f'(z), f'(z))
        tempK31 = gp_dist(z'/ell).^2;

        % Top-right: cov(f'(z), f(x))
        tempK32 = gp_dist(z'/ell, x'/ell);
        tempK33 = sf2 * exp(-tempK32.^2 / 2) .* (1/ell .* tempK32);

        % Bottom-right: sigma^2 I + cov(f(x), f(x))
        K_xx = sigma2_n * diag(ones(size(x,1),1)) + sf2 * exp( - (gp_dist(x'/ell)).^2 / 2 );

        % Assemble K_3
        K_3 = [ sf2 * exp(-tempK31 / 2) .* (1/ell^2) .* (1 - tempK31),   tempK33; ...
                tempK33',                                               K_xx ];
    end
end
