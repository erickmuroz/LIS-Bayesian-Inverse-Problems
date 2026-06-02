%% Assembly of the forward operator G = Jacobian matrix directly related to the Vega
% what we have done so fat is F(theta) = fd_solver(S0, K, r, T, sigma_func,
% N_S, Nt) where sigma_func comes from build_sigma, a full 2d local vol
% surface = THIS is the expensive nonlinear forward model

%Now this build_G..m will linearize this expensive F around theta_star
%(flat surface at 0.20)

function [G, F0, contracts] = build_G(S0, r, theta_star, S_vol_grid, t_vol_grid, N_S, N_t)
%define m contracts
K_contracts = [70, 85, 100, 115, 130];
T_contracts = [0.25, 0.5, 1.0];
[K_mat, T_mat] = meshgrid(K_contracts, T_contracts);
contracts.K = K_mat(:);
contracts.T = T_mat(:);
m = length(contracts.K);
n = length(theta_star);

%perturbation for linearization
eps =1e-3;

%BASELINE sigma_func from theta_star
Sigma_star = reshape(theta_star, length(t_vol_grid), length(S_vol_grid));
sigma_star_func = @(S,t) interp2(S_vol_grid, t_vol_grid, Sigma_star, S, t, 'linear', 0.20);

%BASELINE prices 
fprintf('Computing baseline prices...\n');
F0 = zeros(m, 1);
for i = 1:m
    F0(i) = fd_solver(S0, contracts.K(i), r, contracts.T(i), sigma_star_func, N_S, N_t);
end
fprintf('Baseline done.\n');

% G column by column
fprintf('Building G (%d x %d) — %d PDE solves...\n', m, n, m*n);
G = zeros(m, n);

for j = 1:n
    if mod(j,5) == 0
        fprintf('  column %d / %d\n', j, n);
    end

    % Perturb theta_star in direction j
    theta_pert = theta_star;
    theta_pert(j) = theta_pert(j) + eps;

    % Perturbed sigma_func
    Sigma_pert = reshape(theta_pert, length(t_vol_grid), length(S_vol_grid));
    sigma_pert_func = @(S,t) interp2(S_vol_grid, t_vol_grid, Sigma_pert,S, t, 'linear', 0.20);

    % Perturbed prices
    F_pert = zeros(m, 1);
    for i = 1:m
        F_pert(i) = fd_solver(S0, contracts.K(i), r, contracts.T(i),sigma_pert_func, N_S, N_t);
    end

    % fd column
    G(:, j) = (F_pert - F0) / eps;
end
fprintf('G matrix complete.\n');

end