addpath('src/')
%contract 1
S0 = 100;
K =100;
r = 0.05;
sigma = 0.20;
T = 1;

C = bs_pricer(S0, K, r, sigma, T) %with given values (ATM option) we get that C=10.5 (around 10% of asset
%price)which is expectable for a 1 year ATM option with volatility 20%
%(normal)

v = vega(S0, K, r,sigma, T) %interpreting the value v = 37.52, means that we have f.e. a 1% increase in
%volatility (sigma+0.01) the call price changes by 0,375, ending up to
%10.83, actually sensitive by changes in sigma

%% vega across strikes
K_grid = linspace(60, 140, 100);
v_grid = zeros(1, 100);

for i = 1:100
    v_grid(i) = vega(S0, K_grid(i), r, sigma, T);
end

figure;
plot(K_grid, v_grid, 'LineWidth', 1.5, 'Color', [0.2 0.2 0.6]);
hold on;
xline(S0, '--', 'Color', [0.6 0.1 0.1], 'LineWidth', 1.2);
xlabel('Strike $K$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Vega $\partial C / \partial \sigma$', 'Interpreter', 'latex', 'FontSize', 13);
title('Vega across strikes', 'Interpreter', 'latex', 'FontSize', 14);
text(S0+1, max(v_grid)*0.95, 'ATM', 'FontSize', 11, 'Color', [0.6 0.1 0.1]);
set(gca, 'FontSize', 11, 'Box', 'off');
grid off;

%looking forward to our implementation idea, that v would be one entry of
%our G matrix, when we build G across many contracts and many vol surface
%points, the entries will vary a lot, large near ATM and tiny for deep OTM
%The curve here will be like a row in our G matrix, at K=100 ATM, large
%vega -> observation strongly constrains sigma -> relevant for LIS

%% test fd_solver against bs_pricer
sigma_flat = @(S,t) 0.20; %constant sigma assumption of closed form
%later instead of this "flat constant" value we will have sigma_func as  a
%2D matrix of vol values at every (S,t) grid point


N_S = 200;
N_t = 200;

price_fd = fd_solver(S0, K, r, T, sigma_flat, N_S, N_t)
price_bs = bs_pricer(S0, K, r, 0.20, T)
%correct fd_solver

%% True volatility surface 
[sigma_func_true, theta_true, S_vol_grid, t_vol_grid] = build_sigma(S0);

[S_mat, t_mat] = meshgrid(S_vol_grid, t_vol_grid);
Sigma_mat = reshape(theta_true, length(t_vol_grid), length(S_vol_grid));

figure;
surf(S_vol_grid, t_vol_grid, Sigma_mat);
xlabel('Stock price $S$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Time $t$', 'Interpreter', 'latex', 'FontSize', 13);
zlabel('$\sigma(S,t)$', 'Interpreter', 'latex', 'FontSize', 13);
title('True local volatility surface $\theta_{\rm true}$','Interpreter', 'latex', 'FontSize', 14);
colorbar;
set(gca, 'FontSize', 11, 'Box', 'off');

%% Build G
clear K T sigma
n = length(theta_true);
r=0.05;
theta_star = 0.20 * ones(n, 1);
N_S = 200;
N_t = 200;

fprintf('Starting G computation...\n');
tic
[G_bs, F0, contracts] = build_G(S0, r, theta_star,S_vol_grid, t_vol_grid, N_S, N_t);
t_elapsed = toc;
fprintf('G built in %.1f seconds.\n', t_elapsed);

figure;
imagesc(G_bs);
colorbar;
xlabel('Parameter index $j$ (vol surface point)','Interpreter', 'latex', 'FontSize', 13);
ylabel('Contract index $i$', 'Interpreter', 'latex', 'FontSize', 13);
title('Jacobian $G_{ij} = \partial C_i / \partial \sigma_j$','Interpreter', 'latex', 'FontSize', 14);
set(gca, 'FontSize', 11);

save('G_matrix.mat', 'G_bs', 'F0', 'contracts','theta_star', 'S_vol_grid', 't_vol_grid');
fprintf('G saved.\n');

%% Build prior
Gamma_pr = build_prior(S_vol_grid, t_vol_grid);

figure;
imagesc(Gamma_pr);
colorbar;
xlabel('Parameter index $j$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Parameter index $k$', 'Interpreter', 'latex', 'FontSize', 13);
title('Prior covariance $\Gamma_{pr}$ (squared exponential kernel)', ...
    'Interpreter', 'latex', 'FontSize', 14);
set(gca, 'FontSize', 11);