%% LIS first implementation approach - context: Black Scholes model
% small example: y=G*x + noise 
% n=parameter space dimension
% m=observation space dimension

clear all; clc;
addpath('src/')
 
%% parameters
%two market parameters that are fixed and known
S0 = 100;  %all strikes and vol grids points are defined relative to this
r  = 0.05; %annual risk free rate 
 
%% 1. true vol surface
% establishes the ground truth/reference, i.e. the vol surface you are
% trying to infer
[~, theta_true, S_vol_grid, t_vol_grid] = build_sigma(S0); %define the parameter space, grids on which theta lives. Everything is built on this grids
n = length(theta_true); %synthetic true vol surface I am trying to recover 
 
%% 2. build G (or load if already saved)
%computes the linearized Forward operator G
theta_star = mean(theta_true)*ones(n,1); %linearization point at flat surface (0.28), much closer to theta_true on avg, linearization error shrinks
N_S = 200; N_t = 200; %grid resolution
 
if exist('G_matrix.mat', 'file')
    fprintf('Loading saved G...\n');
    load('G_matrix.mat');
else
    fprintf('Building G -- this takes ~8 minutes...\n');
    tic
    [G_bs, F0, contracts] = build_G(S0, r, theta_star, S_vol_grid, t_vol_grid, N_S, N_t);
    fprintf('G built in %.1f seconds.\n', toc);
    save('G_matrix.mat', 'G_bs', 'F0', 'contracts','theta_star', 'S_vol_grid', 't_vol_grid');
end
 
%% 3. inputs for LIS
% constructing the three inputs for bip, 
m = size(G_bs, 1);
G = G_bs; %Jacobian as the forward operator
 
% Prior
Gamma_pr = build_prior(S_vol_grid, t_vol_grid);
 
% Observation noise
noise_std  = 0.7; 
Gamma_obs  = noise_std^2 * eye(m); %assuming all contracts have the same noise level
 
% Synthetic observations from linearized model
rng(42);
y = F0 + G_bs*(theta_true - theta_star) + noise_std*randn(m,1);
 
%% 4. exact posterior covariance + mean
Gamma_obs_inv = (1/noise_std^2) * eye(m);
H             = G' * Gamma_obs_inv * G;    %hessian of (-)log-likelihood
Gamma_pr_inv  = inv(Gamma_pr);
Gamma_pos     = inv(H + Gamma_pr_inv);

mu_pos = theta_star + Gamma_pos * G' * Gamma_obs_inv * (y - F0);
%% 5. generalized EV problem (SVD -- Remark 4)
%cholesky of Gamma_pr - assuming square root factorization
S_pr      = chol(Gamma_pr, 'lower');       %st. Gamma_pr = S_pr * S_pr'
S_obs_inv = (1/noise_std) * eye(m);
 
%remark 4
A_lis         = S_obs_inv * G * S_pr;      %taking SVD of A_lis solves the EVproblem of Htilde
[~, Delta, Z] = svd(A_lis, 'econ');        %gives the zi (lives in transformed space)
delta         = diag(Delta);               %containing singular values descending
 
%eigenvectors in parameter space
W_hat = S_pr * Z;                          %to undo the change of variables, now in initial parameter space
 
%% 6. build approximate posterior covariance (Theorem 2.3)
% here its showed the "loss function" minimization made concrete
r_max = length(delta);
for r = 1:r_max
    Gamma_pos_approx = Gamma_pr;
    for i = 1:r
        w = W_hat(:,i);                    %output of
        d = delta(i);                      %solving the minimization problem of the Loss functions
        Gamma_pos_approx = Gamma_pos_approx - (d^2/(1+d^2)) * (w*w');
    end
    Gamma_pos_approx_all{r} = Gamma_pos_approx;
end
 
%% 7. approximation errors (Frobenius and Forstner)
frob_errors     = zeros(r_max,1);          %Pre Allocation
forstner_errors = zeros(r_max,1);          %each will store the error for one rank r
 
for r = 1:r_max
    Ga                 = Gamma_pos_approx_all{r};
    frob_errors(r)     = norm(Gamma_pos - Ga, 'fro');
    %Forstner as in paper sqrt(sum(log(lambda_i)^2)
    lambda             = eig(Ga, Gamma_pos); %generalized ev of the pencil
    forstner_errors(r) = sqrt(sum(log(lambda).^2));
end
 
%% 8. optimal projector (Corollary 3.2)
r_plot = sum(delta.^2 >1);                               %choose rank based on eigenvalue spectrum
W_tilde = Gamma_pr_inv * W_hat;
 
P_r = zeros(n, n);
for i = 1:r_plot
    P_r = P_r + W_hat(:,i) * W_tilde(:,i)';
end
oblique_error = norm(P_r^2 - P_r, 'fro');
fprintf('Oblique projector error ||Pr^2 - Pr|| = %.2e\n', oblique_error);
 
%reduced forward operator Gr
G_r = G * P_r;
 
%posterior with Gr
H_r                 = G_r' * Gamma_obs_inv * G_r;
Gamma_pos_projected = inv(H_r + Gamma_pr_inv);
 
%Aproximation2.3 vs. Projected
frob_error_projected = norm(Gamma_pos_projected - Gamma_pos_approx_all{r_plot}, 'fro');
fprintf('Frobenius error approx vs projected = %.2e\n', frob_error_projected);
%the optimal rank r posterior approximation from theorem 2.3 is identical
%to what you get it you simply project the forward operator onto the r most
%likelihood informed directions, two completely different paths, to the
%same result

%% 9. rank r approximate posterior mean + convergence
mu_pos_approx = zeros(n, r_max);   % store all rank-r means

for r = 1:r_max
    % rank-r projector (already built incrementally, rebuild cleanly here)
    P_r_loop = zeros(n, n);
    for i = 1:r
        P_r_loop = P_r_loop + W_hat(:,i) * W_tilde(:,i)';
    end

    % projected forward operator
    G_r_loop = G * P_r_loop;

    % rank-r posterior mean
    H_r_loop  = G_r_loop' * Gamma_obs_inv * G_r_loop;
    Gpos_r    = Gamma_pos_approx_all{r};
    mu_pos_approx(:,r) = theta_star + Gpos_r * G_r_loop' * Gamma_obs_inv * (y - F0); %actual comp
end

% convergence error
signal_norm = norm(mu_pos - theta_star);   % normalization
mean_errors = zeros(r_max, 1);
for r = 1:r_max
    mean_errors(r) = norm(mu_pos_approx(:,r) - mu_pos) / signal_norm;
end
 
%% -----------------------------------------------------------------------
%% PLOTS
%% -----------------------------------------------------------------------
 
% Color palette -- Jakob way
blue_c  = [0.4660 0.6740 0.1880]*0.6 + [1 1 1]*0.4;   %LIS green-ish
red_c   = [0.8500 0.3250 0.0980]*0.7 + [1 1 1]*0.3;   %Forstner orange
c_max   = max(Gamma_pr(:));
 
%% plot 1: eigenvalue spectrum (semilogy -- shows jumps clearly)
% remark: delta_i^2 are the ev of (H, Gamma_pr_inv)
figure;
semilogy(1:r_max, delta.^2, 'o-', ...
         'LineWidth', 2, 'Color', [0.2 0.2 0.6], ...
         'MarkerFaceColor', [0.2 0.2 0.6], 'MarkerSize', 6);
hold on;
yline(1, '--', 'Color', [0.7 0.1 0.1], 'LineWidth', 1.5);
box off;
set(gca, 'FontSize', 13);
xlabel('Direction $i$',           'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\delta_i^2$',            'Interpreter', 'latex', 'FontSize', 14);
title('Eigenvalue spectrum -- LIS pencil $(H,\,\Gamma_{pr}^{-1})$', ...
      'Interpreter', 'latex', 'FontSize', 15);
text(r_max*0.6, 1.4, '$\delta^2 = 1$ threshold', ...
     'Interpreter', 'latex', 'FontSize', 11, 'Color', [0.7 0.1 0.1]);
 
%% plot 2: approximation error vs rank (Frobenius + Forstner)
figure;
t2 = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
 
nexttile;
semilogy(1:r_max, frob_errors, 'o-', 'LineWidth', 2, 'Color', blue_c);
box off;
set(gca, 'FontSize', 13);
xlabel('Rank $r$',       'Interpreter', 'latex', 'FontSize', 14);
ylabel('Error',          'Interpreter', 'latex', 'FontSize', 14);
title('Frobenius error', 'Interpreter', 'latex', 'FontSize', 15);
 
nexttile;
semilogy(1:r_max, forstner_errors, 'o-', 'LineWidth', 2, 'Color', red_c);
box off;
set(gca, 'FontSize', 13);
xlabel('Rank $r$',           'Interpreter', 'latex', 'FontSize', 14);
title('F\"{o}rstner error', 'Interpreter', 'latex', 'FontSize', 15);
 
%% plot 3: Prior vs Posterior covariance (2 panel)
figure;
t3 = tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
 
nexttile;
imagesc(Gamma_pr);
axis equal tight;
clim([0 c_max]);
set(gca, 'FontSize', 12);
xlabel('$j$', 'Interpreter', 'latex');
ylabel('$k$', 'Interpreter', 'latex');
title('Prior $\Gamma_{pr}$', 'Interpreter', 'latex', 'FontSize', 14);
 
nexttile;
imagesc(Gamma_pos);
axis equal tight;
clim([0 c_max]);
colorbar;
set(gca, 'FontSize', 12);
xlabel('$j$', 'Interpreter', 'latex');
title('Posterior $\Gamma_{pos}$', 'Interpreter', 'latex', 'FontSize', 14);
 
%% plot 4: Prior vs Posterior vs LIS approximation (3 panel)
figure;
t4 = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
 
nexttile;
imagesc(Gamma_pr);
axis equal tight;
clim([0 c_max]);
set(gca, 'FontSize', 11);
xlabel('$j$', 'Interpreter', 'latex');
ylabel('$k$', 'Interpreter', 'latex');
title('Prior $\Gamma_{pr}$', 'Interpreter', 'latex', 'FontSize', 13);
 
nexttile;
imagesc(Gamma_pos);
axis equal tight;
clim([0 c_max]);
set(gca, 'FontSize', 11);
xlabel('$j$', 'Interpreter', 'latex');
title('Posterior $\Gamma_{pos}$', 'Interpreter', 'latex', 'FontSize', 13);
 
nexttile;
imagesc(Gamma_pos_approx_all{r_plot});
axis equal tight;
clim([0 c_max]);
colorbar;
set(gca, 'FontSize', 11);
xlabel('$j$', 'Interpreter', 'latex');
title(['LIS approx $\hat{\Gamma}_{pos},\ r=' num2str(r_plot) '$'], ...
      'Interpreter', 'latex', 'FontSize', 13);

%% Plot 5: Diagonal variance comparison
figure;
plot(diag(Gamma_pr),  'b-o', 'LineWidth', 1.5, ...
    'MarkerFaceColor', [0.2 0.2 0.8]); 
hold on;
plot(diag(Gamma_pos), 'r-o', 'LineWidth', 1.5, ...
    'MarkerFaceColor', [0.8 0.2 0.2]);
xline(5.5,  '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xline(10.5, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xline(15.5, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xline(20.5, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
text(3,  max(diag(Gamma_pr))*1.02, '$t=0.25$', 'Interpreter','latex','FontSize',10);
text(8,  max(diag(Gamma_pr))*1.02, '$t=0.50$', 'Interpreter','latex','FontSize',10);
text(13, max(diag(Gamma_pr))*1.02, '$t=1.00$', 'Interpreter','latex','FontSize',10);
text(18, max(diag(Gamma_pr))*1.02, '$t=1.50$', 'Interpreter','latex','FontSize',10);
text(23, max(diag(Gamma_pr))*1.02, '$t=2.00$', 'Interpreter','latex','FontSize',10);
legend('Prior variance', 'Posterior variance', 'Interpreter', 'latex', 'FontSize', 12);
xlabel('Parameter index $j$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Variance $\sigma^2_j$', 'Interpreter', 'latex', 'FontSize', 13);
title('Uncertainty per vol surface parameter', ...
    'Interpreter', 'latex', 'FontSize', 14);
box off; set(gca, 'FontSize', 12);

%% Plot 6: Variance reduction bar chart
var_reduction = 1 - diag(Gamma_pos)./diag(Gamma_pr);

figure;
bar(var_reduction, 'FaceColor', [0.2 0.2 0.6], 'EdgeColor', 'none');
hold on;
xline(5.5,  '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xline(10.5, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xline(15.5, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
xline(20.5, '--', 'Color', [0.5 0.5 0.5], 'LineWidth', 1);
text(3,  1.05, '$t=0.25$', 'Interpreter','latex','FontSize',10);
text(8,  1.05, '$t=0.50$', 'Interpreter','latex','FontSize',10);
text(13, 1.05, '$t=1.00$', 'Interpreter','latex','FontSize',10);
text(18, 1.05, '$t=1.50$', 'Interpreter','latex','FontSize',10);
text(23, 1.05, '$t=2.00$', 'Interpreter','latex','FontSize',10);
xlabel('Parameter index $j$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Variance reduction', 'Interpreter', 'latex', 'FontSize', 13);
title('Relative uncertainty reduction per vol surface parameter', ...
    'Interpreter', 'latex', 'FontSize', 14);
ylim([0 1.1]); box off; set(gca, 'FontSize', 12);


%% mean what did i recover

figure;
t5 = tiledlayout(1,3,'Padding','compact','TileSpacing','compact');

nexttile;
surf(S_vol_grid, t_vol_grid, reshape(theta_true, length(t_vol_grid), length(S_vol_grid)));
title('True $\theta_{\rm true}$','Interpreter','latex','FontSize',13);
xlabel('$S$','Interpreter','latex'); ylabel('$t$','Interpreter','latex');
zlabel('$\sigma(S,t)$','Interpreter','latex'); zlim([0.18 0.42]);

nexttile;
surf(S_vol_grid, t_vol_grid, reshape(mu_pos, length(t_vol_grid), length(S_vol_grid)));
title('Recovered $\mu_{pos}$','Interpreter','latex','FontSize',13);
xlabel('$S$','Interpreter','latex'); ylabel('$t$','Interpreter','latex');
zlabel('$\sigma(S,t)$','Interpreter','latex'); zlim([0.18 0.42]);

nexttile;
surf(S_vol_grid, t_vol_grid, reshape(mu_pos - theta_true, length(t_vol_grid), length(S_vol_grid)));
title('Error $\mu_{pos} - \theta_{\rm true}$','Interpreter','latex','FontSize',13);
xlabel('$S$','Interpreter','latex'); ylabel('$t$','Interpreter','latex');
zlabel('Error','Interpreter','latex');
colorbar;

%% w hat LIs directions
figure;
t_lis = tiledlayout(1, r_plot, 'Padding','compact','TileSpacing','compact');
for i = 1:r_plot
    nexttile;
    w_surface = reshape(W_hat(:,i), length(t_vol_grid), length(S_vol_grid));
    surf(S_vol_grid, t_vol_grid, w_surface);
    title(['$\hat{w}_{' num2str(i) '}$'], 'Interpreter','latex','FontSize',13);
    xlabel('$S$','Interpreter','latex','FontSize',11);
    ylabel('$t$','Interpreter','latex','FontSize',11);
    zlabel('weight','FontSize',10);
    colorbar;
    set(gca,'FontSize',10);
end
sgtitle('LIS directions in vol surface space', 'Interpreter','latex','FontSize',14);


%% rank r mean convergence
figure;
semilogy(1:r_max, mean_errors, 'o-', ...
    'LineWidth', 2, 'Color', [0.2 0.2 0.6], ...
    'MarkerFaceColor', [0.2 0.2 0.6], 'MarkerSize', 6);
hold on;
xline(r_plot, '--', 'Color', [0.7 0.1 0.1], 'LineWidth', 1.5);
text(r_plot + 0.2, max(mean_errors)*0.5, ...
    ['$r^* = ' num2str(r_plot) '$'], ...
    'Interpreter','latex','FontSize',11,'Color',[0.7 0.1 0.1]);
box off;
set(gca,'FontSize',13);
xlabel('Rank $r$', 'Interpreter','latex','FontSize',14);
ylabel('Relative error', 'Interpreter','latex','FontSize',14);
title('Rank-$r$ posterior mean convergence $\|\mu_{pos}^{(r)} - \mu_{pos}\| / \|\mu_{pos} - \theta_\star\|$', ...
    'Interpreter','latex','FontSize',14);
