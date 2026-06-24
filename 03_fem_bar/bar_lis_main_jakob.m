%% LIS implementation - context: static structural load on a 1D bar
%
%  small example: y = G*x + noise
%
%  Difference from bs_lis_main.m:
%   - prior built Jakob's way: exponential kernel on the distributed load,
%     pushed through L_mat -> SINGULAR nodal-force prior Gamma_pr.
%   - the optimal oblique projector P_r = sum v_i w_i' is the main object
%     (Corollary 3.2 / OLR), the central method in this implementation


%chosen combination
% m=30, gamma_obs_var = 1e-8, ell=0.3

clear all; clc;
addpath('src/')

%% parameters
% bar geometry and material that are fixed and known
L    = 2;       % length of the bar
nele = 100;     % number of finite elements
EA   = 4e8;     % axial rigidity of the bar
BC_dof = 1;     % clamped node (Dirichlet)
l    = L / nele;

%% 1. true distributed load
% establishes the ground truth, i.e. the distributed load you are trying to infer
[~, theta_true, z_grid, x_nodes] = assembly_load(L, nele);
n = length(theta_true);

%% 2. assemble G (forward operator) and the operators Jakob's prior 
m = 30; %number of displacement sensors
sensor_seed = 41; %Jakob's bar seed for sensor placement

if exist('G_matrix.mat', 'file')
    fprintf('Loading saved G...\n');
    load('G_matrix.mat');
else
    fprintf('Building G...\n');
    tic
    [G_bar, contracts] = assembly_G(L, nele, EA, m, BC_dof, sensor_seed);
    fprintf('G built in %.3f seconds.\n', toc);
    save('G_matrix.mat', 'G_bar', 'contracts', 'z_grid', 'x_nodes');
end

L_mat = contracts.L_mat; 

%% 3. inputs for LIS
% constructing the three inputs for bip
m = size(G_bar, 1); %should be equal to m=10, size of Gbar 
G = G_bar; %structural operator as the forward operator

%assemnbly prior
[Gamma_pr, S_pr, mu_f] = assembly_prior(nele, L_mat, l);

% Observation noise 
gamma_obs_var = 1e-6;
noise_std     = sqrt(gamma_obs_var);
Gamma_obs     = gamma_obs_var * eye(m); %all sensors same noise level

% measurement model with G_bar = CK^-1
rng(41);
y =G_bar * theta_true + noise_std*randn(m,1);

%% 4. exact posterior covariance + mean (WOODBURY form)
% Singular prior, inv(Gamma_pr) no longer exists => use the Kalman/Woodbury 
tmp       = G*Gamma_pr*G' + Gamma_obs;
Gamma_pos = Gamma_pr - Gamma_pr*G'*(tmp\G)*Gamma_pr;

mu_pos = mu_f + Gamma_pr*G'*(tmp\(y - G*mu_f));

%% 5. generalized EV problem (SVD -- Remark 4, square-root route)
% No chol(Gamma_pr): the singular prior already comes with its rectangular
% square-root S_pr (S_pr*S_pr' = Gamma_pr). Whiten the likelihood and take the
% SVD of R'*S_pr, exactly like calculateLISBasis.
R = G' .* (1 ./ sqrt(diag(Gamma_obs)))';%divides each sensor contribution by its noise standard deviation

[U, Delta, Z] = svd(R' * S_pr);            %solves the EV problem of Htilde
delta = diag(Delta);                       %singular values descending

% numerical rank: discard singular values at machine-noise level
tol = max(size(R' * S_pr)) * eps(max(delta));
r_eff = sum(delta > tol);
U     = U(:, 1:r_eff);
delta = delta(1:r_eff);
Z     = Z(:, 1:r_eff);

% reconstruction basis V and projection basis W 
V = S_pr * Z;
W = R * U .* (1 ./ delta)';

%% 6. optimal oblique projector (Corollary 3.2 / OLR) now as MAIN METHOD
% P_r = sum_{i=1}^r v_i w_i'
% the optimal low-rank projector. The reduced forward operator is G_app = G*P_r,
% and its posterior is the optimal rank-r posterior approximation.
r_plot = sum(delta.^2 > 1);                
r_plot = max(r_plot, 1); 

P_r = zeros(n, n);
for i = 1:r_plot
    P_r = P_r + V(:,i) * W(:,i)';
end
oblique_error = norm(P_r^2 - P_r, 'fro');
fprintf('Oblique projector error ||Pr^2 - Pr|| = %.2e\n', oblique_error);

% reduced forward operator and its posterior (Woodbury, singular prior)
G_r       = G * P_r;
tmp_r     = G_r*Gamma_pr*G_r' + Gamma_obs;
Gamma_pos_proj = Gamma_pr - Gamma_pr*G_r'*(tmp_r\G_r)*Gamma_pr;

%% 7. approximation errors over rank r (Frobenius and Forstner)
r_max = r_eff;
frob_errors     = zeros(r_max,1);
forstner_errors = zeros(r_max,1);
Gamma_pos_approx_all = cell(r_max,1);

for r = 1:r_max
    Pr_r = zeros(n, n);
    for i = 1:r
        Pr_r = Pr_r + V(:,i) * W(:,i)';
    end
    Ga      = G * Pr_r;
    tmp_a   = Ga*Gamma_pr*Ga' + Gamma_obs;
    pos_a   = Gamma_pr - Gamma_pr*Ga'*(tmp_a\Ga)*Gamma_pr;
    Gamma_pos_approx_all{r} = pos_a;

    frob_errors(r)     = norm(Gamma_pos - pos_a, 'fro');
    forstner_errors(r) = foerstner_distance(pos_a, Gamma_pos, W);
end

%% 8. rank r approximate posterior mean + convergence (single theta_true)
mu_pos_approx = zeros(n, r_max);
for r = 1:r_max
    Pr_r = zeros(n, n);
    for i = 1:r
        Pr_r = Pr_r + V(:,i) * W(:,i)';
    end
    Ga    = G * Pr_r;
    tmp_a = Ga*Gamma_pr*Ga' + Gamma_obs;
    mu_pos_approx(:,r) = mu_f + Gamma_pr*Ga'*(tmp_a\(y - Ga*mu_f));
end

signal_norm = norm(mu_pos - mu_f);
mean_errors = zeros(r_max,1);
for r = 1:r_max
    mean_errors(r) = norm(mu_pos_approx(:,r) - mu_pos) / signal_norm;
end

%% 9. STATISTICAL posterior mean error
%on average, over many possible true loads, how well does rank r work?
%(samples from gensamples) 

%so, running the same experiment as s8 but 100 times, each time with a
%different true load drawn from the prior 
N_rep   = 100;
mu_q    = 4e6;
sigma_q = 0.3 * 4e6;
ell_q   = 0.3;

[~, S_u] = gen_samples(N_rep, L, nele, EA, BC_dof, mu_q, sigma_q, ell_q); %discarting the load samples, just keeping the bar response 
C = contracts.C;%for converting in the noisy measurement model 

% pre-build the rank-r OLR operators once to apply the oblique projector 
G_r_all = cell(r_max,1); 
for r = 1:r_max
    Pr_r = zeros(n, n);
    for i = 1:r
        Pr_r = Pr_r + V(:,i) * W(:,i)'; %for each rank r, build the rank r oblique projector, then apply it to G
    end
    G_r_all{r} = G * Pr_r;
end

% inline posterior mean (Woodbury): mu = mu_f + Gamma_pr G'(G Gamma_pr G'+Gobs)^{-1}(y - G mu_f)
meanCalc = @(Gop, yv) mu_f + Gamma_pr * Gop' * ((Gop * Gamma_pr * Gop' + Gamma_obs) \ (yv - Gop * mu_f));

error_stat = zeros(N_rep, r_max);
rng(7);
for j = 1:N_rep
    y_j     = C * S_u(:, j) + noise_std * randn(m, 1); %for each scenario j, C is applied to the displacement field, to pick out the readings at the m sensores. 
    mu_full = meanCalc(G, y_j); %given the y_j, one can compute now the best possible posterior means 
    nrm     = norm(mu_full);
    for r = 1:r_max
        mu_r             = meanCalc(G_r_all{r}, y_j); %for each rank r, cheap rank r posterior mean is calculated 
        error_stat(j, r) = norm(mu_full - mu_r) / nrm;%to evaluate the error, to cpmpare it with the full answer of jth scnenario
    end
end
mean_errors_stat = mean(error_stat, 1);

%% -----------------------------------------------------------------------
%% PLOTS  (same set and style as bs_lis_main)
%% -----------------------------------------------------------------------

blue_c  = [0.4660 0.6740 0.1880]*0.6 + [1 1 1]*0.4;   %LIS green-ish
red_c   = [0.8500 0.3250 0.0980]*0.7 + [1 1 1]*0.3;   %Forstner orange
c_max   = max(Gamma_pr(:));

%% plot 1: eigenvalue spectrum
figure;
semilogy(1:r_max, delta.^2, 'o-', ...
         'LineWidth', 2, 'Color', [0.2 0.2 0.6], ...
         'MarkerFaceColor', [0.2 0.2 0.6], 'MarkerSize', 6);
hold on;
yline(1, '--', 'Color', [0.7 0.1 0.1], 'LineWidth', 1.5);
box off; set(gca, 'FontSize', 13);
xlabel('Direction $i$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('$\delta_i^2$',  'Interpreter', 'latex', 'FontSize', 14);
title('Eigenvalue spectrum -- LIS pencil $(H,\,\Gamma_{pr}^{-1})$', ...
      'Interpreter', 'latex', 'FontSize', 15);
text(r_max*0.6, 1.4, '$\delta^2 = 1$ threshold', ...
     'Interpreter', 'latex', 'FontSize', 11, 'Color', [0.7 0.1 0.1]);

%% plot 2: approximation error vs rank (Frobenius + Forstner)
figure;
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile;
semilogy(1:r_max, frob_errors, 'o-', 'LineWidth', 2, 'Color', blue_c);
box off; set(gca, 'FontSize', 13);
xlabel('Rank $r$', 'Interpreter', 'latex', 'FontSize', 14);
ylabel('Error',    'Interpreter', 'latex', 'FontSize', 14);
title('Frobenius error', 'Interpreter', 'latex', 'FontSize', 15);
nexttile;
semilogy(1:r_max, forstner_errors, 'o-', 'LineWidth', 2, 'Color', red_c);
box off; set(gca, 'FontSize', 13);
xlabel('Rank $r$', 'Interpreter', 'latex', 'FontSize', 14);
title('F\"{o}rstner error', 'Interpreter', 'latex', 'FontSize', 15);

%% plot 3: Prior vs Posterior covariance (2 panel)
figure;
tiledlayout(1,2,'Padding','compact','TileSpacing','compact');
nexttile;
imagesc(Gamma_pr); axis equal tight; clim([0 c_max]);
set(gca, 'FontSize', 12);
xlabel('$j$', 'Interpreter', 'latex'); ylabel('$k$', 'Interpreter', 'latex');
title('Prior $\Gamma_{pr}$', 'Interpreter', 'latex', 'FontSize', 14);
nexttile;
imagesc(Gamma_pos); axis equal tight; clim([0 c_max]); colorbar;
set(gca, 'FontSize', 12);
xlabel('$j$', 'Interpreter', 'latex');
title('Posterior $\Gamma_{pos}$', 'Interpreter', 'latex', 'FontSize', 14);

%% plot 4: Prior vs Posterior vs LIS approximation (3 panel)
figure;
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
nexttile;
imagesc(Gamma_pr); axis equal tight; clim([0 c_max]);
set(gca, 'FontSize', 11);
xlabel('$j$', 'Interpreter', 'latex'); ylabel('$k$', 'Interpreter', 'latex');
title('Prior $\Gamma_{pr}$', 'Interpreter', 'latex', 'FontSize', 13);
nexttile;
imagesc(Gamma_pos); axis equal tight; clim([0 c_max]);
set(gca, 'FontSize', 11);
xlabel('$j$', 'Interpreter', 'latex');
title('Posterior $\Gamma_{pos}$', 'Interpreter', 'latex', 'FontSize', 13);
nexttile;
imagesc(Gamma_pos_approx_all{r_plot}); axis equal tight; clim([0 c_max]); colorbar;
set(gca, 'FontSize', 11);
xlabel('$j$', 'Interpreter', 'latex');
title(['OLR approx $\hat{\Gamma}_{pos},\ r=' num2str(r_plot) '$'], ...
      'Interpreter', 'latex', 'FontSize', 13);

%% Plot 5: Diagonal variance comparison
figure;
plot(diag(Gamma_pr),  'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', [0.2 0.2 0.8]);
hold on;
plot(diag(Gamma_pos), 'r-o', 'LineWidth', 1.5, 'MarkerFaceColor', [0.8 0.2 0.2]);
legend('Prior variance', 'Posterior variance', 'Interpreter', 'latex', 'FontSize', 12);
xlabel('Parameter index $j$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Variance $\sigma^2_j$', 'Interpreter', 'latex', 'FontSize', 13);
title('Uncertainty per load parameter', 'Interpreter', 'latex', 'FontSize', 14);
box off; set(gca, 'FontSize', 12);

%% Plot 6: Variance reduction bar chart
var_reduction = 1 - diag(Gamma_pos)./diag(Gamma_pr);
figure;
bar(var_reduction, 'FaceColor', [0.2 0.2 0.6], 'EdgeColor', 'none');
xlabel('Parameter index $j$', 'Interpreter', 'latex', 'FontSize', 13);
ylabel('Variance reduction', 'Interpreter', 'latex', 'FontSize', 13);
title('Relative uncertainty reduction per load parameter', ...
    'Interpreter', 'latex', 'FontSize', 14);
ylim([0 1.1]); box off; set(gca, 'FontSize', 12);

%% mean: what did i recover (1D load profiles)
figure;
tiledlayout(1,3,'Padding','compact','TileSpacing','compact');
nexttile;
plot(z_grid, theta_true, 'b-o', 'LineWidth', 1.5, 'MarkerFaceColor', [0.2 0.2 0.8]);
title('True $\theta_{\rm true}$','Interpreter','latex','FontSize',13);
xlabel('$z$','Interpreter','latex'); ylabel('$q(z)$','Interpreter','latex');
box off; set(gca,'FontSize',12);
nexttile;
plot(z_grid, mu_pos, 'r-o', 'LineWidth', 1.5, 'MarkerFaceColor', [0.8 0.2 0.2]);
title('Recovered $\mu_{pos}$','Interpreter','latex','FontSize',13);
xlabel('$z$','Interpreter','latex'); ylabel('$q(z)$','Interpreter','latex');
box off; set(gca,'FontSize',12);
nexttile;
plot(z_grid, mu_pos - theta_true, 'k-o', 'LineWidth', 1.5);
title('Error $\mu_{pos} - \theta_{\rm true}$','Interpreter','latex','FontSize',13);
xlabel('$z$','Interpreter','latex'); ylabel('Error','Interpreter','latex');
box off; set(gca,'FontSize',12);

%% w hat: LIS directions (1D load-space profiles)
figure;
tiledlayout(1, r_plot, 'Padding','compact','TileSpacing','compact');
for i = 1:r_plot
    nexttile;
    plot(z_grid, V(:,i), '-o', 'LineWidth', 1.5, 'Color', [0.2 0.2 0.6]);
    title(['$v_{' num2str(i) '}$'], 'Interpreter','latex','FontSize',13);
    xlabel('$z$','Interpreter','latex','FontSize',11);
    ylabel('weight','FontSize',10);
    box off; set(gca,'FontSize',10);
end
sgtitle('LIS directions in load space', 'Interpreter','latex','FontSize',14);

%% rank r mean convergence (single theta_true)
figure;
semilogy(1:r_max, mean_errors, 'o-', ...
    'LineWidth', 2, 'Color', [0.2 0.2 0.6], ...
    'MarkerFaceColor', [0.2 0.2 0.6], 'MarkerSize', 6);
hold on;
xline(r_plot, '--', 'Color', [0.7 0.1 0.1], 'LineWidth', 1.5);
text(r_plot + 0.2, max(mean_errors)*0.5, ['$r^* = ' num2str(r_plot) '$'], ...
    'Interpreter','latex','FontSize',11,'Color',[0.7 0.1 0.1]);
box off; set(gca,'FontSize',13);
xlabel('Rank $r$', 'Interpreter','latex','FontSize',14);
ylabel('Relative error', 'Interpreter','latex','FontSize',14);
title('Rank-$r$ posterior mean convergence', 'Interpreter','latex','FontSize',14);

%% statistical posterior mean error (Jakob's averaged version)
figure;
semilogy(1:r_max, mean_errors_stat, 'o-', ...
    'LineWidth', 2, 'Color', blue_c, 'MarkerFaceColor', blue_c, 'MarkerSize', 6);
hold on;
xline(r_plot, '--', 'Color', [0.7 0.1 0.1], 'LineWidth', 1.5);
text(r_plot + 0.2, max(mean_errors_stat)*0.5, ['$r^* = ' num2str(r_plot) '$'], ...
    'Interpreter','latex','FontSize',11,'Color',[0.7 0.1 0.1]);
box off; set(gca,'FontSize',13);
xlabel('Approximation rank $r$', 'Interpreter','latex','FontSize',14);
ylabel('Relative posterior mean error', 'Interpreter','latex','FontSize',14);
title(['Statistical posterior mean error (averaged over $N_{\rm rep}=' ...
    num2str(N_rep) '$ draws)'], 'Interpreter','latex','FontSize',14);

%% -----------------------------------------------------------------------
%% local helper: squared Forstner distance in projected coordinates
%% -----------------------------------------------------------------------
function df = foerstner_distance(gamma1, gamma_pos, W)
    % Project both covariances into W'(.)W so the singular force-space
    % covariances become SPD, then squared Forstner distance via generalised EVs.
    Lc = chol(W' * gamma1    * W, 'lower');
    Rc = chol(W' * gamma_pos * W, 'lower');
    tmp = Lc' / Rc';
    s   = svd(tmp);
    lam = s.^2;
    lam(abs(lam) < eps) = [];
    df  = dot(log(lam), log(lam));
end
