%% LIS first implementation approach
% small example: y=G*x + noise 
% n=parameter space dimension
% m=observation space dimension

clear all 
clc



%% 1. dimensions, defining where x and y will live R^
n = 6;
m = 4;

%% 2. matrices
%G (m x n) mapping parameter to observations (x-> y)
rng(42)
G = randn(m,n);

%Gamma_pr (n x n)
%requirement of SPD, built as A'A + I to guarantee - no physical knowledge
%of the parameters 
A = randn(n,n);
Gamma_pr = A'*A + eye(n);

%Gamma_obs (m x m)
Gamma_obs = diag([0.5, 0.5, 10.0, 10.0]);

%% 3. exact posterior covariance
Gamma_obs_inv = diag(1./diag(Gamma_obs));
H = G' * Gamma_obs_inv * G; %hessian of (-)log-likelihood
Gamma_pr_inv = inv(Gamma_pr);
Gamma_pos = inv(H + Gamma_pr_inv);

%% 4. generalized EV problem (SVD)
%cholesky of Gamma_pr - assuming square root factorization
S_pr = chol(Gamma_pr, 'lower'); %st. Gamma_pr = S_pr * S_pr'
S_obs_inv = diag(1./sqrt(diag(Gamma_obs)));

%remark 4 
A_lis = S_obs_inv * G * S_pr; %taking SVD of A_lis solves the EVproblem of Htilde
[U, Delta, Z] = svd(A_lis, 'econ'); %gives the zi (lives in transformed space)
delta = diag(Delta); %containing singular values descending 

%eigenvectors in parameter space 
W_hat = S_pr* Z; %to undo the change of variables, now in initial parameter space 

%% 5. build the approximate posterior covariance (Theorem 2.3)
% here its showed the "loss function" minimization made concrete
r_max =length(delta); 

for r = 1:r_max
    Gamma_pos_approx = Gamma_pr;
    for i= 1:r
        w = W_hat(:,i); %output of
        d = delta(i);   %solving the minimization problem of the Loss functions 
        Gamma_pos_approx = Gamma_pos_approx - (d^2 / (1 + d^2)) * (w * w');
    end
    Gamma_pos_approx_all{r} = Gamma_pos_approx;
end 

%% 6. approximation vs real (Frobenius and Forstner)
frob_errors = zeros(r_max,1); %Pre Allocation 
forstner_errors = zeros(r_max, 1); %each will store the error for one rank r

for r = 1:r_max
    Gamma_approx = Gamma_pos_approx_all{r};
    frob_errors(r) = norm(Gamma_pos - Gamma_approx, 'fro');
    %Forstner as in paper sqrt(sum(log(lambda_i)^2) 
    lambda = eig(Gamma_approx,Gamma_pos); %generalized ev of the pencil ()
    forstner_errors(r) = sqrt(sum(log(lambda).^2));
end

figure;
plot(1:r_max, frob_errors, 'b-o', 'LineWidth',2); hold on;
plot(1:r_max, forstner_errors, 'r-o', 'LineWidth',2);
xlabel('rank r');
ylabel('error');
legend('Frobenius', 'Forstner metric');
title('Approximation error vs ranr r (Thm2.3)');
grid on;

%% 7. directions spectrum 
% remark: delta_i^2 are the ev of (H, Gamma_pr_inv)
figure;
stem(1:r_max, delta.^2, 'b', 'LineWidth', 2, 'MarkerSize', 8, ...
    'MarkerFaceColor', 'b');
hold on;
yline(1, 'r--', 'LineWidth', 2, 'Label', 'threshold \delta^2 = 1');
set(gca, 'YScale', 'log');
xlabel('direction i');
ylabel('\delta_i^2  (log scale)');
title('Eigenvalue spectrum — pencil (H, \Gamma_{pr}^{-1})');
grid on;
%% 8. optimal projector corollary 3.2 
r = 2;
%W_tilde = Gamma_pr_inv * W_hat 
W_tilde = Gamma_pr_inv * W_hat;

%build Pr
P_r = zeros(n, n);
for i = 1:r
    P_r = P_r + W_hat(:, i) * W_tilde(:,i)';
end
oblique_error =  norm(P_r^2 - P_r, 'fro');
disp(['Oblique projector error ||Pr^2 - Pr|| =' num2str(oblique_error)])

%reduced forward operator Gr
G_r = G * P_r;

%posterior with Gr 
H_r = G_r' * Gamma_obs_inv * G_r;
Gamma_pos_projected = inv(H_r + Gamma_pr_inv);

%Aproximation2.3 vs. Projected
frob_error_projected = norm(Gamma_pos_projected - Gamma_pos_approx_all{r}, 'fro');
disp(['Frobenius error between approximation and projected covariance = ' num2str(frob_error_projected)]);
%the optimal rank r posterior approximation from theorem 2.3 is identical
%to what you get it you simply project the forward operator onto the r most
%likelihood informed directions, two completely different paths, to the
%same result