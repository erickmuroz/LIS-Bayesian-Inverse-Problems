%% Generate true-state samples for the statistical posterior-mean error
% Instead of a single fixed theta_true, the error is averaged over many random load draws from the prior. 
% This function draws N realisations of the distributed load from the prior, solves the bar (Ku=f) for
% each, and returns the resulting nodal displacement states. From these the
% main script forms noisy measurements y = C u + noise and scores how well the
% rank-r LIS posterior mean reproduces the full-model posterior mean.

%i.e N plausible loads that could realistically exist according to th prior 


function [S_q, S_u] = gen_samples(N, L, nele, EA, BC_dof, mu_q, sigma_q, ell)
% Outputs:
%   S_q      distributed-load samples       (nele x N)
%   S_u      nodal displacement states      (ndof x N), clamped DOF removed

nnode = nele + 1;
l     = L / nele;

%% stiffness matrix K (same assembly as build_G) 
Ke = EA / l;
K  = sparse(nnode, nnode);
K  = K + sparse(1:nele,   1:nele,   Ke, nnode, nnode);
K  = K + sparse(1:nele,   2:nele+1, -Ke, nnode, nnode);
K  = K + sparse(2:nele+1, 1:nele,   -Ke, nnode, nnode);
K  = K + sparse(2:nele+1, 2:nele+1, Ke, nnode, nnode);
K(BC_dof, :) = []; K(:, BC_dof) = [];

%% load mapping operator L_mat (distributed load -> nodal forces) 
% Consistent lumping for linear bar elements: each element load splits l/2 to
% each of its two end nodes.
L_mat = sparse(nnode, nele);
L_mat = L_mat + sparse(1:nele,   1:nele, l/2, nnode, nele);
L_mat = L_mat + sparse(2:nele+1, 1:nele, l/2, nnode, nele);

%% distributed-load prior covariance (exponential kernel) 
% Evaluated at element midpoints, exactly as in build_prior / Jakob.
XQ  = l/2 * (1:2:2*nele);
xq  = XQ(:);
gamma_q = zeros(nele, nele);
for i = 1:nele
    for j = 1:nele
        gamma_q(i, j) = sigma_q^2 * exp(-abs(xq(i) - xq(j)) / ell);
    end
end

%% sample loads from the prior and solve the bar 
xi  = randn(N, nele);
S_q = mu_q .* ones(nele, 1) + chol(gamma_q, 'lower') * xi';   % (nele x N)

F = L_mat * S_q;                 % nodal forces (nnode x N)
F(BC_dof, :) = [];               % clamp

S_u = K \ F;                     % displacement states (ndof x N)
end
