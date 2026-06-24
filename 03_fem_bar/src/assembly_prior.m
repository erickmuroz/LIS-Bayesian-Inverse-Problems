%% Building the prior covariance on the nodal load (Jakob's construction)
% Bar analogue of the Black-Scholes build_prior, but following  LIP_Setup.m exactly.
% The exponential (Ornstein-Uhlenbeck) kernel is placed on
% the DISTRIBUTED LOAD q at the element quadrature points, giving gamma_q.
% 
% The nodal-force prior is then obtained by pushing that covariance through the
% consistent load operator L_mat:
%
% Gamma_pr = L_mat * gamma_q * L_mat'.
%
% -> Significant difference from the direct-on-nodes version: because L_mat maps
% a higher-dimensional load field to nodal forces, Gamma_pr is SINGULAR (rank
% deficient).
%
%Gamma_pr will be intrinsically smooth due to construction, so information will collapse into
%very few directions


function [Gamma_pr, S_pr, mu_f] = assembly_prior(nele, L_mat, l)

% Hyperparameters
mu_q    = 4e6; % prior mean of the distributed load (will be then mapped via L_mat) 
sigma_q = 0.3 * mu_q;% prior std (~30% of baseline)
ell     = 0.3;% correlation length along the bar

% quadrature points (exponential kernel needs to be avaluated at specific
% coordinates along the bar to build the gamma_q)
XQ = l/2 * (1:2:2*nele);
xq = XQ(:);

% exponential kernel on the distributed load q
gamma_q = zeros(nele, nele);
for i = 1:nele
    for j = 1:nele
        gamma_q(i,j) = sigma_q^2 * exp(-abs(xq(i) - xq(j)) / ell);
    end
end

% map to the nodal force covariance via L_mat 
Gamma_pr = L_mat * gamma_q * L_mat';

% square-root factor for the singular prior 
S_pr = L_mat * chol(gamma_q, 'lower');

% prior mean nodal force from the mu_q also via L_mat
mu_f = L_mat * (mu_q .* ones(nele, 1));
end
