%% Building a smooth prior covariance on the vol surface
% uses squared exponential kernel on (S,t) grid ~ financially motivated
% since nearby points in the grid have correlated volatility 
% -> Significant prior - in jakobs implementation prior comes directly from
% the physics (stiffness matrix) of the beam problem. 

function Gamma_pr = build_prior(S_vol_grid, t_vol_grid)
%full list of (S,t) grid points
[S_mat, t_mat] = meshgrid(S_vol_grid, t_vol_grid);
S_pts = S_mat(:);  
t_pts = t_mat(:);  
n = length(S_pts);

% Hyperparameters  financially motivated (Cont & da Fonseca 2002)
alpha = 0.05;   % prior std: vol can deviate ~5% from baseline
ell_S = 30;     % length scale in S: correlation decays over ~30 price units
ell_t = 0.75;   % length scale in t: correlation decays over ~0.75 years

%kernel matrix
Gamma_pr = zeros(n, n);
for j = 1:n
    for k = 1:n
        dS = S_pts(j) - S_pts(k);
        dt = t_pts(j) - t_pts(k);
        Gamma_pr(j,k) = alpha^2 * exp(-dS^2/(2*ell_S^2) ...
            -dt^2/(2*ell_t^2));
    end
end

% Nugget for numerical stability (SPD guarantee)
Gamma_pr = Gamma_pr + 1e-6 * eye(n);

end