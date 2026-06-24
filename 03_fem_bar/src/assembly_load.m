%% True distributed load profile
% Bar analogue of build_sigma. It defines the ground truth that the inverse
% problem tries to recover: a spatially distributed axial load q(z) on a
% clamped 1D bar. In the inverse problem the load comes from theta (our
% parameter), one value per interior node.

function [load_func, theta_true, z_grid, x_nodes] = assembly_load(L, nele)

% nodal coordinates of the bar (DEFINES where the state lives)
nnode   = nele + 1;
l       = L / nele;
x_nodes = (0:l:L)';  

% the parameter (distributed load) lives on the interior nodes only:
% the clamped node 1 carries no unknown load DOF
z_grid  = x_nodes(2:end);           % interior nodes (nele x 1)
n       = length(z_grid);

% true distributed-load shape: constant baseline + Gaussian bump at mid-span
q0      = 4e6; % baseline distributed load
bump    = 2e6; % bump amplitude
z_c     = L/2; % bump centre (mid-span)
w       = L/8; % bump width as deviation

q_true  = q0 + bump * exp(-(z_grid - z_c).^2 / (2*w^2));

theta_true = q_true;                 % what we want to infer (n x 1)

% load_func via linear interpolation over the interior nodes
load_func = @(z) interp1(z_grid, q_true, z, 'linear', q0);
end
