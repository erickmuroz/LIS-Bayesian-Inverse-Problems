%% True local volatility surface 
% It takes a stock price S and t as inputs and returns the local volatility
% vol(S,t) at that point. In the inverse problem, the surface comes from
% theta (our parameter)
% for a nice realistic synthetic experiment we should have a shape where
% higher volatility for low strikes and slightly higher for high strikes 

function [sigma_func, theta_true, S_vol_grid, t_vol_grid]= build_sigma(S0)
% input is current stock price
% outputs
% sigma_func - function handle sigma_func(S,t) for fd_solver
% theta_true -  true parameter vector
% S_vol_grid - stock price grid for vol surface
% t_vol_grid - time grid for vol surface

%define (S,t) grid for the vol surface
S_vol_grid = linspace(0.5*S0, 1.5*S0,5); %5 stock price points
t_vol_grid = [0.25, 0.5, 1.0, 1.5, 2.0]; %5 time points

%true vol surface on the grid
[S_mat, t_mat] = meshgrid(S_vol_grid, t_vol_grid);

%volatility desired shape (Martin Haugh shape)
Sigma_true = 0.20 + 0.10 * exp(-(S_mat - S0).^2 / (2*30^2)) ./ sqrt(t_mat);
theta_true = Sigma_true(:);

%get sigma_func via 2D interpolation (......)
sigma_func = @(S,t) interp2(S_vol_grid, t_vol_grid, Sigma_true, S, t, 'linear', 0.20);
end 