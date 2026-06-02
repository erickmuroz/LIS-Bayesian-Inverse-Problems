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
N_S = 200;
N_t = 200;

price_fd = fd_solver(S0, K, r, T, sigma_flat, N_S, N_t)
price_bs = bs_pricer(S0, K, r, 0.20, T)
%correct fd_solver