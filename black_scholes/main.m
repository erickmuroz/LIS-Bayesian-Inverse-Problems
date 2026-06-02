addpath('src/')

S0 = 100;
K =100;
r = 0.05;
sigma = 0.20;
T = 1;

C = bs_pricer(S0, K, r, sigma, T)
