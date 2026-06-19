%% Closed-form formula for Call prices (testing)
% simplest possible piece, introducing a bit the inputs of what the formula
% will do
%% Inputs
% S0: Current stock/asset price
% K: Strike price
% r: risk free rate
% sigma: volatility (crucial, accounts for uncertainty in our inverse prob
% T: time to expiration 
%% Output
% C: call option price 

%eventually bs_pricer.m will be called inside build_G.m, to evaluate our
%expensive F(theta), now is just the pricer

%% Computation of the Black Scholes European Call option price
function C = bs_pricer(S0, K, r, sigma, T)
d1 = (log(S0/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T));
d2= d1 -sigma*sqrt(T);

C = S0 * normcdf(d1) - K*exp(-r*T) * normcdf(d2);
end