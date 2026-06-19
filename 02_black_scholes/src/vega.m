%% "Vega" returns the sensitivity of the call price to volatility
% Vega, (part of the "greeks" in finance) is defined as dC/dsigma, it takes the same inputs as bc.pricer and
% returns how sensitive is the call price wrt to changes in volatility
% what we use here is (according to Martin Haugh paper) is the expression
% of vega as a product of S0 qsrt(T) and density function of the normal
% distribution

function v = vega(S0, K, r, sigma, T)
d1 = (log(S0/K) + (r + sigma^2/2)*T) / (sigma*sqrt(T));
v = S0 * sqrt(T) * normpdf(d1);
end