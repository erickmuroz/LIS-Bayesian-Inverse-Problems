%% Finite difference Solver for local volatility surface
% we understand the vol surface as a function of stock price and time
% intead of vol(K,T), now vol(S,t) -> Forward model is no longer a
% closed-form formula, but a solution of a PDE, which is what we were
% missing in the first implentation. 

% The PDE will be solved backwards in time from T to 0 on a grid in (S,t)
% Scheme: backward euler, central differences + upwind scheme
% Space grid: Ns points from 0 to Smax

function price = fd_solver(S0, K, r, T, sigma_func, N_S, N_t)
%sigma_func is the function handle, sigma_func(S,t) returns local vol
%output of the function will be the option price V(S0,0)

%Grid setup
S_max = 3 * S0;
dS = S_max / N_S;
dt = T / N_t;
S = (0:N_S)'* dS;

%Terminal/boundary condition (p)
V = max(S -K,0); %at expiration you know exactly what the option is worth and tells you that if S_T > K you should exercise the option

%Time stepping backwards (BE)
for k = N_t:-1:1
    t_current = k*dt;
    %local vol at current time step 
    sigma_vec = arrayfun(@(s) sigma_func(s, t_current), S);

    %coefficients
    alpha = 0.5 * sigma_vec.^2 .* S.^2 * dt/dS^2 -0.5*r*S*dt/dS;
    beta = 1 + sigma_vec.^2 .* S.^2 *dt/dS^2 + r*dt;
    gamma = 0.5 * sigma_vec.^2 .* S.^2 * dt/dS^2 + 0.5*r*S*dt/dS;

    %interior points
    n_int = N_S - 1;
    dl = -alpha(2:end-1);
    dm = beta(2:end-1);
    du = -gamma(2:end-1);

    A = diag(dm) + diag(dl(2:end), -1) + diag(du(1:end-1), 1);

    rhs = V(2:end-1);

    %bc
    rhs(1) = rhs(1) + alpha(2) *0;
    rhs(end) = rhs(end) + gamma(end-1) * (S_max - K*exp(-r*t_current));

    %solve LS
    V_int = A \ rhs;

    %update
    V(1)= 0;
    V(2:end-1) = V_int;
    V(end) = S_max - K*exp(-r*t_current);
end

price = interp1(S, V, S0, 'linear');
end