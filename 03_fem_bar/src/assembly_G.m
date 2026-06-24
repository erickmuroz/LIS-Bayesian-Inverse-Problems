%% Assembly of the forward operator G for the static structural bar problem
% Follows Jakob Scheffels' construction (LIP_Setup.m / febar.m).
%
% Physics. A clamped 1D bar in static equilibrium:
% observation model y = C u + e s.t 
%   K u = f  =>  u = K^{-1} f,
% 
% and a subset of m nodal displacements is observed: y = C u. The composite
% forward operator mapping the unknown load DOFs to the observations is
%   G = C K^{-1},  built as  G = (K \ C')'.
%
% K \ C' solves K X = C' for the matrix X = G'. Each column of C' is the
% indicator vector of one sensor node. 
%
% This version also returns L_mat (the load mapping operator) and the
% clamped stiffness K, because prior construction pushes the load
% covariance through L_mat (map for mean and Gamma)

function [G, contracts] = assembly_G(L, nele, EA, m, BC_dof, sensor_seed)
nnode = nele + 1;
l     = L / nele;

%% stiffness matrix K (linear bar elements, identical to Jakob's assembleK)
Ke = EA / l;
K  = sparse(nnode, nnode);
K  = K + sparse(1:nele,   1:nele,    Ke, nnode, nnode);
K  = K + sparse(1:nele,   2:nele+1, -Ke, nnode, nnode);
K  = K + sparse(2:nele+1, 1:nele,   -Ke, nnode, nnode);
K  = K + sparse(2:nele+1, 2:nele+1,  Ke, nnode, nnode);

%% load mapping operator L_mat (distributed load -> nodal forces)
L_mat = sparse(nnode, nele);
L_mat = L_mat + sparse(1:nele,   1:nele, l/2, nnode, nele);
L_mat = L_mat + sparse(2:nele+1, 1:nele, l/2, nnode, nele);

%%  sensor placement (random interior nodes, Jakob's LIP_Setup strategy)
rng(sensor_seed);
m_pos = randi([1, nnode], 1, m);
m_pos = unique(m_pos);
m_pos = setdiff(m_pos, BC_dof);
while numel(m_pos) ~= m
    m_pos = [m_pos, randi([2, nele], 1, m - numel(m_pos))];
    m_pos = unique(m_pos);
    m_pos = setdiff(m_pos, BC_dof);
end
m_pos = sort(m_pos);

%% observation operator C (m x nnode), for later build of G
C = zeros(m, nnode);
for i = 1:m
    C(i, m_pos(i)) = 1;
end

%% apply clamp: remove fixed DOF from K, C and L_mat
% K: drop fixed row and column;  C: drop fixed column;  L_mat: drop fixed row.
K(BC_dof, :) = []; K(:, BC_dof) = [];
C(:, BC_dof) = [];
L_mat(BC_dof, :) = [];

%%  forward operator G = C K^{-1}, built as (K \ C')'
G  = (K \ C')';
G  = full(G);

%% sensor info and operators needed downstream
x_nodes        = (0:l:L)';
contracts.node  = m_pos(:);
contracts.z     = x_nodes(m_pos);
contracts.C     = C;
contracts.L_mat = L_mat;    
contracts.K     = K;            
end
