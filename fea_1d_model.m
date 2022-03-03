% This MATLAB script implements 1D FEA with linear
% shape functions, for the problem of an axial 
% deformation member hanging from a fixed support
% under its own weight.  A corresponding FEniCS
% implementation is given in the file
% "fea_1d_model_fenics.py".

% Properties of the bar:
E = @(z)(1.0);
A = @(z)(1.0);
rho = @(z)(1.0);
% Force per unit length from self weight:
g = 1.0;
f = @(z)(rho(z)*A(z)*g);
% Finite element mesh:
z_nodes = [0,0.25,0.5,0.75,1];
N_node = numel(z_nodes);
L = z_nodes(N_node);
N_el = N_node-1;
% In 2D or 3D, this would come from a 
% data file determining mesh connectivity.
el_node_map = zeros(N_el,2);
for i=1:N_el
    el_node_map(i,1) = i;
    el_node_map(i,2) = i+1;
end
% Initialize stiffness matrix and
% load vector to zero:
K = zeros(N_node,N_node);
F = zeros(N_node,1);
% Loop over elements:
for el=1:N_el
    % Nodes corresponding to this
    % element:
    node1 = el_node_map(el,1);
    node2 = el_node_map(el,2);
    z1 = z_nodes(node1);
    z2 = z_nodes(node2);
    h = z2-z1;
    % One-point quadrature:
    z_mid = 0.5*(z1+z2);
    phi1 = 0.5; % (midpoint evaluation)
    phi2 = 0.5;
    phi1_prime = -1/h; % (constant derivative)
    phi2_prime = 1/h;
    EA = E(z_mid)*A(z_mid);
    % Add element's integrals to stiffness matrix 
    % and load vector:
    K(node1,node1) = K(node1,node1) + h*EA*phi1_prime*phi1_prime;
    K(node1,node2) = K(node1,node2) + h*EA*phi2_prime*phi1_prime;
    K(node2,node1) = K(node2,node1) + h*EA*phi1_prime*phi2_prime;
    K(node2,node2) = K(node2,node2) + h*EA*phi2_prime*phi2_prime;
    F(node1) = F(node1) + h*f(z_mid)*phi1;
    F(node2) = F(node2) + h*f(z_mid)*phi2;
end
% Apply boundary condition at z=0 (node 1):
F(1) = 0;
K(1,:) = 0;
K(1,1) = 1;
% Solve system:
alpha = K\F;
% Plot the result:
plot(z_nodes,alpha);

% Exact solution for E=A=rho=g=1; not
% relevant if the coefficients are modified.
u_exact = @(z)(0.5*z.*(2*L-z));
z_fine = linspace(0,L,10000);
hold on;
plot(z_fine,u_exact(z_fine));
