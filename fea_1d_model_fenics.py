"""
This script implements the same problem as the
MATLAB script "fea_1d_model.m", but using FEniCS.
Note that x is used for the vertical direction rather 
than z, to match FEniCS's default naming of integration 
measures and mesh coordinates.
"""

from fenics import *

# Define the N-element mesh on the interval
# (0,L), using FEniCS's built-in
# uniform mesh generator for convenience.
L = 1.0
N = 4
mesh = IntervalMesh(N,0,L)

# Restrict to 1-point (degree-zero)
# quadrature to exactly match the MATLAB
# code.  (By default, FEniCS uses heuristics
# to estimate an appropriate degree of
# Gaussian quadrature.)
dx = dx(metadata={"quadrature_degree":0})

# Define properties of the bar as
# anonymous functions of position
# (or "lambda expressions" in Python):
x = SpatialCoordinate(mesh)[0]
E = lambda x : 1.0
A = lambda x : 1.0
rho = lambda x : 1.0
g = 1.0
f = lambda x : rho(x)*A(x)*g

# Discrete function space of continuous
# Galerkin (CG) functions of polynomial
# degree 1 (i.e., linear shape functions):
V = FunctionSpace(mesh,"CG",1)

# Displacement solution:
u = Function(V)

# 1D strain is the 0-th component
# (i.e., x-component) of the gradient.
eps = grad(u)[0]

# Energy functional:
U = 0.5*E(x)*A(x)*(eps**2)*dx
P = f(x)*u*dx
Pi = U - P

# Fixed displacement at x=0:
bc = DirichletBC(V,Constant(0),"near(x[0],0)")

# Minimize energy subject to to the BC, using
# FEniCS's automated computer algebra to
# perform the directional derivative with respect
# to displacement.
solve(derivative(Pi,u)==0,u,bcs=[bc,])

# Print out the solution vector for comparison
# with the MATLAB results; note that FEniCS's
# mapping from nodes to unknowns is in reverse
# order from the MATLAB code.
print(u.vector().get_local())

# Plot the solution:
from matplotlib import pyplot as plt
plot(u)
plt.show()
