"""
A rectangular column hanging and subjected to self weight.  
The exact solution for a traction BC at the top is given in
the textbook, while other BCs can be applied to illustrate the
St. Venant principle.  
"""

from dolfin import *

aspect_ratio = 8
N = 8
mesh = BoxMesh(Point(0,0,0),Point(1,1,-aspect_ratio),N,N,aspect_ratio*N)
x = SpatialCoordinate(mesh)
V = VectorFunctionSpace(mesh,"CG",1)

u = Function(V)
eps = sym(grad(u))
I = Identity(3)
mu = Constant(1e1)
lmbda = Constant(1e1)
sigma = 2*mu*eps + lmbda*tr(eps)*I
f_mag = Constant(2)
f = as_vector([0,0,-f_mag])

# Symmetry BCs (all cases):
bcs = [DirichletBC(V.sub(0),Constant(0),"x[0]<DOLFIN_EPS"),
       DirichletBC(V.sub(1),Constant(0),"x[1]<DOLFIN_EPS")]

# BCs for three cases; leave only one un-commented #######

# Case 1: Traction BCs
vol = Constant(assemble(1*dx(domain=mesh)))
area = 1.0
t_mag = Constant(vol*f_mag/area)
top_char = conditional(gt(x[2],-DOLFIN_EPS),1.0,Constant(0))
t = top_char*as_vector([0,0,t_mag])
bcs += [DirichletBC(V.sub(2),Constant(0),
                    "x[0]<DOLFIN_EPS && x[1] < DOLFIN_EPS "
                    +"&& x[2] > -DOLFIN_EPS","pointwise"),]

# Case 2: Sliding BCs
#t = Constant((0,0,0))
#bcs += [DirichletBC(V.sub(2),Constant(0),"x[2] > -DOLFIN_EPS"),]

# Case 3: Glued BCs
#t = Constant((0,0,0))
#bcs += [DirichletBC(V,Constant((0,0,0)),"x[2] > -DOLFIN_EPS"),]

# End of cases ###########################################

# Energy minimization:
U = (0.5*inner(sigma,eps) - dot(f,u))*dx - dot(t,u)*ds
solve(derivative(U,u)==0,u,bcs=bcs)

# Write to Paraview file for visualization:
u.rename("u","u")
File("u.pvd") << u
