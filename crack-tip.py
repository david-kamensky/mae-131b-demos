"""
Finite element solution for the stress field around a crack
tip, to illustrate stress singularities.
"""

from dolfin import *
from mshr import *

# Setting up the domain and our finite element
# mesh of it:
N = 16
mesh = UnitSquareMesh(2*N,2*N)
            
# Refine for extra resolution around the
# crack tip:
num_refinements = 6
for i in range(num_refinements):
    markers = MeshFunction("bool", mesh,
                           mesh.topology().dim())
    markers.set_all(False)
    for cell in cells(mesh):
        if(cell.midpoint().distance(Point(0.5,0)) < 0.5**i):
            markers[cell.index()] = True
    mesh = refine(mesh, markers)

# Finite-dimensional space of displacement
# functions to consider in the finite element
# analysis:
V = VectorFunctionSpace(mesh,"CG",1)
x = SpatialCoordinate(mesh)
n = FacetNormal(mesh)

# Displacement field:
u = Function(V)

# Strain and stress in linear elasticity:
def eps(u):
    return sym(grad(u))
# Lam\'e parameters:
lam = Constant(1e2)
mu = Constant(1e2)
# Modified parameters for plane stress problem:
lam_star = 2*lam*mu/(lam + 2*mu)
mu_star = mu
I = Identity(2)
def sigma(eps):
    return lam_star*tr(eps)*I + 2*mu*eps

# Traction on top boundary:
p = Constant(1)
chi_top = conditional(gt(x[1],Constant(1-1e-6)),1.0,Constant(0))
t = chi_top*as_vector([Constant(0),p])

# Potential energy of the structure:
U = 0.5*inner(sigma(eps(u)),eps(u))*dx - dot(t,u)*ds

# Find a stationary point of potential energy, subject to
# symmetry conditions on the left and half of the bottom, to
# model a crack tip at x[0] = 0.5.
bcs = [DirichletBC(V.sub(0),Constant(0),"x[0] < DOLFIN_EPS"),
       DirichletBC(V.sub(1),Constant(0),
                   "x[1] < DOLFIN_EPS && x[0] > 0.5-DOLFIN_EPS")]
solve(derivative(U,u)==0,u,bcs=bcs,
      solver_parameters={"newton_solver":{"linear_solver":"mumps"}})

# Writing displacement solution to a Paraview file
# for visualization:
u.rename("u","u")
File("u.pvd") << u

# Get stress components on each element and
# write out to Paraview files for visualization:
S = FunctionSpace(mesh,"DG",0)
s_xx = project(sigma(eps(u))[0,0],S)
s_yy = project(sigma(eps(u))[1,1],S)
s_xy = project(sigma(eps(u))[0,1],S)
s_xx.rename("s_xx","s_xx")
s_yy.rename("s_yy","s_yy")
s_xy.rename("s_xy","s_xy")
File("s_xx.pvd") << s_xx
File("s_yy.pvd") << s_yy
File("s_xy.pvd") << s_xy

# Same for polar stress components (with
# origin at crack tip):
xp = x - as_vector([0.5,Constant(0)])
r = sqrt(dot(xp,xp))
e_r = as_vector([xp[0]/r,xp[1]/r])
e_t = as_vector([-e_r[1],e_r[0]])
Q = as_matrix([[e_r[0],e_t[0]],
               [e_r[1],e_t[1]]])
sigma_rt = Q.T*sigma(eps(u))*Q
s_rr = project(sigma_rt[0,0],S)
s_tt = project(sigma_rt[1,1],S)
s_rt = project(sigma_rt[0,1],S)
s_rr.rename("s_rr","s_rr")
s_tt.rename("s_tt","s_tt")
s_rt.rename("s_rt","s_rt")
File("s_rr.pvd") << s_rr
File("s_tt.pvd") << s_tt
File("s_rt.pvd") << s_rt
