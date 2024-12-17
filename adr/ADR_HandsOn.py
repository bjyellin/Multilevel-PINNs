"""

  - Advection-Diffusion-Reaction(u) = f    in the unit square
            u = u_D  on the boundary

No stabilization
"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, 'P', 1)

mu = Constant(1.0)
beta0 = 1000.0
beta1 = 1000.0
beta = Constant((beta0,beta1))
betamag = np.sqrt(beta0*beta0+beta1*beta1)

h = mesh.hmax()

# Define boundary condition
#u_D = Expression('(exp(beta0/mu*x[0])-1)/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1])-1)/(exp(beta1/mu)-1)', \
#  degree=3, beta0=beta0, beta1=beta1, mu=mu)

#uex = Expression('(exp(beta0/mu*x[0])-1)/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1])-1)/(exp(beta1/mu)-1)', \
#  degree=3, beta0=beta0, beta1=beta1, mu=mu)

#dxuex = Expression('beta0*(exp(beta0/mu*x[0]))/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1])-1)/(exp(beta1/mu)-1)', \
#  degree=3, beta0=beta0, beta1=beta1, mu=mu)
#dyuex = Expression('beta1*(exp(beta0/mu*x[0])-1)/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1]))/(exp(beta1/mu)-1)', \
#  degree=3, beta0=beta0, beta1=beta1, mu=mu)


tol = 1E-10

def left(x, on_boundary):
    return on_boundary and x[0] <= tol

def right(x, on_boundary):
    return on_boundary and x[0] >= 1.-tol

def bottom(x, on_boundary):
    return on_boundary and x[1] <= tol

def top(x, on_boundary):
    return on_boundary and x[1] >= 1.-tol


bcl = DirichletBC(V, Constant(1.0), left)
bcr = DirichletBC(V, Constant(0.0), right)
bct = DirichletBC(V, Constant(0.0), top)
bcb = DirichletBC(V, Constant(1.0), bottom)

bcs = [bcl,bcr,bct,bcb]

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
#mu = Constant(1.0)
#beta0 = 10.0
#beta1 = 1000.0
#beta = Constant((10.0,1000.0))
delta = 1.0

sigma =  Constant(0.0)
a = mu*dot(nabla_grad(u), nabla_grad(v))*dx + betamag*h/2*dot(nabla_grad(u), nabla_grad(v))*dx + \
    dot(beta,nabla_grad(u))*v*dx+sigma*u*v*dx # 
    
    
L = f*v*dx

#equation = inner(nabla_grad(u), nabla_grad(v))*dx == f*v*dx

#prm = parameters["krylov_solver"] # short form
#prm["absolute_tolerance"] = 1E-10
#prm["relative_tolerance"] = 1E-6
#prm["maximum_iterations"] = 1000
#set_log_level(DEBUG)
set_log_level(LogLevel.ERROR)


# Compute solution
u = Function(V)
solve(a==L, u, bcs)#, solver_parameters={"linear_solver": "cg", "preconditioner": "ilu"})
info(parameters,True)

# Save solution to file in VTK format
vtkfile = File('adr/HandsOn.pvd')
vtkfile << u

