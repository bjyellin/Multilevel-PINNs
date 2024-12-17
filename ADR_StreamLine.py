"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

"""

from __future__ import print_function
from fenics import *
import matplotlib.pyplot as plt
import numpy as np

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, 'P', 1)

mu = Constant(1.0)
beta0 = 10.0
beta1 = 100.0
beta = Constant((beta0,beta1))


# Define boundary condition
u_D = Expression('(exp(beta0/mu*x[0])-1)/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1])-1)/(exp(beta1/mu)-1)', \
  degree=3, beta0=beta0, beta1=beta1, mu=mu)

uex = Expression('(exp(beta0/mu*x[0])-1)/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1])-1)/(exp(beta1/mu)-1)', \
  degree=3, beta0=beta0, beta1=beta1, mu=mu)

dxuex = Expression('beta0*(exp(beta0/mu*x[0]))/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1])-1)/(exp(beta1/mu)-1)', \
  degree=3, beta0=beta0, beta1=beta1, mu=mu)
dyuex = Expression('beta1*(exp(beta0/mu*x[0])-1)/(exp(beta0/mu)-1)*(exp(beta1/mu*x[1]))/(exp(beta1/mu)-1)', \
  degree=3, beta0=beta0, beta1=beta1, mu=mu)


def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(0.0)
#mu = Constant(1.0)
#beta0 = 10.0
#beta1 = 1000.0
#beta = Constant((10.0,1000.0))


normb = np.sqrt(beta0*beta0+beta1*beta1)
h = mesh.hmax() #h = CellDiameter(mesh)
print(h)
delta = 1.0

sigma =  Constant(0.0)
a = mu*dot(nabla_grad(u), nabla_grad(v))*dx + dot(beta,nabla_grad(u))*v*dx+sigma*u*v*dx + \
    delta/normb*h*dot(beta,nabla_grad(u))*dot(beta,nabla_grad(v))*dx 
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
solve(a==L, u, bc)#, solver_parameters={"linear_solver": "cg", "preconditioner": "ilu"})
info(parameters,True)
# Plot solution and mesh
#plot(u)
#plot(mesh)
#print('Mesh')
#print(str(mesh))

# Save solution to file in VTK format
vtkfile = File('adrSD/solution.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

err = uex - u
error_L2 = np.sqrt(assemble(err**2*dx(mesh)))
gerrx = grad(u)[0]-dxuex
gerry = grad(u)[1]-dyuex
error_H1 = np.sqrt(assemble((gerrx**2+gerry**2)*dx(mesh)))

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)

error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

"""
coor = mesh.coordinates()
if mesh.num_vertices() == len(vertex_values_u):
  for i in range(mesh.num_vertices()):
    print("u(",coor[i][0],",", coor[i][1], ") =" , vertex_values_u[i])
"""

# Print errors
print('error_L2  =', error_L2)
print('error_H1  =', error_H1)
print('error_max =', error_max)
# Hold plot
#plt.show()
