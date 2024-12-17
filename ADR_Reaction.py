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
import lumping_scheme

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 2)

mu = Constant(0.001)
beta0 = 0.0
beta1 = 0.0
beta = Constant((beta0,beta1))
sigma =  Constant(100.0)


# Define boundary condition
u_D = Expression('exp(sqrt(sigma/2*mu)*(x[0]+x[1]))', \
  degree=3, sigma=sigma, mu=mu)

uex = Expression('exp(sqrt(sigma/2*mu)*(x[0]+x[1]))', \
  degree=3, sigma=sigma, mu=mu)

dxuex = Expression('sqrt(sigma/2*mu)*exp(sqrt(sigma/2*mu)*(x[0]+x[1]))', \
  degree=3, sigma=sigma, mu=mu)

dyuex = Expression('sqrt(sigma/2*mu)*exp(sqrt(sigma/2*mu)*(x[0]+x[1]))', \
  degree=3, sigma=sigma, mu=mu)


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

#a = mu*dot(nabla_grad(u), nabla_grad(v))*dx + dot(beta,nabla_grad(u))*v*dx+sigma*u*v*dx
#a = mu*dot(nabla_grad(u), nabla_grad(v))*dx + dot(beta,nabla_grad(u))*v*dx+\
#    sigma*u*v*dx(scheme="vertex", metadata={"degree":1, "representation":"quadrature"}) 
a = mu*dot(nabla_grad(u), nabla_grad(v))*dx + dot(beta,nabla_grad(u))*v*dx+\
    sigma*u*v*dx(scheme="lumped", degree=2)
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
print('Mesh')
print(str(mesh))

# Save solution to file in VTK format
vtkfile = File('adr/solution_Reaction_P2ML.pvd')
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
