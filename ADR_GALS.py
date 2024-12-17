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
import torch 
import pandas as pd 

# Create mesh and define function space
mesh = UnitSquareMesh(16, 16)
V = FunctionSpace(mesh, 'P', 2)

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
h = CellDiameter(mesh)
print(h)
delta = 1.0
tau = delta*h/normb

sigma =  Constant(0.0)
a = mu*dot(nabla_grad(u), nabla_grad(v))*dx + dot(beta,nabla_grad(u))*v*dx+sigma*u*v*dx + \
    tau*(-mu*div(nabla_grad(u))+dot(beta,nabla_grad(u))+sigma*u)*\
        (-mu*div(nabla_grad(v))+dot(beta,nabla_grad(v))+sigma*v)*dx
L = f*v*dx + tau*f*(-mu*div(nabla_grad(v))+dot(beta,nabla_grad(v))+sigma*v)*dx
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
vtkfile = File('adrGALS/solution.pvd')
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

#Generate samples from GALS finite element solution 
num_interpolation_points = 10
x_vals = np.linspace(0, 1, num_interpolation_points)
y_vals = np.linspace(0, 1, num_interpolation_points)

print("about to define meshgrid")

#Use meshgrid to generate a grid of x,y values
x, y = np.meshgrid(x_vals, y_vals)




#Keep only points on the interior to compute loss on 
# Create a mask to select only interior points (0 < x < 1 and 0 < y < 1)
print("about to define interior mask")
interior_mask = (x > 0) & (x < 1) & (y > 0) & (y < 1)
x_interior = x[interior_mask]
y_interior = y[interior_mask]

#Evaluate u at all of the x,y values
print("about to define vectorized u")
u_vectorized = np.vectorize(u)
u_values_interior = u_vectorized(x_interior, y_interior)

print("about to initialize gals_solution tensor")
gals_solution = np.zeros((u_values_interior.shape[0], 3))
# breakpoint()
gals_solution[:,0] = x_interior.flatten()
gals_solution[:,1] = y_interior.flatten()
gals_solution[:,2] = u_vectorized(x_interior,y_interior).flatten()

gals_df = pd.DataFrame(gals_solution)

gals_df.to_csv('gals_solution.csv', index=True)
breakpoint()
gals_tensor = torch.tensor(gals_solution,dtype=torch.float32, requires_grad=True)
# breakpoint()
# for j in range(len(x)):
#     for i in range(len(x[0])):      
#       gals_solution[i, 2] = u(x[i], y[i])
# breakpoint()