"""
FEniCS tutorial demo program: Poisson equation with Dirichlet conditions.
Test problem is chosen to give an exact solution at all nodes of the mesh.

  -Laplace(u) = f    in the unit square
            u = u_D  on the boundary

  u_D = 1 + x^2 + 2y^2
    f = -6
"""

from __future__ import print_function
from fenics import *
from dolfin import *
import matplotlib.pyplot as plt
import numpy as np
import math
import lumping_scheme

# Create mesh and define function space
mesh = UnitSquareMesh(8, 8)
V = FunctionSpace(mesh, 'P', 1)

# Define boundary condition
u_D = Expression('1 + x[0]*x[0] + 2*x[1]*x[1]', degree=2)

def boundary(x, on_boundary):
    return on_boundary

bc = DirichletBC(V, u_D, boundary)

# Define variational problem
u = TrialFunction(V)
v = TestFunction(V)
f = Constant(-6.0)
mu = Constant(0.00001)
beta0 = 0.0
beta1 = 000.0
beta = Constant((beta0,beta1))

sigma =  Constant(10000.0)
a = mu*dot(nabla_grad(u), nabla_grad(v))*dx + dot(beta,nabla_grad(u))*v*dx+\
    sigma*u*v*dx(scheme="vertex", metadata={"degree":1, "representation":"quadrature"}) 
L = f*v*dx



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
vtkfile = File('adrML/solution_lumped.pvd')
vtkfile << u

# Compute error in L2 norm
error_L2 = errornorm(u_D, u, 'L2')

# Compute maximum error at vertices
vertex_values_u_D = u_D.compute_vertex_values(mesh)
vertex_values_u = u.compute_vertex_values(mesh)
import numpy as np
error_max = np.max(np.abs(vertex_values_u_D - vertex_values_u))

"""
coor = mesh.coordinates()
if mesh.num_vertices() == len(vertex_values_u):
  for i in range(mesh.num_vertices()):
    print("u(",coor[i][0],",", coor[i][1], ") =" , vertex_values_u[i])
"""

# Print errors
#print('error_L2  =', error_L2)
#print('error_max =', error_max)

# Hold plot
#plt.show()
#plt.ion
#plt.interactive('b')
