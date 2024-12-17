from fine_train_random_guess3 import *

#True data: data_source = 0
#FEM data: data_source = 1
#No data: data_source = 2

problem = 'adr'
num_interpolation_points = 60
data_source = 1
physics_weight = 1
boundary_weight = 2
data_weight = 1

num_epochs = 1000

run_random_fine(problem, num_interpolation_points, data_source, num_epochs, physics_weight, boundary_weight, data_weight)