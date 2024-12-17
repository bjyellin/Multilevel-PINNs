from coarse_train3 import *
from fine_train3 import *

#Examples to run

#No data: Fine training only with physics, no data

#Data assimilation tests
#Fine training with coarse data (from fine_train_random_guess3)

#True data: data_source = 0
#FEM data: data_source = 1
#No data: data_source = 2

problem = 'discontinuous'
num_interpolation_points = 5
data_source = 1
coarse_physics_weight = 1000
coarse_data_weight = 100
fine_physics_weight = 100
boundary_weight = 100

num_coarse_epochs = 2500
num_fine_epochs = 500

run_multilevel_coarse(problem, num_interpolation_points, data_source, num_coarse_epochs, coarse_physics_weight, boundary_weight)

run_multilevel_fine(problem, num_interpolation_points, data_source, num_fine_epochs, fine_physics_weight, boundary_weight, coarse_data_weight)


