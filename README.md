# Multilevel-PINNs
How to run these files: 



The following problems are implemented for the multilevel code: 

To run this code and control the parameters used for training, modify the "driver.py" file. 
Change "problem" variable to the problem you would like to solve.

problem = 'discontinuous'



On the domain $\Omega=(-1,1)\times(-1,1)$, 

$-\Delta u = -2(x<y) \text{ on } \Omega$

$u(x,y)=\frac{(x-y)^2}{2}(x<y)\text{ on }\partial\Omega$

-------------

problem = 'nonlinear'

On the domain $\Omega=(-1,1)\times(-1,1)$,

$-\Delta u + u^3 = e^{3(x+y)}-2e^{x+y} \text{ on }\Omega$

$u(x,y) = e^{x+y} \text{ on } \partial\Omega$

You can choose the number of interpolation points where the losses are computed. 


**To Do: Describe which losses are computed on which points**

-------------

The following problems are implemented for the randomly initialized code: 

To run this code and control the parameters used for training, modify the "random_driver.py" file. 

problem = 'discontinuous'

problem = 'nonlinear'

problem = 'adr'

  On the domain $\Omega=(0,1)\times(0,1)$ let $\Gamma$ denote the boundary. 
  
  $\mu\Delta u + \vec{\beta}\cdot\nabla u=0$
  
  $u(x,y)=0 \text{ on } \Gamma_{left}$
  
  $u(x,y)=0 \text{ on } \Gamma_{bottom}$
  
  $u(x,y)=1 \text{ on } \Gamma_{top}$
  
  $u(x,y)=1 \text{ on } \Gamma_{right}$

Inside of driver.py, you can modify the weights on the different terms of the loss function 

To change the source of data for training, modify the data_source variable. 

data_source = {0,1,2}
- data_source = 0 (samples noisy true solution to simulate real data)
- data_source = 1 (samples the finite element solution)
  - This is implemented for the discontinuous case and the stabilized advection diffusion case
- data_source = 2 (samples no data and just uses the equations to solve the PDE as a forward problem)


