# Multilevel-PINNs
How to run these files: 



The following problems are implemented for the multilevel code: 

To run multilevel code: 
Run python3 driver.py.
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

-------------

problem = 'adr'

On the domain $Omega=(0,1)\times(0,1)$, 

$\vec{beta}$


Inside of driver.py, you can modify the weights on the different terms of the loss function 



