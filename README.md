# Multilevel-PINNs
How to run these files: 

To run multilevel code: 
Run python3 driver.py

The following problems are implemented for the multilevel code: 
On the domain $\Omega=(-1,1)\times(-1,1)$, 
\begin{align}
-\Delta u &= -2(x<y) \text{ on } \Omega\\
u(x,y)&=\frac{(x-y)^2}{2}(x<y)\text{ on }\partial\Omega
\end{align}

Inside of driver.py, you can modify the weights on the different terms of the loss function 



