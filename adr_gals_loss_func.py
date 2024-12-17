import torch
import numpy as np
from nn import *
from ADR_GALS import *
# from fem import data_points

#Define the loss function on the interior of the domain to calculate
#how closely the learned function solves the PDE you are trying to solve
def adr_gals_loss_func(x,y):

    beta0 = 10.0
    beta1 = 100.0
    mu = 1
    
    u = model(torch.cat((x,y), dim=1))
    # breakpoint()
    n = len(u)

    torch.autograd.set_detect_anomaly(True)

    u2 = u.clone()

    u_x = torch.autograd.grad(u, x,
                            grad_outputs=torch.ones_like(u),
                            retain_graph=True,
                            create_graph=True,
                            allow_unused=True)[0]

    u_x_clone = u_x.clone()

    u_xx = torch.autograd.grad(
        u_x_clone, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True,

    )[0]

    u_xx_clone = u_xx.clone()

    u_y = torch.autograd.grad(
        u2, y,
        grad_outputs=torch.ones_like(u2),
        retain_graph=True,
        create_graph=True,
        allow_unused=True
    )[0]

    u_y_clone = u_y.clone()

    u_yy = torch.autograd.grad(
        u_y_clone, y,
        grad_outputs=torch.ones_like(u2),
        retain_graph=True,
        create_graph=True
    )[0]

    u_yy_clone = u_yy.clone()

    return mu*(u_xx+u_yy)+beta0*u_x+beta1*u_y

#Change this so the loss func is only called on the interior points

print(adr_gals_loss_func(gals_tensor[:,0].reshape(-1,1),gals_tensor[:,1].reshape(-1,1)))
breakpoint()
    # if problem == 'discontinuous': 
    #     def f_true(x,y,problem):
    #         return -2*(x<y)
    #     return torch.norm(-torch.squeeze(u_xx_clone)-torch.squeeze(u_yy_clone)-torch.squeeze(f_true(x,y,problem)))
    
    # if problem == 'nonlinear':
    #     def f_true(x,y,problem):
    #         return torch.exp(3*(torch.tensor(x)+torch.tensor(y)))-2*torch.exp(torch.tensor(x)+torch.tensor(y))
    #     return torch.norm(-torch.squeeze(u_xx)-torch.squeeze(u_yy)+torch.squeeze(u)**3-torch.squeeze(f_true(x,y,problem)))
