import torch
import numpy as np
from nn import *
from problem import *


#Define the loss function on the interior of the domain to calculate
#how closely the learned function solves the PDE you are trying to solve
def interior_loss_func2(x,y,samples_f,model):
    x.requires_grad = True
    y.requires_grad = True
    # print(x.shape)
    # print(y.shape)
    # Define model. Store data in an n x 2 array where each row is a data point

    # I need to initialize both u and sigma in the training
    # because they both need to be learned
    v = model(torch.cat((x, y), dim=1))

    n = len(v)

    u = v[:,0]
    sigma = v[:,1]

    # sigma = sigma_model(torch.cat((x,y),dim=1))

    u = torch.squeeze(u)
    sigma = torch.squeeze(sigma)

    # Take derivatives that appear in the heat equation
    u_x = torch.autograd.grad(
        u, x,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]

    u_y = torch.autograd.grad(
        u, y,
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )[0]



    # H1_norm = torch.sum(u**2) +


    # Now include the derivatives of sigma that I will need
    div_sigma = torch.autograd.grad(
        sigma, x,
        grad_outputs=torch.ones_like(sigma),
        retain_graph=True,
        create_graph=True)[0]+ torch.autograd.grad(sigma, y, grad_outputs=torch.ones_like(sigma), retain_graph=True,
                                                 create_graph=True)[0]
    true_f = f_true(x,y)

    # print("sigma is ", sigma)

    # Extracting individual gradients from the tuple
    grad_u_x, grad_u_y = torch.autograd.grad(
        u, [x, y],
        grad_outputs=torch.ones_like(u),
        retain_graph=True,
        create_graph=True
    )

    # print("grad_u_x: ",grad_u_x)
    # print("grad_u_y: ", grad_u_y)

    # Calculating the interior loss using the gradients
    # interior_loss = torch.norm(
    #     (torch.squeeze(sigma) - (torch.squeeze(grad_u_x) + torch.squeeze(grad_u_y)))) ** 2 + torch.norm(
    #     (torch.squeeze(div_sigma) + torch.squeeze(f_true(x, y))))

    # print("In interior loss func")
    # print("sigma.shape: ", sigma.shape)
    # print("torch.squeeze(u_x) + torch.squeeze(u_y)) shape: ", (torch.squeeze(u_x) + torch.squeeze(u_y)).shape)
    # print("torch squeeze div_sigma shape: ", torch.squeeze(div_sigma).shape)
    # print("f_true(x,y) shape", torch.squeeze(f_true(x,y)).shape)
    # print("Whole first term squared shape: ",(torch.squeeze(sigma) - (torch.squeeze(u_x) + torch.squeeze(u_y))).shape)
    # print("_________________________________")


    #12/19 6:30 pm : Changed the +torch.squeeze(f_true(x,y) to -torch.squeeze(f_true(x,y) because I think that's how the calculus should work out
    return torch.norm((torch.squeeze(sigma) - (torch.squeeze(grad_u_x) + torch.squeeze(grad_u_y))))**2 + torch.norm((-torch.squeeze(div_sigma) - torch.squeeze(f_true(x,y))))
    # print("The shape of this object is ",(torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)).shape)
    # print("the problem term is ", (torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True)))

    # interior_loss = torch.norm((torch.squeeze(sigma) - (torch.squeeze(grad_u_x) + torch.squeeze(grad_u_y))))**2 + torch.norm((torch.squeeze(div_sigma) + torch.squeeze(f_true(x, y))))
    #
    # return interior_loss
    # return torch.norm((torch.squeeze(sigma) - torch.squeeze((torch.autograd.grad(u, [x, y], grad_outputs=torch.ones_like(u), retain_graph=True, create_graph=True))))) ** 2 + torch.norm(
    #     (torch.squeeze(div_sigma) + torch.squeeze(f_true(x, y))))