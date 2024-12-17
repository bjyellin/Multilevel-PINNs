from nn import *
from problem import *

import torch

def dirichlet_boundary_loss_func(x,y,model,problem):

  if problem == 'discontinuous':
    
    def g_true(x, y, problem):
      return (torch.tensor(x)-torch.tensor(y))*(torch.tensor(x)-torch.tensor(y))/2.0*(x<y)

    torch.set_default_dtype(torch.float32)
    # breakpoint()
    pred = model(torch.cat((x.double(), y.double()),dim=1))
    # print("dtype of x is ", x.dtype)
    
    pred = torch.squeeze(pred)
    g_val = torch.squeeze(g_true(x, y, problem))
    return (torch.norm(pred-g_val)**2)
  
  if problem == 'nonlinear': 
    
    def g_true(x,y,problem):
      return torch.exp(torch.tensor(x)+torch.tensor(y))

    torch.set_default_dtype(torch.float32)
    pred = model(torch.cat((x,y),dim=1))
    pred = torch.squeeze(pred)
    g_val = torch.squeeze(g_true(x, y, problem))
    return (torch.norm(pred-g_val)**2)
  
  if problem == 'adr': 
    def g_true(x,y,problem):
    
      bc = torch.zeros_like(x)
      x_min = 0
      x_max = 1
      y_min = 0
      y_max = 1
      #Left boundary
      bc = torch.where(x==x_min, torch.tensor(0.0, device=x.device), bc)

      #Right boundary
      bc = torch.where(x == x_max, torch.tensor(1.0, device=x.device), bc)

      # Bottom boundary: y = y_min
      bc = torch.where(y == y_min, torch.tensor(0.0, device=y.device), bc)

      # Top boundary: y = y_max
      bc = torch.where(y == y_max, torch.tensor(1.0, device=y.device), bc)

      return bc 
    # print("x shape ", x.shape)
    pred = model(torch.cat((x.reshape(-1,1),y.reshape(-1,1)),dim=1))
    g_val = torch.squeeze(g_true(x,y,problem))

    return torch.norm(pred-g_val)**2