import torch
import torch.nn as nn
from interior_loss_func import *
# from interior_loss_func2 import *
from dirichlet_boundary_loss_func import *
from fem import *
from coarse_train3 import *

import timeit

import matplotlib.pyplot as plt


def run_multilevel_fine(problem, num_interpolation_points, data_source, num_fine_epochs, physics_weight = 20, boundary_weight = 100, coarse_data_weight=20):


    def interior_loss_func(x,y,samples_f,model,problem):
        x = torch.tensor(x, dtype=torch.float64)
        y = torch.tensor(y, dtype=torch.float64)
        x.requires_grad = True
        y.requires_grad = True

        u = model(torch.cat((x,y), dim=1))

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


        if problem == 'discontinuous': 
            def f_true(x,y,problem):
                return -2*(x<y)
            return torch.norm(-torch.squeeze(u_xx_clone)-torch.squeeze(u_yy_clone)-torch.squeeze(f_true(x,y,problem)))
        
        if problem == 'nonlinear':
            def f_true(x,y,problem):
                return torch.exp(3*(torch.tensor(x)+torch.tensor(y)))-2*torch.exp(torch.tensor(x)+torch.tensor(y))
            return torch.norm(-torch.squeeze(u_xx)-torch.squeeze(u_yy)+torch.squeeze(u)**3-torch.squeeze(f_true(x,y,problem)))

        if problem == 'adr':
            beta0 = 10.0
            beta1 = 100.0
            mu = 1
            # breakpoint()
            return torch.norm(mu*(u_xx+u_yy) + beta0*u_x + beta1*u_y)



    print("Running the fine training")
    class NeuralNetwork(nn.Module):
        def __init__(self):
                super(NeuralNetwork, self).__init__()
                torch.set_default_dtype(torch.float64)

                self.flatten = nn.Flatten()
                #Input layer is 2 dimensional because I have (x,y) information in data 
                #Output layer is 1 dimensional because I want to output the temperature
                #at that particular point 
                
                self.linear_tanh_stack = nn.Sequential(
                    nn.Linear(2,20),
                    nn.Tanh(),
                    nn.Linear(20,20),
                    nn.Tanh(),
                    nn.Linear(20,15),
                    nn.Tanh(),
                    nn.Linear(15,10),
                    nn.Tanh(),
                    nn.Linear(10,1)
                )

        def initialize_weights(self):
            print("About to initialize weights")
            # Print the layer names from the checkpoint
            
            checkpt = torch.load("coarse_result_for_fine_init.pth", map_location=lambda storage, loc: storage)
            print("Just set the losses from the coarse training")
            for key in checkpt["state_dict"].keys():
                print(key)
            # breakpoint()
            model.load_state_dict(checkpt["state_dict"])
            print("Just initialized weights")
            return model

        def forward(self, x):
                #x = self.flatten(x)
                torch.set_default_dtype(torch.float64)
                # print("x dtype is ",x.dtype)
                logits = self.linear_tanh_stack(x)
                return logits
    
    def u_true(x, y, problem):
        
        if problem == 'discontinuous':
            return (torch.tensor(x)-torch.tensor(y))*(torch.tensor(x)-torch.tensor(y))/2.0*(x<y)
        
        if problem == 'nonlinear': 
            return torch.exp(torch.tensor(x)+torch.tensor(y))
    
    def f_true(x, y, problem):
        if problem == 'discontinuous':
            return -2*(x<y)
        
        if problem == 'nonlinear':
            return torch.exp(3*(torch.tensor(x)+torch.tensor(y)))-2*torch.exp(torch.tensor(x)+torch.tensor(y))

    def g_true(x, y, problem):
        if problem == 'discontinuous':
            return (torch.tensor(x)-torch.tensor(y))*(torch.tensor(x)-torch.tensor(y))/2.0*(x<y)
        
        if problem == 'nonlinear': 
            return torch.exp(torch.tensor(x)+torch.tensor(y))

    #Load the neural network parameters learned during coarse training 
    network = NeuralNetwork().initialize_weights()
    

    print("About to show the parameters I am setting")
    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")
    # breakpoint()
    criterion = torch.nn.MSELoss() # Choose appropriate loss for your task
    # criterion = torch.optim.LBFGS(model.parameters(),max_iter=1000,line_search_fn='strong_wolfe')

    

    #Define a finer set of points to compute losses over 
    x_super_fine = torch.arange(-1,1.1,1/(3*num_interpolation_points))
    y_super_fine = torch.arange(-1,1.1,1/(3*num_interpolation_points))

    X_super_fine, Y_super_fine = torch.meshgrid(x_super_fine, y_super_fine)

    all_points_super_fine = torch.stack((X_super_fine.flatten(), Y_super_fine.flatten()), dim=-1)

    all_points_super_fine.requires_grad = True

    


    def generate_boundary(num_points):
        # Generate equally spaced points along each edge
        edges = [
            np.linspace([-1, -1], [1, -1], num_points // 4),  # bottom edge
            np.linspace([1, -1], [1, 1], num_points // 4),   # right edge
            np.linspace([1, 1], [-1, 1], num_points // 4),   # top edge
            np.linspace([-1, 1], [-1, -1], num_points // 4)   # left edge
        ]

        # Concatenate all edge points
        points = np.vstack(edges)

        return points
    #------------------------

    boundary_points = torch.tensor(generate_boundary(1600), requires_grad=True)

    
    interior_mask = ~((all_points_super_fine[:, 0] == 1) | (all_points_super_fine[:, 0] == -1) | 
                    (all_points_super_fine[:, 1] == 1) | (all_points_super_fine[:, 1] == -1))

    interior_points = all_points_super_fine[interior_mask]

    # Create the interior mask
    interior_mask_super_fine = ~((all_points_super_fine[:, 0] == 1) | 
                                (all_points_super_fine[:, 0] == -1) | 
                                (all_points_super_fine[:, 1] == 1) | 
                                (all_points_super_fine[:, 1] == -1))
    

    #Extract interior points and create a new tensor 
    boundary_points = torch.tensor(boundary_points.detach().numpy(), requires_grad=True)
    interior_points = torch.tensor(interior_points.detach().numpy(), requires_grad=True)
    interior_points_super_fine = torch.tensor(all_points_super_fine[interior_mask_super_fine].detach().numpy(), requires_grad=True)#all_points_super_fine[interior_mask_super_fine]
    boundary_points_super_fine = torch.tensor(boundary_points.detach().numpy(), requires_grad=True)


    boundary_points.requires_grad = True
    interior_points.requires_grad = True
    interior_points_super_fine.requires_grad=True

    #Store the data obtained from FEM 
    fem_data = u_fem_np


    losses = []
    boundary_losses = []
    interior_losses = []
    
    #This is what to use if I want to pick either real data or FEM data 
    a, b, c = (data_source==0, data_source==1, data_source==2)

    #This is what to use if I want to pick both real and FEM data 
    # p=0.5
    # sampling_true_data,sampling_fem_data = 1,0

    sampling_true_data = a
    sampling_fem_data = b
    sampling_no_data = c
    # print("sampling true data: ", sampling_true_data)
    # print("sampling fem data: ", sampling_fem_data)
    
    
    def closure():
        # print("Sampling true data: ", sampling_true_data)
        torch.set_default_dtype(torch.float64)

        #Store the points on which the FEM solution is computed 
        # Create a meshgrid
        coarse_x_vals = torch.linspace(-1, 1, num_interpolation_points)
        coarse_y_vals = torch.linspace(-1, 1, num_interpolation_points)
        x_vals = torch.linspace(-1, 1, 3*num_interpolation_points)
        y_vals = torch.linspace(-1, 1, 3*num_interpolation_points)
        coarse_points_x, coarse_points_y = torch.meshgrid(coarse_x_vals, coarse_y_vals, indexing='ij')
        fine_points_x, fine_points_y = torch.meshgrid(x_vals, y_vals, indexing='ij')


        # 2. Interpolate FEM data
        # You can use SciPy for interpolation
        from scipy.interpolate import griddata
        fine_points_x = fine_points_x.flatten().reshape(-1,1)
        fine_points_y = fine_points_y.flatten().reshape(-1,1)
        unpacked = torch.zeros_like(fine_points_x.flatten().reshape(-1,1))
        x_test = [fine_points_x[i].item() for i in range(len(unpacked))]
        y_test = [fine_points_y[i].item() for i in range(len(unpacked))]

        #Create a tensor that stores the PINN prediction at each point on the mesh
        #The FEM data fit is using a coarse FEM solution, interpolating the coarse FEM solution
        #and then fitting a PINN to the interpolated data
        # if sampling_fem_data:
        #     #Create a tensor that stores the FEM solution at each point on the mesh
        #     fem_interpolated = torch.zeros(fine_points_x.shape[0],3)
        #     for i in range(len(x_test)):
        #         fem_interpolated[i,:]=torch.tensor([x_test[i],y_test[i],u(x_test[i],y_test[i])])
        #     true_sol_interpolated = torch.zeros(fine_points_x.shape[0],3)
        #     pinn_interpolated = torch.zeros(fine_points_x.shape[0],3)
        #     model_on_mesh = model(torch.cat((fine_points_x,fine_points_y),dim=1))
        #     for i in range(len(x_test)):
        #         pinn_interpolated[i,:] = torch.tensor([fine_points_x[i],fine_points_y[i],model_on_mesh[i]]) 

        #     #Compare the PINN to the fem solution interpolated on those points
        #     data_fit_interior_loss_fem = criterion(fem_interpolated[:,2],pinn_interpolated[:,2])
            
            
        
        #Define coarse points where solution is known
        coarse_points_x = torch.tensor(coarse_points_x.flatten().reshape(-1,1), dtype=torch.float64)
        coarse_points_y = torch.tensor(coarse_points_y.flatten().reshape(-1,1), dtype=torch.float64)

        coarse_points = torch.hstack([coarse_points_x,coarse_points_y])
        
        interior_mask_coarse = ~((coarse_points[:, 0] == 1) | (coarse_points[:, 0] == -1) | 
                    (coarse_points[:, 1] == 1) | (coarse_points[:, 1] == -1))

        interior_points_coarse = coarse_points[interior_mask_coarse]
        interior_points_coarse_x = interior_points_coarse[:, 0].reshape(-1,1)
        interior_points_coarse_y = interior_points_coarse[:, 1].reshape(-1,1)
        # breakpoint()
        #FIGURE OUT IF THE GALS STUFF SHOULD BE INCORPORATED IN THE PHYSICS OR DATA FIT LOSS
        
        
        if (problem == 'nonlinear' or problem == 'discontinuous') and sampling_true_data:


            noise=np.random.normal(0,1,size=interior_points_coarse_x.shape)
            
            noise_tensor = torch.from_numpy(noise).to(torch.float64)
            true_sol_interpolated = torch.zeros(interior_points_coarse_x.shape[0],3)

            pinn_pred_on_data = model(torch.cat((interior_points_coarse_x,interior_points_coarse_y),dim=1))

            samples_f = f_true(interior_points_super_fine[:,0],interior_points_super_fine[:,1], problem)
            residual = interior_loss_func(interior_points_super_fine[:,0].reshape(-1,1),interior_points_super_fine[:,1].reshape(-1,1),samples_f,model,problem)
 
            samples_u = u_true(interior_points_coarse_x,interior_points_coarse[:,1].reshape(-1,1),problem)+noise_tensor
            
            coarse_data_fit = criterion(samples_u,pinn_pred_on_data)


        elif (problem == 'nonlinear' or problem == 'discontinuous'):
            samples_f = f_true(interior_points_super_fine[:,0],interior_points_super_fine[:,1], problem)
            residual = interior_loss_func(interior_points_super_fine[:,0].reshape(-1,1),interior_points_super_fine[:,1].reshape(-1,1),samples_f,model,problem)
            
            # print("coarse data fit loss: ", coarse_data_fit)

        if problem == 'adr':
            samples_f = None
            data_fit_interior_loss_true_sol = 0 
            residual = interior_loss_func(interior_points_super_fine[:,0].reshape(-1,1),interior_points_super_fine[:,1].reshape(-1,1),samples_f, model, problem)
            
        # print("residual: ", residual)

        #Compute boundary residual
        g_vals = g_true(boundary_points_super_fine[:,0].reshape(-1,1), boundary_points_super_fine[:,1].reshape(-1,1), problem)
        boundary_loss = dirichlet_boundary_loss_func(boundary_points_super_fine[:,0].reshape(-1,1),boundary_points_super_fine[:,1].reshape(-1,1), g_vals, model, problem)
        # breakpoint()

        if problem == 'discontinuous' and sampling_true_data: 
            # print("Coarse data fit loss: ", coarse_data_fit)
            # print("Boundary loss: ", boundary_loss)
            # print("Physics loss: ", residual)
            total_loss = physics_weight*residual + boundary_weight*boundary_loss + coarse_data_weight*coarse_data_fit #+ sampling_true_data*data_fit_interior_loss_true_sol 
        
        # if problem == 'discontinuous' and sampling_fem_data: 
        #     total_loss = physics_weight*residual + boundary_weight*boundary_loss + sampling_fem_data*data_fit_interior_loss_fem

        if problem == 'discontinuous' and sampling_no_data: 
            total_loss = physics_weight*residual + boundary_weight*boundary_loss  
        
        if problem == 'nonlinear' and sampling_true_data: 
            total_loss = physics_weight*residual + boundary_weight*boundary_loss + coarse_data_weight*coarse_data_fit#+ sampling_true_data*data_fit_interior_loss_true_sol

        if problem == 'adr':
            total_loss = physics_weight*residual + boundary_weight*boundary_loss

        #Store the interior and boundary losses to plot 
        if problem == ('discontinuous' or problem == 'nonlinear') and sampling_true_data:
            interior_losses.append(coarse_data_fit)
        boundary_losses.append(boundary_loss)
        losses.append(total_loss)
        total_loss = total_loss.to(torch.float64)
        # print(losses)
        # breakpoint()
        total_loss.backward()
        
        return total_loss

    #Training loop
    adam_lbfgs_cutoff = int(3*num_fine_epochs/4)
    tic = timeit.default_timer()
    for epoch in range(num_fine_epochs):
        #Try switching optimiziers during training
        if epoch < adam_lbfgs_cutoff:
            optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
        if epoch == adam_lbfgs_cutoff:
            print("Just switched to using LBFGS in the fine training, we're gonna find this minimum!")
        if epoch >= adam_lbfgs_cutoff:
            optimizer =  torch.optim.LBFGS(model.parameters(),max_iter=1000,line_search_fn='strong_wolfe')

        optimizer.zero_grad()
        # breakpoint()
        loss=optimizer.step(closure)
        # breakpoint()
        if epoch%10 == 0:
            print("Epoch ", epoch)
            print("Total loss: ", loss.item())

    toc = timeit.default_timer()

    #Redefine the fine mesh
    # Create a meshgrid
    x_vals = torch.linspace(-1, 1, 50)
    y_vals = torch.linspace(-1, 1, 50)
    fine_points_x, fine_points_y = torch.meshgrid(x_vals, y_vals, indexing='ij')

    #Evaluate the model on the fine mesh
    # breakpoint()
    fine_points_x = fine_points_x.to(torch.float64)
    fine_points_y = fine_points_y.to(torch.float64)
    model_on_mesh = model(torch.cat((fine_points_x.reshape(-1,1), fine_points_y.reshape(-1,1)), dim=1)).detach().numpy()

    from scipy.interpolate import griddata

    
    #Evaluate model on the points where we know the finite element solution
    fem_data_tensor = torch.tensor(fem_data)
    fem_x = fem_data_tensor[:, 0]
    fem_y = fem_data_tensor[:, 1]
    fem_z = fem_data_tensor[:, 2]


    # Interpolate FEM data onto the fine mesh
    fem_interpolated = griddata((fem_x, fem_y), fem_z, (fine_points_x, fine_points_y), method='linear')

    # 3. Reshape for plotting
    model_on_mesh_reshaped = model_on_mesh.reshape(50, 50)  # PINN trained on 
    fem_interpolated_reshaped = fem_interpolated.reshape(50, 50)

    # 4. Plot the results
    import matplotlib.pyplot as plt
    import matplotlib.gridspec as gridspec
    
    plt.figure(figsize=(12, 5))

    # Create a figure with a grid layout
    fig = plt.figure(figsize=(15, 5))
    gs = gridspec.GridSpec(1, 4, width_ratios=[1, 1, 1, 1], wspace=0.3)

    # # Plot FEM results
    # ax1 = fig.add_subplot(gs[0])
    # ax1.set_title("FEM Solution")
    # img1 = ax1.imshow(fem_interpolated_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    # ax1.set_xlabel('x')
    # ax1.set_ylabel('y')
    # cbar1 = plt.colorbar(img1, ax=ax1, fraction=0.046, pad=0.1)  # Adjust pad for colorbar spacing
    # cbar1.set_label('Z_{fem}')

    # # Plot Neural Network predictions
    # ax2 = fig.add_subplot(gs[1])
    # ax2.set_title("Neural Network Prediction")
    # img2 = ax2.imshow(model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    # ax2.set_xlabel('x')
    # ax2.set_ylabel('y')
    # cbar2 = plt.colorbar(img2, ax=ax2, fraction=0.046, pad=0.1)  # Adjust pad for colorbar spacing
    # cbar2.set_label('Z_{NN}')

    # # Difference between FEM and PINN
    # ax3 = fig.add_subplot(gs[2])
    # ax3.set_title("Difference between FEM and PINN")
    # img3 = ax3.imshow(fem_interpolated_reshaped - model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    # ax3.set_xlabel('x')
    # ax3.set_ylabel('y')
    # cbar3 = plt.colorbar(img3, ax=ax3, fraction=0.046, pad=0.1)  # Adjust pad for colorbar spacing
    # cbar3.set_label('Difference')

    # plt.tight_layout()  # Adjust layout
    # plt.show()


    # fem_data_x = fem_data_tensor[:, 0].unsqueeze(dim=1).to(torch.float32)
    # fem_data_y = fem_data_tensor[:, 1].unsqueeze(dim=1).to(torch.float32)

    if problem == 'nonlinear':

        #Plotting the FEM solution
        fig1, ax1 = plt.subplots(figsize=(6, 6))

        # Plot the FEM Solution
        cax1 = ax1.imshow(fem_interpolated_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')

        # Title, labels for first plot
        ax1.set_title("FEM Solution")
        ax1.set_xlabel('x')
        ax1.set_ylabel('y')

        # Add the colorbar to the first figure
        cbar1 = fig1.colorbar(cax1, ax=ax1, fraction=0.046, pad=0.04)
        cbar1.set_label('FEM')

        plt.savefig(f'FEM Solution_{problem}_data_source_{data_source} (1)',dpi=100)

        # Create the second figure (can be empty or contain other plots)
        plt.clf()
        fig2, ax2 = plt.subplots(figsize=(6, 6))

        # Plotting the NN prediction
        cax2 = ax2.imshow(model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')

        if sampling_fem_data and sampling_true_data:

            ax2.set_title("PINN on Noisy True Solution")
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            cbar2 = fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
            # cbar2.set_label('PINN Prediction', labelpad=15)
            plt.savefig(f'PINN prediction on real data_{problem}_data_source_{data_source}_ (2).png', dpi=100)
            plt.tight_layout()

        elif sampling_true_data:
            # plt.title("PINN on Noisy True Solution")
            ax2.set_title("PINN on Noisy True Solution")
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            cbar2 = fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
            # cbar2.set_label('PINN Prediction', labelpad=15)
            plt.savefig(f'PINN prediction on real data_{problem}_data_source_{data_source}__multilevel (2).png', dpi=100)
            plt.tight_layout()

            plt.clf()
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            cax3 = ax3.imshow(fem_interpolated_reshaped-model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
            ax3.set_title("FEM - PINN on Noisy True Solution")
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            cbar3 = fig3.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('PINN Prediction', labelpad=15)
            plt.savefig(f'FEM minus PINN trained on Data_{problem}_data_source_{data_source}_multilevel (3).png', dpi=100)
            plt.tight_layout()
        
        elif sampling_fem_data:
            # plt.title("PINN on FEM Data")
            ax2.set_title("PINN on FEM Solution")
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            cbar2 = fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('PINN Prediction', labelpad=15)
            plt.savefig(f'PINN prediction on FEM data_{problem}_data_source_{data_source} (2).png', dpi=100)
            plt.tight_layout()

            plt.clf()
            fig3, ax3 = plt.subplots(figsize=(6, 6))
            cax3 = ax3.imshow(fem_interpolated_reshaped-model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
            ax3.set_title("FEM - PINN on FEM Solution")
            ax3.set_xlabel('x')
            ax3.set_ylabel('y')
            cbar3 = fig3.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
            cbar3.set_label('PINN Prediction', labelpad=15)
            plt.savefig(f'FEM minus PINN trained on FEM_{problem}_data_source_{data_source} (3).png')
            plt.tight_layout()

            #Plot differernce between FEM and True solution
            x_true = np.linspace(-1,1,fem_interpolated_reshaped.shape[0])
            y_true = np.linspace(-1,1,fem_interpolated_reshaped.shape[0])
            X_true, Y_true = np.meshgrid(x_true, y_true)
            true_sol = u_true(X_true, Y_true, problem)
            # breakpoint()
            plt.clf()
            fig4, ax4 = plt.subplots(figsize=(6, 6)) 
            cax4 = ax4.imshow(fem_interpolated_reshaped-np.asarray(true_sol), extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
            ax4.set_title("FEM - True Solution")
            # ax4.set_xlabel('x')
            # ax4.set_y_label('y')
            cbar4 = fig4.colorbar(cax4, ax=ax4, fraction=0.046, pad=0.04)
            cbar4.set_label('FILL IN', labelpad=15)
            plt.savefig(f'FEM minus True Solution_{problem}')
            
            
            #Plot difference between FEM-PINN and True solution 
            # fem_x and fem_y
            # breakpoint()
            plt.clf()
            fig5, ax5 = plt.subplots(figsize=(6, 6))
            true_sol = u_true(fine_points_x, fine_points_y, problem)
            # breakpoint()
            cax5 = ax5.imshow(model_on_mesh_reshaped-np.asarray(true_sol), extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
            ax5.set_title("FEM PINN - True Solution")
            cbar5 = fig5.colorbar(cax5, ax=ax5, fraction=0.046, pad=0.04)
            cbar5.set_label('FILL IN', labelpad=15)
            plt.savefig(f'FEM PINN minus True Solution_{problem}')
            
            # fig5, ax5 = plt.subplot(figsize=(6,6))
            # cax5 = ax5.imshow(fe)

        else:
            ax2.set_title("PINN Without Data")
            ax2.set_xlabel('x')
            ax2.set_ylabel('y')
            cbar2 = fig2.colorbar(cax2, ax=ax2, fraction=0.046, pad=0.04)
            cbar2.set_label('PINN Prediction', labelpad=15)
            plt.savefig(f'PINN Prediction Without Data (2).png', dpi=100)
            plt.tight_layout()

        #Plotting FEM minus PINN 
        # Create the second figure (can be empty or contain other plots)
        
        # if sampling_true_data:
        #     plt.clf()
        #     fig3, ax3 = plt.subplots(figsize=(6, 6))
        #     cax3 = ax3.imshow(fem_interpolated_reshaped-model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        #     ax3.set_title("FEM - PINN on Noisy True Solution")
        #     ax3.set_xlabel('x')
        #     ax3.set_ylabel('y')
        #     cbar3 = fig3.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
        #     cbar3.set_label('PINN Prediction', labelpad=15)
        #     plt.savefig(f'FEM minus PINN trained on Data_{problem}_data_source_{data_source} (3).png', dpi=100)
        #     plt.tight_layout()
        
        # if sampling_fem_data:
        #     plt.clf()
        #     fig3, ax3 = plt.subplots(figsize=(6, 6))
        #     cax3 = ax3.imshow(fem_interpolated_reshaped-model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        #     ax3.set_title("FEM - PINN on FEM Solution")
        #     ax3.set_xlabel('x')
        #     ax3.set_ylabel('y')
        #     cbar3 = fig3.colorbar(cax3, ax=ax3, fraction=0.046, pad=0.04)
        #     cbar3.set_label('PINN Prediction', labelpad=15)
        #     plt.savefig(f'FEM minus PINN trained on FEM_{problem}_data_source_{data_source} (3).png')
        #     plt.tight_layout()

        plt.clf()
        true_solution = u_true(fine_points_x,fine_points_y,problem).reshape(50,50)
        fig4, ax4 = plt.subplots(figsize=(6, 6))
        cax4 = ax4.imshow(true_solution, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        
        ax4.set_title("True Solution")
        ax4.set_xlabel('x')
        ax4.set_ylabel('y')

        # Add the colorbar to the figure
        cbar4 = fig4.colorbar(cax4, ax=ax4, fraction=0.046, pad=0.04)
        cbar4.set_label('True Solution')

        plt.savefig(f'TrueSol_discont_{problem} (4)', dpi=100)
        plt.tight_layout()

        plt.clf()
        fig5, ax5 = plt.subplots(figsize=(6, 6))
        cax5 = ax5.imshow(fem_interpolated_reshaped-np.asarray(true_solution), extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        ax5.set_title("FEM - True Solution")
        ax5.set_xlabel('x')
        ax5.set_ylabel('y')

        # Add the colorbar to the figure
        cbar5 = fig5.colorbar(cax5, ax=ax5, fraction=0.046, pad=0.04)
        cbar5.set_label('FEM - True Solution')

        plt.savefig(f"FEM_True_diff_{problem}_multilevel (5)", dpi=100)
        plt.tight_layout()

        plt.clf()
        fig6, ax6 = plt.subplots(figsize=(6,6))
        # f = plt.figure()
        # f.set_figwidth(5)
        # f.set_figheight(5)
        cax6 = ax6.imshow(np.array(model_on_mesh_reshaped)-np.array(true_solution), extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        ax6.set_title("PINN - True Solution")
        ax6.set_xlabel('x')
        ax6.set_ylabel('y')

        cbar6 = fig6.colorbar(cax6, ax=ax6, fraction=0.046, pad=0.04)
        cbar6.set_label('PINN - True Solution')
        label = cbar6.ax.yaxis.label
        # label.set_position((3,3))
        plt.subplots_adjust(left=None, right=None, bottom=None, top=None, wspace=None, hspace=None) 
        plt.tight_layout()
        plt.savefig(f"PINN_True_diff_{problem}_data_source_{data_source}_multilevel (6)", dpi=100)

        #Plot boundary and interior losses 
        # breakpoint()
        boundary_losses = [boundary_loss.detach().item() for boundary_loss in boundary_losses]
        # breakpoint()
        # interior_losses = [interior_loss.detach().item() for interior_loss in interior_losses]

        plt.clf()
        plt.semilogy(np.arange(1,len(boundary_losses)+1), boundary_losses)
        plt.title("Boundary Losses")
        plt.xlabel("Epoch")
        plt.ylabel("Boundary Loss")
        plt.tight_layout()
        plt.savefig(f"boundary_losses_{problem}_data_source_{data_source} (7)")

        # plt.clf()
        # plt.semilogy(np.arange(1,len(interior_losses)+1), interior_losses)
        # plt.title("Interior Losses")
        # plt.xlabel("Epoch")
        # plt.ylabel("Interior Loss")
        # plt.tight_layout()
        # plt.savefig(f"interior_losses_{problem}_data_source_{data_source} (8)")

        # plt.subplot(1,6,4)
        # plt.title("True Solution")
        # plt.imshow(true_solution, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        # cbar = plt.colorbar(label='True Solution', fraction=0.08, pad=0.2)
        # plt.xlabel('x')
        # plt.ylabel('y')

        

        # Show both figures

        # # Plot FEM results
        # plt.subplot(1, 6, 1)
        # plt.title("FEM Solution")
        # plt.imshow(fem_interpolated_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        # # cbar = plt.colorbar(label='FEM', fraction=0.08, pad=0.2)
        # # cbar.ax.set_aspect(10)
        # plt.xlabel('x')
        # plt.ylabel('y')

        # # plt.subplot(1,7,2)
        # # cbar = plt.colorbar(label='FEM', fraction=0.08, pad=0.2)

        # # Plot Neural Network predictions

        # if sampling_true_data: 
        #     plt.subplot(1, 6, 2) 
        #     plt.title("PINN on Real Data")
        #     plt.imshow(model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        #     cbar = plt.colorbar(label='NN', fraction=0.08, pad=0.2)
        #     # cbar.ax.set_aspect(10)
        #     plt.xlabel('x')
        #     plt.ylabel('y')

        # if sampling_fem_data:
        #     plt.subplot(1, 6, 2)
        #     plt.title("PINN on FEM Data")
        #     plt.imshow(model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        #     cbar = plt.colorbar(label='NN', fraction=0.08, pad=0.2)
        #     # cbar.ax.set_aspect(10)
        #     plt.xlabel('x')
        #     plt.ylabel('y')


        # #Difference between FEM and PINN
        # plt.subplot(1, 6, 3)
        # plt.title("FEM - PINN")
        # plt.imshow(fem_interpolated_reshaped-model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        # cbar = plt.colorbar(label='FEM-NN', fraction=0.08, pad=0.2)
        # plt.xlabel('x')
        # plt.ylabel('y')

        # true_solution = u_true(fine_points_x,fine_points_y,problem).reshape(50,50)

        # plt.subplot(1,6,4)
        # plt.title("True Solution")
        # plt.imshow(true_solution, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        # cbar = plt.colorbar(label='True Solution', fraction=0.08, pad=0.2)
        # plt.xlabel('x')
        # plt.ylabel('y')

        # plt.subplot(1,6,5)
        # plt.title("FEM - True Solution")
        # plt.imshow(np.array(fem_interpolated_reshaped)-np.array(true_solution), extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        # cbar = plt.colorbar(label='FEM - True Solution', fraction=0.08, pad=0.2)
        # plt.xlabel('x')
        # plt.ylabel('y')

        # plt.subplot(1,6,6)
        # plt.title("PINN-True Solution ")
        # plt.imshow(np.array(model_on_mesh_reshaped)-np.array(true_solution), extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
        # cbar = plt.colorbar(label='PINN - True Solution', fraction=0.08, pad=0.2)
        # plt.xlabel('x')
        # plt.ylabel('y')

        # plt.subplots_adjust(wspace=1.5) 

    plt.clf()
    if problem == 'nonlinear': 

        # Plot Neural Network predictions

        if sampling_true_data: 
            
            plt.subplot(1, 3, 1) 
            plt.title("PINN on Real Data")
            plt.imshow(model_on_mesh_reshaped, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
            cbar = plt.colorbar(label='NN', shrink=0.8, pad=0.3)
            # cbar.ax.set_aspect(10)
            plt.xlabel('x')
            plt.ylabel('y')

            true_solution = u_true(fine_points_x,fine_points_y,problem).reshape(50,50)

            plt.subplot(1, 3, 2)
            plt.title("True Solution")
            plt.imshow(true_solution, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
            cbar = plt.colorbar(label='True Solution', shrink=0.8, pad=0.3)
            plt.xlabel('x')
            plt.ylabel('y')

            plt.subplot(1, 3, 3) 
            plt.title("PINN-True Solution")
            plt.imshow(model_on_mesh_reshaped-np.asarray(true_solution), extent=(-1,1,-1,1), origin='lower', cmap='viridis')
            cbar = plt.colorbar(label='PINN-True Solution', shrink=0.8, pad=0.3)
            plt.xlabel('x')
            plt.ylabel('y')

            plt.savefig(f"nonlinear_PINN_trained_on_real_data_results_{num_interpolation_points}_noisy_data_points.png")

    if problem == 'adr':
        plt.imshow(model_on_mesh_reshaped, extent=(-1,1,-1,1), origin='lower', cmap='viridis')
        cbar = plt.colorbar(label="PINN", shrink=0.8, pad=0.3)
        plt.title('Advection Diffusion Reaction PINN Solution')
        plt.xlabel('x')
        plt.ylabel('y')

        plt.savefig(f"adr_PINN.png")

    plt.tight_layout() 

    # if sampling_true_data:
    #     if problem == 'discontinuous':
    #         plt.savefig(f"discont_PINN_trained_on_real_data_results_{num_interpolation_points}_noisy_data_points.png")
    #     if problem == 'nonlinear': 
    #         plt.savefig(f"nonlinear_PINN_trained_on_real_data_results_{num_interpolation_points}_noisy_data_points.png")
    # if sampling_fem_data: 
    #     if problem == 'discontinuous':
    #         plt.savefig(f"discont_PINN_trained_on_FEM_data_results_{num_interpolation_points}_fem_data_points.png")
    #     if problem == 'nonlinear': 
    #         plt.savefig(f"nonlinear_PINN_trained_on_FEM_data_results_{num_interpolation_points}_fem_data_points.png")
    # if not sampling_true_data and not sampling_fem_data:
    #     if problem == 'discontinuous': 
    #         plt.savefig(f"discontinuous_PINN_no_data.png")
    





    #Plot the difference between exact solution and FEM 

    #Plot the difference between exact solution and PINN informed by FEM 


    # Calculate the difference and relative error
    difference = fem_interpolated_reshaped - model_on_mesh_reshaped
    relative_error = np.abs(difference) / (np.abs(fem_interpolated_reshaped) + 1e-10)  # Avoid division by zero

    # Relative error plot
    # plt.subplot(1,4,4)
    # plt.title("Relative Error")
    # plt.imshow(relative_error, extent=(-1, 1, -1, 1), origin='lower', cmap='viridis')
    # cbar = plt.colorbar(label='Relative error', shrink=0.8, pad=0.1)
    # plt.xlabel('x')
    # plt.ylabel('y')
    

    evaluation_points = torch.zeros(7,2)

    aux1, aux2 = 1/np.sqrt(3), np.sqrt(3/5)
    aux3 = np.sqrt(14/15)


    #Define evaluation points
    evaluation_points[0] = torch.tensor([0, 0])
    evaluation_points[1] = torch.tensor([aux1, aux2])
    evaluation_points[2] = torch.tensor([-aux1, aux2])
    evaluation_points[3] = torch.tensor([aux1, -aux2])
    evaluation_points[4] = torch.tensor([-aux1, -aux2])
    evaluation_points[5] = torch.tensor([aux3, 0])
    evaluation_points[6] = torch.tensor([-aux3, 0])

    #Call the model and the true solution and compute |true - predicted| and integrate that
    true_model_diff = []
    for i in range(evaluation_points.shape[0]):

        u_val = u_true(evaluation_points[i][0],evaluation_points[i][1],problem)

        model_fine_val = model(evaluation_points[i].to(torch.float64))
        # breakpoint()
        if u_val is not None: 
            true_model_diff.append((u_val-model_fine_val)**2)

            true_model_diff = [difference for difference in true_model_diff]

    #Define the weights
    weights = np.array([8/7, 20/36, 20/36, 20/36, 20/36, 20/63, 20/63])

    if problem == 'nonlinear' or problem == 'discontinuous':
        residual = 0
        for i in range(evaluation_points.shape[0]):
            residual = residual + weights[i]*true_model_diff[i]
        # breakpoint()
        print("Integrated error is ",np.sqrt(residual.detach()))

    #Add code to write to an output file 

    with open("output.txt",'a') as f:

        if problem == 'nonlinear' or problem == 'discontinuous':
            f.write("\n"+"2nd Order Nonlinear Coarse Initialization Fine Training: "+str(toc-tic)+str(" seconds to train"))
            f.write("\n"+"Sampling true data "+str(sampling_true_data))
            f.write("\n"+"Sampling fem data: "+str(sampling_fem_data))
            # f.write("\n"+"max error: "+ str(max_error))
            # f.write("\n"+"mean error: "+ str(mean_error))
            f.write("\n"+str("L^2 error ")+str(np.sqrt(residual.detach()))+"\n")

            return np.sqrt(residual.detach())

    # breakpoint()

noisy_data_residuals = []
fem_data_residuals = []

#Problems implemented 
# 'discontinuous' (Has FEM Data, Has true data)
# 'nonlinear' (No FEM Data, Has true data)
# 'adr' (No FEM Data, No true data)

problem = 'discontinuous'

#True data: data_source = 0
#FEM data: data_source = 1
#No data: data_source = 2
#num_interpolation_points gets multiplied by 5 inside of fine training. 
# run_multilevel_fine(problem, num_interpolation_points=20, data_source=0, physics_weight = 20, boundary_weight = 100)

#Automated plotting (run this once I get the plots all looking nice)
# for sampling_true_data in range(2):
#     print("sampling true data ", sampling_true_data)
#     for num_interpolation_points in range(5,50,5):
        
#         if sampling_true_data == 0: 
#             print("Sampling fem data")
#             residual = run(problem, sampling_true_data, num_interpolation_points)
#             fem_data_residuals.append(residual)

#         if sampling_true_data == 1:
#             print("Sampling true data")
#             residual = run(problem, sampling_true_data, num_interpolation_points)
#             noisy_data_residuals.append(residual)
        
        

# breakpoint()
    # breakpoint()

    #Error computations

    # max_error = max(np.abs(fem_interpolated_reshaped-model_on_mesh_reshaped))
    # mean_error = np.mean(np.abs(fem_interpolated_reshaped-model_on_mesh_reshaped))

    # print("Max error is ", max_error)
    # print("Mean error is ", mean_error)


    #Analyze the Hessian matrix 

    # # After training or at checkpoints, compute the Hessian
    # def compute_hessian(x, y, samples_f,model):
    #     # Forward pass
    #     outputs = model(inputs)
    #     loss = interior_loss_func(x, y, samples_f, model)

    #     # Compute first derivatives
    #     grads = torch.autograd.grad(loss, model.parameters(), create_graph=True)

    #     hessian = []
    #     for g in grads:
    #         row = []
    #         for i in range(len(g)):
    #             # Compute second derivatives
    #             second_derivative = torch.autograd.grad(g[i], model.parameters(), retain_graph=True)
    #             row.append(second_derivative)
    #         hessian.append(row)

    #     return hessian

    # # Example usage
    # inputs = fem_data[:,:2]
    # targets = fem_data[:,2]
    # samples_f = f_true(inputs[:,0],inputs[:,1])
    # hessian = compute_hessian(x, y, samples_f, model)

