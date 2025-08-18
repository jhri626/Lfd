from tools.animation import TrajectoryPlotter
import torch
import matplotlib.pyplot as plt
import importlib
from initializer import initialize_framework
import numpy as np

# Parameters
# title = 'Dynamic system with latent dynamics /BCSDM with $\eta=0.2$'
title = 'Dynamic system with CONDOR framework latent dim 256'
eta=256
params_name = '2nd_order_2D'
x_t_init = np.array([[0.5, 0.6], [-0.75, 0.9], [0.9, -0.9], [-0.9, -0.9], [0.9, 0.9], [0.9, 0.3], [-0.9, -0.1],
                     [-0.9, 0.0], [0.4, 0.4], [0.9, -0.1], [-0.9, -0.5], [0.9, -0.5], [-0.25, 0.5],[-0.5, -0.5]])  # initial states
zeros_to_add = np.zeros((x_t_init.shape[0], 2))  # shape = (12, 2)
x_t_init = np.hstack((x_t_init, zeros_to_add)) 
simulation_length = 2000
results_base_directory = './'

# Load parameters
Params = getattr(importlib.import_module('params.' + params_name), 'Params')
params = Params(results_base_directory)
params.results_path += params.selected_primitives_ids + '/'
params.load_model = True

# Initialize framework
learner, _, data = initialize_framework(params, params_name, verbose=False)

# Initialize dynamical system



fontdict = {
    'fontsize': 16,
    'fontweight': 'normal'  
}

demonstrations_eval = data['demonstrations train'][...,0]
primitive_ids = np.array(data['demonstrations primitive id'])
x_min = np.array(data['x min'])
x_max = np.array(data['x max'])

x_t_init

dynamical_system = learner.init_dynamical_system(initial_states=torch.FloatTensor(x_t_init).cuda(),delta_t=params.delta_t)

# Initialize trajectory plotter
fig, ax = plt.subplots()
fig.set_size_inches(8, 8)    
fig.show()
trajectory_plotter = TrajectoryPlotter(fig, fontdict=fontdict, x0=x_t_init.T, pause_time=1e-5, goal=data['goals training'][0], title=title)

ax_bg = trajectory_plotter._ax  # background trajectories are drawn on the same Axes

# plot each demonstration as a continuous gray line


# force full redraw so that background lines appear immediately
trajectory_plotter._fig.canvas.draw()
trajectory_plotter._fig.canvas.flush_events()
# Simulate dynamical system and plot
for i in range(simulation_length):
    # Do transition
    x_t = dynamical_system.transition(space='task')['desired state']

    # Update plot
    trajectory_plotter.update(x_t.T.cpu().detach().numpy())

for demo in demonstrations_eval:
    # demo has shape (T_steps, 2)
    x_vals = demo[:, 0]       # x-coordinate time series
    y_vals = demo[:, 1]       # y-coordinate time series
    ax_bg.plot(
        x_vals,
        y_vals,
        color='lightgray',
        alpha=0.5,
        linewidth=6,
        zorder=0
    )
fig.savefig(results_base_directory + f'final_trajectory_ dim :{eta}.png', dpi=300)
plt.close(fig)