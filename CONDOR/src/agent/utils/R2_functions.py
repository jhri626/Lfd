import torch
import math
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.colors import ListedColormap

def Uniform_sampling_R2(xtraj, batch_size, xlim=None, ylim=None):
    
    if xtraj.get_device() == -1:
        Ttraj_torch = xtraj
        xtraj = xtraj.detach().numpy()
    else:
        Ttraj_torch = xtraj
        xtraj = xtraj.cpu().detach().numpy()

    if xlim is None:
        xmin = np.min(xtraj, axis=0)
        xmax = np.max(xtraj, axis=0)
        xlength = np.linalg.norm(xmax - xmin)
        min_offset = xlength / 4
        max_offset = xlength / 4

        random_x1 = torch.tensor(np.random.uniform(low=np.min(xtraj[:,0]) - min_offset,
                                                high=np.max(xtraj[:,0]) + max_offset,
                                                size=[batch_size, 1]))
        random_x2 = torch.tensor(np.random.uniform(low=np.min(xtraj[:,1]) - min_offset,
                                                high=np.max(xtraj[:,1]) + max_offset,
                                                size=[batch_size, 1]))
    else:
        random_x1 = torch.tensor(np.random.uniform(low=xlim[0], high=xlim[1], size=[batch_size, 1]))
        random_x2 = torch.tensor(np.random.uniform(low=ylim[0], high=ylim[1], size=[batch_size, 1]))

    random_x = torch.cat([random_x1, random_x2], dim=1)
    
    return random_x


def gvf_R2(xsample, eta, xtraj, xdottraj):
    """
    Compute vector field for given sample points
    
    Args:
        xsample: (B, D) sample points
        eta: scaling parameter
        xtraj: (B, T, D) trajectory data
        xdottraj: (B, T, D) velocity data
        
    Returns:
        (B, D) vector at each sample point
    """
    B = xsample.shape[0]  # batch size
    T = xtraj.shape[1]    # sequence length
    D = xtraj.shape[2]    # feature dimension
    
    # Expand (B, 1, D) to (B, T, D)
    xsample_expanded = xsample.unsqueeze(1)
    # Compute pairwise distance (B, T)
    distance = torch.norm(xsample_expanded - xtraj, dim=2)
    
    # Find index of the closest point (B,)
    index_closest = torch.argmin(distance, dim=1)
    
    # Create batch indices
    batch_idx = torch.arange(B, device=xtraj.device)
    
    # Retrieve closest point and velocity (B, D)
    closest_point = xtraj[batch_idx, index_closest]
    V1 = xdottraj[batch_idx, index_closest]
    
    # Compute velocity direction
    vel = closest_point - xsample
    if eta == torch.inf:
        return vel
    else:
        return V1 + eta * vel

    

def streamline_plot_R2(xtraj, xdottraj=None, eta=5, grid_step = 101, a_max=50, ax=None, figsize=(10,10), self=None):
    # set
    eps = 1e-3
    aut = cm.get_cmap('autumn', 128 * 2.5)
    new_aut  = ListedColormap(aut(range(256)))
    
    # make figure
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot()
    
    # make grid
    xmin, _ = torch.min(xtraj, axis=0)
    xmax, _ = torch.max(xtraj, axis=0)
    xlength = torch.norm(xmax - xmin)
    offset = xlength / 4
    x1_linspace = torch.linspace(xmin[0]-offset, xmax[0]+offset, grid_step)
    x2_linspace = torch.linspace(xmin[1]-offset, xmax[1]+offset, grid_step)
    x1, x2 = torch.meshgrid(x1_linspace, x2_linspace)    
    xmesh = torch.cat([x1.unsqueeze(-1), x2.unsqueeze(-1)], dim=-1)
    xmesh_long = xmesh.reshape(-1, 2).to(xtraj)
    
    # get vector field
    if self is not None:
        xtraj_batch = xtraj.unsqueeze(0).repeat(len(xmesh_long),1,1)
        xdot_long = self.forward(xmesh_long.to('cuda:0'), xtraj_batch.to('cuda:0'), eta=eta) ### 'cuda:0' 수정하기
    else:
        xdot_long = gvf_R2(xmesh_long, eta, xtraj, xdottraj)
    xdot_grid = xdot_long.cpu().reshape(grid_step, grid_step, 2)
    color_grid = np.clip(xdot_grid.norm(dim=-1).detach().numpy(), a_max=a_max, a_min=0)
    
    if torch.is_tensor(xtraj):
        xtraj = xtraj.detach().cpu().numpy()
    plt.plot(xtraj[0, 0], xtraj[0, 1], 'cs', markersize=15, zorder=3)
    plt.plot(xtraj[-1, 0], xtraj[-1, 1], 'bo', markersize=15, zorder=3)
    plt.plot(xtraj[:, 0], xtraj[:, 1], 'g', zorder=2)
    
    if self is not None:
        res = plt.streamplot(x1.T.numpy(), x2.T.numpy(), xdot_grid[:, :, 0].T.detach().numpy(), xdot_grid[:, :, 1].T.detach().numpy(),
                        density=3, color=color_grid, linewidth=1, cmap=new_aut, zorder=1) #cmap=color_map)
    
    else:
        res = plt.streamplot(x1.T.numpy(), x2.T.numpy(), xdot_grid[:, :, 0].T.numpy(), xdot_grid[:, :, 1].T.numpy(),
                            density=3, color=color_grid, linewidth=1, cmap=new_aut, zorder=1) #cmap=color_map)
    plt.colorbar(res.lines)
    
    return ax


def xtraj_to_xdottraj(xtraj, dt):
    xdottraj = torch.zeros_like(xtraj)
    xdottraj[:-1] = xtraj[1:] - xtraj[:-1]
    return xdottraj/dt


def R2_interpolation(xtraj, step, dt):
    length = len(xtraj)
    if dt is not None:
        dt_inter = dt * (length-1)/(step-1)
    else:
        dt_inter = None
    
    xtraj_inter = []
    for i in range(step):
        t = i*(length-1)/(step-1)
        start = int(t)
        end = math.ceil(t)
        s = t % 1
        
        xs = xtraj[start]
        xe = xtraj[end]
        x = xs + s*(xe-xs)
        
        xtraj_inter.append(x)
    
    return np.array(xtraj_inter), dt_inter


# def LASA_traj

# def Gaussian_sampling_LASA(xtraj, std, batch_size=1):
#     eps = 1e-10
#     if std == 0:
#         std = eps
    
    
#     distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * p_std)
#     Gaussian_x = distribution.sample((batch_size,)).unsqueeze(-1).to(xtraj)
#     p_samples = traj_samples[:, :3, 3:4] + Gaussian_x
#     random_T = torch.cat([torch.cat([R_samples, p_samples], dim=2), torch.zeros(batch_size, 1, 4, device=qtraj.device)],
#                          dim=1).detach()

#     return random_T



# def LASA_deeponet_traj_gen(start_SE3, Ttraj, model=None, eta_R=0.1, eta_p=0.1, time_step=1, num_update=400, self=None):
    
#     SE3_traj = torch.zeros([num_update,4,4])
#     if self is not None:
#         self  = self.to(start_SE3)
    
#     for i in range(num_update):
#         if i == 0:
#             T_old = start_SE3
#         else:
#             T_old = T_new.to(start_SE3)
#         if len(T_old.shape) == 2:
#             T_old = T_old.unsqueeze(0).to(start_SE3)
        
#         T_flat = h(T_old)
#         SE3_traj[i] = T_old
        
#         # model forward
#         if model is not None:
#             vf_T = model.forward(T_old, Ttraj.to(start_SE3), eta_R, eta_p)
#         elif self is not None:
#             vf_T = self.forward(T_old, Ttraj.to(start_SE3), eta_R, eta_p)
        
#         #update
#         R = T_old[:,:3,:3]
#         dR_dtheta = Dexp_so3(skew(log_SO3(R)))

#         Rdot = torch.einsum('nijk,ni->njk', dR_dtheta, vf_T[:,:3].to(dR_dtheta))
#         Rt = R.transpose(1, 2)
#         R = R @ exp_so3(Rt @ Rdot*time_step)
#         p = vf_T[:,3:].to(T_old)*time_step + T_old[:,:3,3]
        
#         T_new = torch.eye(4,4)
#         T_new[:3,:3] = R.squeeze()
#         T_new[:3,3] = p
    
#     return SE3_traj