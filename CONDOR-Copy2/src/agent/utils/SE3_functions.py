import numpy as np
import torch
import matplotlib.pyplot as plt
# from agent.utils.Lie_old import *
# from agent.utils.Lie import *
from utils.Lie_old import *
from utils.Lie import *
import scipy
import math
# import agent.utils.LieGroup_torch as LieGroup
import utils.LieGroup_torch as LieGroup

import time


def SE3_interpolation(Ttraj, step, dt=None):
    length = len(Ttraj)
    if dt is not None:
        dt_inter = dt * (length-1)/(step-1)
    else:
        dt_inter = None
    
    Ttraj_inter = []
    for i in range(step):
        t = i*(length-1)/(step-1)
        start = int(t)
        end = math.ceil(t)
        s = t % 1
        
        Ts = Ttraj[start]
        Te = Ttraj[end]
        R = Ts[:3,:3] @ exp_so3(s*log_SO3((Ts[:3,:3].T @ Te[:3,:3]).unsqueeze(0)))
        p = Ts[:3,3] + s*(Te[:3,3] - Ts[:3,3])
        
        T = np.eye(4)
        T[:3,:3] = R
        T[:3,3] = p
        Ttraj_inter.append(T)
    
    return np.array(Ttraj_inter), dt_inter

def proj_minus_one_plus_one(x):
    eps = 1e-6
    x = torch.min(x, (1 - eps) * (torch.ones(x.shape).to(x)))
    x = torch.max(x, (-1 + eps) * (torch.ones(x.shape).to(x)))
    return x

def SO3_uniform_sampling(batch_size):
    SO3_sampler = scipy.stats.special_ortho_group(dim=3)
    R = SO3_sampler.rvs(batch_size)
    return R

def log_SO3(R):
    eps = 1e-4
    
    trace = torch.sum(R[:, range(3), range(3)], dim=1).to(R)
    omega = torch.zeros(R.shape).to(R)
    theta = torch.acos(torch.clip((trace - 1) / 2, -1, 1)).to(R)
    temp = theta.unsqueeze(-1).unsqueeze(-1).to(R)

    omega[(torch.abs(trace + 1) > eps) * (theta > eps)] = ((temp / (2 * torch.sin(temp))) * (R - R.transpose(1, 2)))[
        (torch.abs(trace + 1) > eps) * (theta > eps)]

    omega_temp = (R[torch.abs(trace + 1) <= eps] - torch.eye(3).to(R)) / 2

    omega_vector_temp = torch.sqrt((omega_temp[:, range(3), range(3)] + torch.ones(3).to(R)).clip(min=0))
    
    A = omega_vector_temp[:, 1] * torch.sign(omega_temp[:, 0, 1])
    B = omega_vector_temp[:, 2] * torch.sign(omega_temp[:, 0, 2])
    C = omega_vector_temp[:, 0]
    omega_vector = torch.cat([C.unsqueeze(1), A.unsqueeze(1), B.unsqueeze(1)], dim=1)
    omega[torch.abs(trace + 1) <= eps] = skew(omega_vector) * math.pi

    return omega


def log_SO3_T(T):
    # dim T = n,4,4
    R = T[:, 0:3, 0:3]  # dim n,3,3
    p = T[:, 0:3, 3].unsqueeze(-1)  # dim n,3,1
    n = T.shape[0]
    W = log_SO3(R.to(T))  # n,3,3

    return torch.cat([torch.cat([W, p], dim=2), torch.zeros(n, 1, 4).to(T)], dim=1)  # n,4,4


def log_SE3(T):
    #dim T = n,4,4
    R = T[:,0:3,0:3] # dim n,3,3
    p = T[:,0:3,3].unsqueeze(-1) # dim n,3,1
    n = T.shape[0]
    W = log_SO3(R) #n,3,3
    #print(W)
    w = skew(W) #n,3
    
    wsqr = torch.tensordot(w,w, dims=([1],[1]))[[range(n),range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1) # dim = (n,1)
    wnorm = torch.sqrt(wsqr) # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed) # dim - (n,1)
    wnorm_inv = 1/wnorm_unsqueezed # dim = (n)
    cw = torch.cos(wnorm).view(-1,1).unsqueeze(-1) # (dim = n,1)
    sw = torch.sin(wnorm).view(-1,1).unsqueeze(-1) # (dim = n,1)
    
    P = torch.eye(3).to(T) + (1-cw)*(wnorm_inv**2)*W + (wnorm_unsqueezed - sw) * (wnorm_inv**3) * torch.matmul(W,W) #n,3,3
    v = torch.inverse(P)@p #n,3,1
    return torch.cat([torch.cat([W,v],dim=2),torch.zeros(n,1,4).to(T)],dim=1)

def Uniform_sampling_SE3(Ttraj, batch_size, position_limit=None):
    
    if Ttraj.get_device() == -1:
        Ttraj_torch = Ttraj
        Ttraj = Ttraj.detach().numpy()
    else:
        Ttraj_torch = Ttraj
        Ttraj = Ttraj.cpu().detach().numpy()
        
    SO3_sampler = scipy.stats.special_ortho_group(dim=3)
    random_R = torch.tensor(SO3_sampler.rvs(batch_size), dtype=dtype)
    
    if isinstance(log_SO3(random_R), str):
        random_q = Uniform_sampling_SE3(Ttraj_torch, batch_size)
        return random_q

    if position_limit is None:
        qmin = np.min(Ttraj[:, 0:3,3], axis=0)
        qmax = np.max(Ttraj[:, 0:3,3], axis=0)
        qlength = np.linalg.norm(qmax - qmin)
        min_offset = qlength / 1
        max_offset = qlength / 1

        random_p1 = torch.tensor(np.random.uniform(low=np.min(Ttraj[:,0,3]) - min_offset,
                                                   high=np.max(Ttraj[:,0,3]) + max_offset,
                                                   size=[batch_size, 1]), dtype=dtype)
        random_p2 = torch.tensor(np.random.uniform(low=np.min(Ttraj[:,1,3]) - min_offset,
                                                high=np.max(Ttraj[:,1,3]) + max_offset,
                                                size=[batch_size, 1]), dtype=dtype)
        random_p3 = torch.tensor(np.random.uniform(low=np.min(Ttraj[:,2,3]) - min_offset,
                                                high=np.max(Ttraj[:,2,3]) + max_offset,
                                                size=[batch_size, 1]), dtype=dtype)
    else:
        random_p1 = torch.tensor(np.random.uniform(low=position_limit[0][0], high=position_limit[0][1], size=[batch_size, 1]), dtype=dtype)
        random_p2 = torch.tensor(np.random.uniform(low=position_limit[1][0], high=position_limit[1][1], size=[batch_size, 1]), dtype=dtype)
        random_p3 = torch.tensor(np.random.uniform(low=position_limit[2][0], high=position_limit[2][1], size=[batch_size, 1]), dtype=dtype)

    random_T = torch.eye(4,4).expand(batch_size,4,4).to(torch.float)
    random_T = random_T.clone()
    random_T[:,0:3,0:3] = random_R
    random_T[:,0:3,3] = torch.cat((random_p1,random_p2,random_p3),dim=1)
    random_T.requires_grad = True

    return random_T


def Gaussian_sampling_SE3(Ttraj, w_std, p_std, batch_size):
    eps = 1e-10
    if w_std == 0:
        w_std = eps
    if p_std == 0:
        p_std = eps
    # num_timesteps = qtraj.shape[0]
    # T = exp_so3_T(qtraj)
    T = Ttraj
    num_timesteps = T.shape[0]
    traj_samples = T[torch.randint(0, num_timesteps, (batch_size,))]
    w_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * w_std)
    p_distribution = torch.distributions.multivariate_normal.MultivariateNormal(torch.zeros(3), torch.eye(3) * p_std)
    
    Gaussian_w = w_distribution.sample((batch_size,)).to(Ttraj)
    Gaussian_p = p_distribution.sample((batch_size,)).unsqueeze(-1).to(Ttraj)
    R_samples = traj_samples[:, :3, :3] @ exp_so3(Gaussian_w)
    p_samples = traj_samples[:, :3, 3:4] + Gaussian_p  # 끝 부분에 3:4는 3 한거랑 결과 값은 같게나오지만 dim=(n,3,1)이 나옴 (3 넣으면 (n,3))
    random_T = torch.cat([torch.cat([R_samples, p_samples], dim=2), 
                          torch.zeros(batch_size, 1, 4, device=Ttraj.device)],dim=1).detach()
    
    ######### if you need random_q ########
    # random_T.requires_grad=True
    # random_q = screw_bracket(log_SO3_T(random_T))
    # if isinstance(random_q, str):
    #     random_q = Gaussian_sampling(qtraj, w_std, p_std, batch_size)
    # random_q.requires_grad = True  # n x 6 dim

    return random_T  # random_q


def exp_so3_T(S):
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3

    eps = 1e-014
    W = skew(w)
    T = torch.cat([torch.cat([exp_so3(w), v], dim=2), (torch.zeros(n, 1, 4, device=S.device))], dim=1)
    T[:, -1, -1] = 1
    return T

def Dexp_so3(w):
    R = exp_so3(w)
    N = w.shape[0]
    Id = torch.eye(3).to(w)
    dRdw = torch.zeros(N, 3, 3, 3).to(w)
    wnorm = torch.sqrt(torch.einsum('ni,ni->n', w, w))
    eps = 1e-5
    e_skew = skew(Id)
    if w.shape == (N, 3):
        W = skew(w)
    else:
        W = w
        w = skew(W)
        assert (False)
    temp1 = torch.einsum('ni,njk->nijk', w, W) # ni11, n1jk -> nijk   
    temp2 = torch.einsum('njk,nki->nij', W, (-R + torch.eye(3).to(w)))
    temp2_2 = skew(temp2.reshape(N * 3, 3)).reshape(N, 3, 3, 3)
    wnorm_square = torch.einsum('ni,ni->n', w, w).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)  # (N, 1, 1, 1) torch.tensordot(a, b)[range(10), range(10)]
    dRdw[wnorm > eps] = (((temp1 + temp2_2) / wnorm_square) @ R.unsqueeze(1))[wnorm > eps]
    dRdw[wnorm < eps] = e_skew

    # dRdw is (N, 3, 3, 3) tensor. dRdw(n, i, :, :) is dR/dwi of n^th sample from the batch

    return dRdw

def Vb_to_qdot(Vb, T):
    R = T[:, :3, :3]
    Rt = R.transpose(1, 2)

    w = Vb[:, :3,:]
    v = R @ Vb[:, 3:,:]
    
    Dexp = Dexp_so3(skew(log_SO3(R.to(T.device))))
    Temp = torch.einsum('nij, nkjl -> nkil', Rt, Dexp)
    Dexp_qdot_w = skew(Temp.reshape(-1, 3, 3)).reshape(-1, 3, 3)

    qdot = (torch.cat([torch.inverse(Dexp_qdot_w.transpose(1, 2)) @ w, v], dim=1).squeeze(2))

    return qdot

def qdot_to_Vb(qdot, T):
    R = T[:, :3, :3]
    Rt = R.transpose(1, 2)
    
    qdot_w = qdot[:,:3,:].to(T)
    qdot_v = qdot[:,3:,:].to(T)
    
    Dexp = Dexp_so3(skew(log_SO3(R)))
    Temp = torch.einsum('nij, nkjl -> nkil', Rt, Dexp)
    Dexp_qdot_w = skew(Temp.reshape(-1, 3, 3)).reshape(-1, 3, 3)
    
    Vb = torch.cat([Dexp_qdot_w.transpose(1, 2) @ qdot_w ,Rt @ qdot_v], dim=1)
    
    return Vb

def Tdot_to_Vb(Tdot, T):
    if len(Tdot.shape) == 2 and Tdot.shape[-1] == 4:
        Tdot = Tdot.unsqueeze(0)
    if len(Tdot.shape) == 1 and Tdot.shape[-1] == 6:
        Tdot = Tdot.unsqueeze(0)
    if len(T.shape) == 2 and T.shape[-1] == 4:
        T = T.unsqueeze(0)
    if len(T.shape) == 1 and T.shape[-1] == 6:
        T = T.unsqueeze(0)
    
    Vb = inverse_SE3(T) @ Tdot # torch.einsum('nij, njk -> nik', inverse_SE3(T), Tdot)
    return Vb

def Rdot_to_wb(Rdot, R):
    wb = R.transpose(-1,-2) @ Rdot
    return wb

def wb_to_Rdot(wb, R):
    Rdot = R @ wb
    return Rdot

def Vb_to_Tdot(Vb, T):
    if len(Vb.shape) == 2 and Vb.shape[-1] == 4:
        Vb = Vb.unsqueeze(0)
    if len(Vb.shape) == 1 and Vb.shape[-1] == 6:
        Vb = Vb.unsqueeze(0)
    if Vb.shape[-1] == 6:
        Vb = skew_se3(Vb)
    if len(T.shape) == 2 and T.shape[-1] == 4:
        T = T.unsqueeze(0)
    if len(T.shape) == 1 and T.shape[-1] == 6:
        T = T.unsqueeze(0)
    
    Tdot = torch.einsum('nij, njk -> nik', T, Vb)
    return Tdot

def Tdot_to_wbpdot(Tdot, T):
    Vb = Tdot_to_Vb(Tdot, T)
    wb = screw_bracket(Vb)[:,:3]
    wb_skew = skew(Vb[:,:3,:3])
    pdot = Tdot[:, :3,3]
    return torch.cat([wb, pdot], dim=-1)

def wbpdot_to_Vb(wbpdot, T):
    R = T[:, :3, :3]
    w = wbpdot[:, :3]
    pdot = wbpdot[:, 3:].unsqueeze(-1)
    wb = skew(w)
    Rdot = R @ wb
    Tdot = torch.cat([torch.cat([Rdot, pdot], dim=-1), torch.zeros(T.shape[0],1,4).to(T)], dim=-2)
    Vb = Tdot_to_Vb(Tdot, T)
    return Vb

def wbpdot_to_Tdot(wbpdot, T):
    Vb = wbpdot_to_Vb(wbpdot, T)
    Tdot = Vb_to_Tdot(Vb, T)
    return Tdot

def Vb_to_Vs(Vb, T):
    if Vb.shape[-1] == 6:
        Vb = skew_se3(Vb)
    T_Vb = torch.einsum('nij, njk -> nik', T, Vb)
    Vs = torch.einsum('nij, njk -> nik', T_Vb, inverse_SE3(T))
    return Vs

def Vs_to_Vb(Vs, T):
    if Vs.shape[-1] == 6:
        Vs = skew_se3(Vs)
    Tinv_Vs = torch.einsum('nij, njk -> nik', inverse_SE3(T), Vs)
    Vb = torch.einsum('nij, njk -> nik', Tinv_Vs, T)
    return Vb

def geodesic_SO3(R1, R2, t): #t : 0 ~ 1
    if R1.shape[0] == 3 and R2.shape[0] != 3:
        R1 = R1.repeat(R2.shape[0],1,1)
    elif R1.shape[0] == 3:
        R1 = R1.unsqueeze(0)
    if R2.shape[0] == 3 and R1.shape[0] != 3:
        R2 = R2.repeat(R1.shape[0],1,1)
    elif R2.shape[0] == 3:
        R2 = R2.unsqueeze(0)
    
    geodesic_point = R1@exp_so3(t*log_SO3(torch.einsum('nji, njk -> nik', R1, R2))).squeeze()
    
    return geodesic_point.squeeze()


def get_geodesic_dist_SO3(R1, R2):
    if R1.shape[0] == 3 and R2.shape[0] != 3:
        R1 = R1.repeat(R2.shape[0],1,1)
    elif R1.shape[0] == 3:
        R1 = R1.unsqueeze(0)
    if R2.shape[0] == 3 and R1.shape[0] != 3:
        R2 = R2.repeat(R1.shape[0],1,1)
    elif R2.shape[0] == 3:
        R2 = R2.unsqueeze(0)
    
    dist = torch.linalg.matrix_norm(log_SO3(torch.einsum('nji, njk -> nik', R1, R2)), dim=(1,2)).squeeze()
    
    return dist


def get_closest_point_SO3(R, traj, index=True):
    SO3_traj_dist = get_geodesic_dist_SO3(R, traj)
    index_closest = torch.argmin(SO3_traj_dist)
    
    if index == True:
        return traj[index_closest], index_closest
    else:
        return traj[index_closest]


def parallel_transport_SO3(R1, R2, V):
    # print("V :", V[0])
    w = log_SO3(R1.transpose(-1,-2) @ R2)
    R1TV = R1.transpose(-1,-2) @ V
    V_parallel = R1 @ exp_so3(0.5*w) @ R1TV @ exp_so3(0.5*w)
    # print("V_parallel :", V_parallel[0])
    return V_parallel
    # R2R1T = torch.einsum('nij, nkj -> nik', R2.to(R1), R1)
    # return torch.einsum('nij, njk -> nik', R2R1T, V.to(R1))


def vel_geo_0_SO3(R1, R2): # vel at R1 to R2
    W = log_SO3(torch.einsum('nji, njk -> nik', R1, R2.to(R1)))
    Rdot = torch.einsum('nij, njk -> nik', R1, W.to(R1))
    return Rdot

def Rtraj_to_Rdottraj(R):
    R1 = R[:-1].to(R)
    R2 = R[1:].to(R)
    W = log_SO3(torch.einsum('nji, njk -> nik', R1, R2))
    Rdot = torch.einsum('nij, njk -> nik', R1, W.to(R))
    Rdot = torch.cat([Rdot, torch.zeros([1,3,3]).to(R)], dim=0)
    Rdot_proj = Rdot_projection(Rdot, R)
    return Rdot_proj

def Rdot_projection(Rdot, R):
    return R@(R.permute(0,2,1)@Rdot - Rdot.permute(0,2,1)@R)/2


def gvf_SO3(Rsample, eta, Rtraj, Rdottraj):
    Rtraj_closest, index_closest = get_closest_point_SO3(Rsample, Rtraj, index=True)
    Rdot_closest = Rdottraj[index_closest]
    
    V1 = parallel_transport_SO3(Rtraj_closest, Rsample, Rdot_closest)
    vel = vel_geo_0_SO3(Rsample, Rtraj_closest)
    
    if type(eta) != torch.Tensor:
        eta = torch.zeros(len(Rsample), 1).to(Rtraj) + eta
    elif len(eta.shape) == 1:
        eta = eta.unsqueeze(1).to(Rtraj)
    
    return V1 + eta * vel


def gvf_SE3(Tsample, eta_R, eta_p, Ttraj, Tdottraj, version='Tdot'):
    psample_long = Tsample[0:3,3].unsqueeze(0).repeat(Ttraj.shape[0],1)
    distance_p = torch.norm(psample_long - Ttraj[:,0:3,3].to(Tsample), dim=1)
    
    index_closest = torch.argmin(distance_p)
    closest_point = Ttraj[index_closest,0:3,3]
    pdot_parallel = Tdottraj[index_closest,0:3,3].to(Tsample)
    pdot_contract = closest_point.to(Tsample) - Tsample[0:3,3].to(Tsample)
    
    # Rdot
    Rdot_closest = Tdottraj[index_closest,0:3,0:3]
    Rdot_parallel = parallel_transport_SO3(Ttraj[index_closest,0:3,0:3].unsqueeze(0), Tsample[0:3,0:3].unsqueeze(0), Rdot_closest.unsqueeze(0)).squeeze()
    Rdot_contract = vel_geo_0_SO3(Tsample[0:3,0:3].unsqueeze(0), Ttraj[index_closest,0:3,0:3].unsqueeze(0)).squeeze()
    
    if version == 'qdot':
        ##### qdt version #####
        # [Rdot,pdot]
        Tdot_parallel = torch.zeros(4,4).to(Tsample)
        Tdot_parallel[0:3,0:3] = Rdot_parallel
        Tdot_parallel[0:3,3] = pdot_parallel
        Tdot_contract = torch.zeros(4,4).to(Tsample)
        Tdot_contract[0:3,0:3] = Rdot_contract
        Tdot_contract[0:3,3] = pdot_contract
        
        # twist
        Vb_parallel = Tdot_to_Vb(Tdot_parallel, Tsample).squeeze()
        Vb_contract = Tdot_to_Vb(Tdot_contract, Tsample).squeeze()
        
        # qdot
        qdot_parallel = Vb_to_qdot(screw_bracket(Vb_parallel.unsqueeze(0)).unsqueeze(2), Tsample.unsqueeze(0)).squeeze()
        qdot_contract = Vb_to_qdot(screw_bracket(Vb_contract.unsqueeze(0)).unsqueeze(2), Tsample.unsqueeze(0)).squeeze()
        
        # qdot_parallel = Tdot_to_qdot(Tdot_parallel, Tsample).squeeze()
        # qdot_contract = Tdot_to_qdot(Tdot_contract, Tsample).squeeze()
        
        if eta_R == torch.inf or eta_p == torch.inf:
            qdot = qdot_contract
        else:
            qdot = torch.zeros(6)
            qdot[0:3] = qdot_parallel[0:3] + eta_R*qdot_contract[0:3]
            qdot[3:] = qdot_parallel[3:] + eta_p*qdot_contract[3:]
                
        return qdot
        
    elif version == 'Tdot':
        # Tdot version
        if eta_R == torch.inf or eta_p == torch.inf:
            Tdot = torch.zeros(4,4).to(Tsample)
            # Rdot_contract = Rdot_proejction(Rdot_contract.unsqueeze(0), Tsample[0:3,0:3].unsqueeze(0)).squeeze()
            Tdot[:3,:3] = Rdot_contract
            Tdot[:3,3] = pdot_contract
        else:
            Tdot = torch.zeros(4,4).to(Tsample)
            Tdot[:3,:3] = Rdot_parallel + eta_R*Rdot_contract
            # Tdot[:3,:3] = Rdot_proejction(Tdot[:3,:3].unsqueeze(0), Tsample[0:3,0:3].unsqueeze(0)).squeeze()
            Tdot[:3,3] = pdot_parallel + eta_p*pdot_contract
        
        return Tdot


def get_geodesic_dist_SE3(T1, T2, c=1, d=100):
    dist_R = get_geodesic_dist_SO3(T1[:,0:3,0:3], T2[:,0:3,0:3])
    dist_p = torch.norm(T1[:,0:3,3] - T2[:,0:3,3], dim=-1)
    dist = torch.sqrt(c*dist_R**2 + d*dist_p**2)
    return dist


def get_closest_point_SE3(Tsample, Ttraj, c=1, d=100, index=True):
    nb, _, _ = Tsample.shape
    nt, _, _ = Ttraj.shape
    Tsample = Tsample.unsqueeze(1).repeat(1, nt, 1, 1).reshape(nb*nt, 4, 4)
    Ttraj = Ttraj.unsqueeze(0).repeat(nb, 1, 1, 1).reshape(nb*nt, 4, 4)
    dist = get_geodesic_dist_SE3(Tsample, Ttraj, c=c, d=d).reshape(nb, nt)
    index_closest = torch.argmin(dist, dim=1)
    T_closest = Ttraj[index_closest,:,:]
    if index:
        return T_closest, index_closest
    else:
        return T_closest
    

def Tdot_to_qdot(Tdot, T):
    Vb = Tdot_to_Vb(Tdot, T)
    qdot = Vb_to_qdot(screw_bracket(Vb).unsqueeze(-1), T)
    return qdot


def gvf_SE3_distance(Tsample, eta_R, eta_p, Ttraj, Tdottraj, c=1, d=100, version='Tdot'):
    if len(Tsample.shape) == 2:
        Tsample = Tsample.unsqueeze(0)
    
    T_closest, index_closest = get_closest_point_SE3(Tsample, Ttraj, c=c, d=d, index=True)
    
    # pdot
    pdot_parallel = Tdottraj[index_closest,0:3,3].to(Tsample)
    pdot_contract = T_closest[:,0:3,3].to(Tsample) - Tsample[:,0:3,3].to(Tsample)
    
    # Rdot
    Rdot_closest = Tdottraj[index_closest,0:3,0:3]
    Rdot_parallel = parallel_transport_SO3(Ttraj[index_closest,0:3,0:3], Tsample[:,0:3,0:3], Rdot_closest)
    Rdot_contract = vel_geo_0_SO3(Tsample[:,0:3,0:3], Ttraj[index_closest,0:3,0:3])
    
    if eta_R == torch.inf or eta_p == torch.inf:
        Tdot = torch.zeros_like(Tsample).to(Tsample)
        Tdot[:,:3,:3] = Rdot_contract
        Tdot[:,:3,3] = pdot_contract
    else:
        Tdot = torch.zeros_like(Tsample).to(Tsample)
        Tdot[:,:3,:3] = Rdot_parallel + eta_R*Rdot_contract
        Tdot[:,:3,3] = pdot_parallel + eta_p*pdot_contract
    if version == 'qdot':
        output = Tdot_to_qdot(Tdot, Tsample)
    elif version == 'wbpdot':
        output = Tdot_to_wbpdot(Tdot, Tsample)
    else:
        output = Tdot
    
    return output

# def gvf_SE3_gaussian(Tsample, eta_R, eta_p, Ttraj, Tdottraj, num_sampling, w_std=0, p_std=0):
    
#     Tsample_gaussian = Gaussian_sampling_SE3(Tsample, w_std=w_std, p_std=p_std, batch_size=num_sampling)
#     qdot_list = []
    
#     for num in range(num_sampling):
#         qdot = gvf_SE3(Tsample_gaussian[num], eta_R, eta_p, Ttraj, Tdottraj)
#         qdot_list.append(qdot.unsqueeze(0))
#     qdot_torch = torch.cat(qdot_list, dim=0)
#     qdot_mean = qdot_torch.mean(dim=0)
    
#     return qdot_mean

# def gvf_SE3_gaussian_Vs(Tsample, eta_R, eta_p, Ttraj, Tdottraj, num_sampling, w_std=0, p_std=0):
    
#     Tsample_gaussian = Gaussian_sampling_SE3(Tsample, w_std=w_std, p_std=p_std, batch_size=num_sampling)
#     Vs_list = []
    
#     for num in range(num_sampling):
#         qdot = gvf_SE3(Tsample_gaussian[num], eta_R, eta_p, Ttraj, Tdottraj)
#         Vb = qdot_to_Vb(Tsample_gaussian[num].unsqueeze(0),qdot.unsqueeze(0).unsqueeze(2)).squeeze()
#         Vs = Vb_to_Vs(Tsample_gaussian[num].unsqueeze(0), Vb.unsqueeze(0))
#         Vs_list.append(Vs)
#     Vs_torch = torch.cat(Vs_list, dim=0)
#     Vs_mean = Vs_torch.mean(dim=0)
#     Vb_mean = Vs_to_Vb(Tsample, Vs_mean.unsqueeze(0))
#     qdot_mean = Vb_to_qdot(Tsample, skew_se3(Vb_mean).unsqueeze(2))
#     return qdot_mean.squeeze()
    
def Ttraj_to_Tdottraj(Ttraj, dt=0.002):
    Rtraj = Ttraj[:,0:3,0:3].type(torch.float).to(Ttraj)
    p = Ttraj[:,0:3,3].type(torch.float).to(Ttraj)
    Rdottraj = Rtraj_to_Rdottraj(Rtraj)
    pdottraj = torch.cat((p[1:]-p[:-1],torch.zeros([1,3]).to(Ttraj)),dim=0)
    Tdottraj = torch.zeros(Ttraj.shape).to(Ttraj)
    Tdottraj[:,0:3,0:3] = Rdottraj
    Tdottraj[:,0:3,3] = pdottraj
    return Tdottraj/dt

def Ttraj_to_Tdottraj_list(Ttraj_list, dt_list):
    Tdottraj_list = []
    for i in range(len(Ttraj_list)):
        Tdottraj = Ttraj_to_Tdottraj(Ttraj_list[i], dt_list[i])
        Tdottraj_list.append(Tdottraj)
    return Tdottraj_list

def SE3_update(T, Tdot, dt):
    wb = T[0:3,0:3].T @ Tdot[:3,:3]
    Rb = exp_so3(wb.unsqueeze(0)*dt).squeeze()
    R = T[0:3,0:3] @ Rb
    p = Tdot[:3,3].to(T)*dt + T[:3,3]
    T_new = torch.eye(4,4)
    T_new[:3,:3] = R.squeeze()
    T_new[:3,3] = p
    return T_new.to(T)

def SE3_update_batch(T, Tdot, dt):
    wb = T[:,0:3,0:3].permute(0,2,1) @ Tdot[:,:3,:3]
    Rb = exp_so3(wb*dt)
    R = T[:,0:3,0:3] @ Rb
    p = Tdot[:,:3,3].to(T)*dt + T[:,:3,3]
    T_new = torch.eye(4,4).unsqueeze(0).repeat(T.shape[0],1,1)
    T_new[:,:3,:3] = R
    T_new[:,:3,3] = p
    return T_new.to(T)

def SE3_gvf_traj_gen(start_SE3, eta_R, eta_p, SE3_demo, num_update, dt=0.001):
    Tdottraj = Ttraj_to_Tdottraj(SE3_demo)
    SE3_traj = torch.zeros([num_update,4,4])
    
    for i in range(num_update):
        if i == 0:
            T_old = start_SE3.to(start_SE3)
        else:
            T_old = T_new
        
        SE3_traj[i] = T_old
        
        vf_T = gvf_SE3_distance(T_old.to(start_SE3), eta_R, eta_p, SE3_demo.to(start_SE3),Tdottraj.to(start_SE3), c=1, d=100).to(start_SE3).squeeze()
        
        ##### qdot version #####
        # #update R
        # dR_dtheta = Dexp_so3(skew(log_SO3(T_old[:3,:3].unsqueeze(0).to(start_SE3))))
        # R = T_old[:3,:3].unsqueeze(0)
        # Rdot = torch.einsum('nijk,ni->njk', dR_dtheta.to(start_SE3), vf_T[:3].unsqueeze(0))
        # Rt = R.transpose(1, 2)
        # R = R.to(start_SE3) @ exp_so3(Rt.to(start_SE3) @ Rdot.to(start_SE3))
        # #update p
        # p = vf_T[3:].to(start_SE3) + T_old[:3,3].to(start_SE3)
        ##### qdot version end #####
        
        ##### Tdot version #####
        wb = T_old[0:3,0:3].T @ vf_T[:3,:3] * dt
        Rb = exp_so3(wb.unsqueeze(0)).squeeze()
        R = T_old[0:3,0:3] @ Rb
        p = vf_T[:3,3].to(start_SE3) * dt + T_old[:3,3].to(start_SE3)
        ##### Tdot end #####
        
        #update SE3
        T_new = torch.eye(4,4)
        T_new[:3,:3] = R.squeeze()
        T_new[:3,3] = p

    return SE3_traj

# def h(X):
#     n = X.shape[0]
#     R = X[:, :3, :3]
#     Rv = R.reshape(n, -1)
#     b = X[:, 0:3, 3]
#     X_se3 = torch.cat((Rv, b), dim=1)
#     return X_se3

def mat2vec_SE3(X):
    # xshape = X.shape
    # R = X[..., :3, :3]
    # Rv = R.reshape(*(xshape[:-2]), -1)
    # b = X[..., 0:3, 3]
    # X_se3 = torch.cat((Rv, b), dim=-2)
    # return X_se3
    
    xshape = X.shape
    R = X[:, :3, :3]
    Rv = R.reshape(*(xshape[:-2]), -1)
    b = X[:, 0:3, 3]
    X_se3 = torch.cat((Rv, b), dim=-1)
    return X_se3

def mat2vec_SE3_batch(X):
    xshape = X.shape
    R = X[:, :, :3, :3]
    Rv = R.reshape(*(xshape[:-2]), -1)
    b = X[:, :, 0:3, 3]
    X_se3 = torch.cat((Rv, b), dim=-1)
    return X_se3

def vec2mat_SE3(V):
    xshape = V.shape
    R = V[:, :9].reshape(*(xshape[:-1]), 3, 3)
    b = V[:, 9:].unsqueeze(-1)
    X_se3 = torch.cat((R, b), dim=-1)
    eye = (torch.eye(4).to(V)[-1]).unsqueeze(0).unsqueeze(0).repeat(*xshape[:-1], 1, 1)
    X_se3 = torch.cat([X_se3, eye], dim=-2)
    return X_se3

def mat2vec_SE3_traj(Xtraj):
    n = Xtraj.shape[0]
    m = Xtraj.shape[1]
    R = Xtraj[:, :, :3, :3]
    Rv = R.reshape(n, m, -1)
    b = Xtraj[:, :, 0:3, 3]
    Xtraj_se3 = torch.cat((Rv, b), dim=2)
    return Xtraj_se3.reshape(n, -1)

def SE3_gvfnet_traj_gen(start_SE3, model_type, model=None, eta_R=0.1, eta_p=0.1, time_step=1, num_update=200, self = None):
    
    SE3_traj = torch.zeros([num_update,4,4])
    if self is not None:
        self  = self.to(start_SE3)
    
    for i in range(num_update):
        if i == 0:
            T_old = start_SE3
        else:
            T_old = T_new.to(start_SE3)
        if len(T_old.shape) == 2:
            T_old = T_old.unsqueeze(0)
        
        T_flat = mat2vec_SE3(T_old)
        SE3_traj[i] = T_old
        
        if model_type == 'vanilla':
            vf_T = model.forward(T_old, eta_R, eta_p)        
        elif model_type == 'parallel':
            vf_T = model.parallel_vf.forward(T_flat)
        elif model_type == 'contracting':
            time_step = 0.2
            vf_T = model.contracting_vf.forward(T_flat)
        elif model == 'sddm':
            vf_T = model.forward(T_old, eta_R, eta_p)
        elif model == 'sddm_vanilla':
            vf_T = model.vanilla_vf.forward(T_old, eta_R, eta_p)
        elif model == 'sddm_parallel':
            vf_T = model.vanilla_vf.parallel_vf.forward(T_flat)
        elif model == 'sddm_contracting':
            time_step = 0.2
            vf_T = model.vanilla_vf.contracting_vf.forward(T_flat)
        else:
            vf_T = self.forward(T_old, eta_R, eta_p)
        
        #update
        R = T_old[:,:3,:3]
        dR_dtheta = Dexp_so3(skew(log_SO3(R)))

        Rdot = torch.einsum('nijk,ni->njk', dR_dtheta, vf_T[:,:3].to(dR_dtheta))
        Rt = R.transpose(1, 2)
        R = R @ exp_so3(Rt @ Rdot*time_step)
        p = vf_T[:,3:].to(T_old)*time_step + T_old[:,:3,3]
        
        T_new = torch.eye(4,4)
        T_new[:3,:3] = R.squeeze()
        T_new[:3,3] = p
    
    return SE3_traj

def SE3_latentnet_traj_gen(start_SE3, model_type, latent=None, model=None, eta_R=0.1, eta_p=0.1, time_step=1, num_update=400, self = None):
    
    SE3_traj = torch.zeros([num_update,4,4])
    if self is not None:
        self  = self.to(start_SE3)
    
    for i in range(num_update):
        if i == 0:
            T_old = start_SE3
        else:
            T_old = T_new.to(start_SE3)
        if len(T_old.shape) == 2:
            T_old = T_old.unsqueeze(0)
        
        T_flat = mat2vec_SE3(T_old)
        SE3_traj[i] = T_old
        
        if model_type == 'vanilla' and latent is not None:
            vf_T = model.forward(T_old, eta_R, eta_p, latent)
        elif model_type == 'vanilla':
            vf_T = model.forward(T_old, eta_R, eta_p)
        elif model_type == 'parallel':
            vf_T = model.parallel_vf.forward(T_flat)
        elif model_type == 'contracting':
            time_step = 0.2
            vf_T = model.contracting_vf.forward(T_flat)
        elif model == 'sddm':
            vf_T = model.forward(T_old, eta_R, eta_p)
        elif model == 'sddm_vanilla':
            vf_T = model.vanilla_vf.forward(T_old, eta_R, eta_p)
        elif model == 'sddm_parallel':
            vf_T = model.vanilla_vf.parallel_vf.forward(T_flat)
        elif model == 'sddm_contracting':
            time_step = 0.2
            vf_T = model.vanilla_vf.contracting_vf.forward(T_flat)
        elif latent is not None:
            vf_T = self.forward(T_old, eta_R, eta_p, latent)
        else:
            vf_T = self.forward(T_old, eta_R, eta_p)
        
        #update
        R = T_old[:,:3,:3]
        dR_dtheta = Dexp_so3(skew(log_SO3(R)))

        Rdot = torch.einsum('nijk,ni->njk', dR_dtheta, vf_T[:,:3].to(dR_dtheta))
        Rt = R.transpose(1, 2)
        R = R @ exp_so3(Rt @ Rdot*time_step)
        p = vf_T[:,3:].to(T_old)*time_step + T_old[:,:3,3]
        
        T_new = torch.eye(4,4)
        T_new[:3,:3] = R.squeeze()
        T_new[:3,3] = p
    
    return SE3_traj

def SE3_deeponet_traj_gen(start_SE3, Ttraj, model=None, eta_R=5, eta_p=5, dt=0.0705, num_update=200, self=None):
    
    SE3_traj = torch.zeros([num_update,4,4])
    if self is not None:
        self  = self.to(start_SE3)
    
    for i in range(num_update):
        if i == 0:
            T_old = start_SE3
        else:
            T_old = T_new.to(start_SE3)
        if len(T_old.shape) == 2:
            T_old = T_old.unsqueeze(0).to(start_SE3)
        
        # T_flat = h(T_old)
        SE3_traj[i] = T_old
        
        # model forward
        if model is not None:
            vf_T = model.forward(T_old, Ttraj.to(start_SE3), eta_R, eta_p)
        elif self is not None:
            vf_T = self.forward(T_old, Ttraj.to(start_SE3), eta_R, eta_p)
        
        if vf_T.shape[-1] == 4:
            Tdot = vf_T
            # print("T_old shape", T_old.shape)
            # print("Tdot shape", Tdot.shape)
            T_new = SE3_update(T_old.squeeze(), Tdot.squeeze(), dt)
        else:
            # update
            R = T_old[:,:3,:3]
            dR_dtheta = Dexp_so3(skew(log_SO3(R)))

            Rdot = torch.einsum('nijk,ni->njk', dR_dtheta, vf_T[:,:3].to(dR_dtheta))
            Rt = R.transpose(1, 2)
            R = R @ exp_so3(Rt @ Rdot*dt)
            p = vf_T[:,3:].to(T_old)*dt + T_old[:,:3,3]
            
            T_new = torch.eye(4,4)
            T_new[:3,:3] = R.squeeze()
            T_new[:3,3] = p
    
    return SE3_traj


def SE3_RSDS_traj_gen(start_SE3, model=None, dt=0.0705, num_update=200, self=None):
    ts = time.time()
    SE3_traj = torch.zeros([num_update,4,4])
    # if self is not None:
    #     self  = self.to(start_SE3)
    
    for i in range(num_update):
        # print(i)
        if i == 0:
            T_old = start_SE3
        else:
            T_old = T_new.to(start_SE3)
        if len(T_old.shape) == 2:
            T_old = T_old.unsqueeze(0).to(start_SE3)
        
        # T_flat = h(T_old)
        SE3_traj[i] = T_old
        
        # model forward
        if model is not None:
            vf_T = model.forward(T_old).detach().cpu()
        elif self is not None:
            vf_T = self.forward(T_old).detach().cpu()
        
        if vf_T.shape[-1] == 4:
            Tdot = vf_T
            # print("T_old shape", T_old.shape)
            # print("Tdot shape", Tdot.shape)
            T_new = SE3_update(T_old.squeeze(), Tdot.squeeze(), dt)
        else:
            # update
            R = T_old[:,:3,:3]
            dR_dtheta = Dexp_so3(skew(log_SO3(R)))

            Rdot = torch.einsum('nijk,ni->njk', dR_dtheta, vf_T[:,:3].to(dR_dtheta))
            Rt = R.transpose(1, 2)
            R = R @ exp_so3(Rt @ Rdot*dt)
            p = vf_T[:,3:].to(T_old)*dt + T_old[:,:3,3]
            
            T_new = torch.eye(4,4)
            T_new[:3,:3] = R.squeeze()
            T_new[:3,3] = p
    # print(f'time elapsed for traj generation: {time.time()-ts}')
    return SE3_traj


def vec_6dim_to_12dim(x, vec):
    if x.shape[1:] == (12,):
        R = x[:, :9].reshape(-1, 3, 3)
    elif x.shape[1:] == (4, 4):
        R = x[:, :3, :3]

    w = vec[:, :3]
    v = vec[:, 3:]
    # I = torch.eye(3).to(x).unsqueeze(0).repeat(x.shape[0], 1, 1)
    Rdot_I = skew(w)
    Rdot_x = R @ Rdot_I
    vec_12dim = torch.cat([Rdot_x.reshape(-1, 9), v], dim=1)
    return vec_12dim


def vec_12dim_to_6dim(x, vec):
    if x.shape[1:] == (12,):
        R = x[:, :9].reshape(-1, 3, 3)
    elif x.shape[1:] == (4, 4):
        R = x[:, :3, :3]
    if vec.shape[1:] == (12,):
        Rdot = vec[:, :9].reshape(-1, 3, 3)
        pdot = vec[:, 9:]
    elif vec.shape[1:] == (4, 4):
        Rdot = vec[:, :3, :3]
        pdot = vec[:, :3, 3]
    # Rdot = vec[:, :9].reshape(-1, 3, 3)
    W = R.transpose(1, 2) @ Rdot
    w = skew(W)
    # breakpoint()
    vec_6dim = torch.cat([w, pdot], dim=1)
    return vec_6dim


def tangent_gaussian_sampling(q, std_R=0.2, std_p=1, sample_size=100):
    if q.shape[-1] == 6:
        if len(q.shape) == 1:
            squeezed = True
            x = exp_so3_T(q.unsqueeze(0))
    elif q.shape[-2:] == (4, 4):
        x = q
        if len(x.shape) == 2:
            squeezed = True
            x = x.unsqueeze(0)
    nx = len(x)
    x = x.unsqueeze(1) # n, 1, 4, 4
    R = x[:, :, :3, :3]
    p = x[:, :, :3, 3].unsqueeze(-1)
    wsample = torch.empty(nx, sample_size, 3).to(q).normal_(mean=0, std=std_R)
    psample = torch.empty(nx, sample_size, 3, 1).to(q).normal_(mean=0, std=std_p)
    
    Rsample_at_I = exp_so3(wsample.reshape(-1, 3)).reshape(nx, sample_size, 3, 3)
    Rsample = R @ Rsample_at_I
    xsample = torch.cat([Rsample, psample + p], dim=-1)
    last_row = torch.zeros(nx, sample_size, 1, 4).to(q)
    last_row[:, :, -1, -1] = 1 
    xsample = torch.cat([xsample, last_row], dim=-2)
    if squeezed:
        xsample = xsample.squeeze(0)
    return xsample


def resample_traj_torch(traj, length):
    a = len(traj)
    if length <= 0 or a == 0:
        raise ValueError("New length must be greater than 0 and original list must not be empty.")
    
    indices = np.linspace(0, a - 1, length).astype(int)
    traj_length = [traj[i].unsqueeze(0) for i in indices]
    traj_length = torch.cat(traj_length, dim=0)
    return traj_length