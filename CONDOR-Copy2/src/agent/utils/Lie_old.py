import numpy as np
from numpy.linalg import inv
import torch
dtype = torch.float

def skew(w):
    # input = n,3
    n = w.shape[0]
    # 3x3 skew --> vector
    if w.shape == (n, 3, 3):
        W = torch.cat([-w[:, 1, 2].unsqueeze(-1), w[:, 0, 2].unsqueeze(-1),
                       -w[:, 0, 1].unsqueeze(-1)], dim=1)
    # 3 dim vector --> skew
    else:
        zero1 = torch.zeros(n, 1, 1).to(w)
        w = w.unsqueeze(-1).unsqueeze(-1)
        W = torch.cat([torch.cat([zero1, -w[:,2],  w[:, 1]], dim=2),
                       torch.cat([w[:, 2], zero1,  -w[:, 0]], dim=2),
                       torch.cat([-w[:, 1], w[:, 0],  zero1], dim=2)], dim=1)
    return W

def screw_bracket(V):
    if isinstance(V, str):
        # print(V)
        return 'trace error'
    n = V.shape[0]
    out = 0
    if V.shape == (n, 4, 4):
        out = torch.cat([-V[:, 1, 2].unsqueeze(-1), V[:, 0, 2].unsqueeze(-1),
                         -V[:, 0, 1].unsqueeze(-1), V[:, :3, 3]], dim=1)
    else:
        W = skew(V[:, 0:3])
        out = torch.cat([torch.cat([W, V[:, 3:].unsqueeze(-1)], dim=2),
                         torch.zeros(n, 1, 4).to(V)], dim=1)
        # print(torch.cat([W, V[:, 0:3].unsqueeze(-1)], dim=2))
    return out


def exp_so3(Input):
    # shape(w) = (3,1)
    # shape(W) = (3,3)
    n = Input.shape[0]
    if Input.shape==(n,3,3):
        W = Input
        w = skew(Input)
    else:
        w = Input
        W = skew(w)
    wnorm_sq = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)\
    wnorm_sq_unsqueezed = wnorm_sq.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1)
    
    wnorm = torch.sqrt(wnorm_sq)  # (dim = n)
    wnorm_unsqueezed = torch.sqrt(wnorm_sq_unsqueezed)  # dim - (n,1)
    
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    w0 = w[:, 0].unsqueeze(-1).unsqueeze(-1)
    w1 = w[:, 1].unsqueeze(-1).unsqueeze(-1)
    w2 = w[:, 2].unsqueeze(-1).unsqueeze(-1)
    eps = 1e-4
    # R = torch.zeros(n,3,3)
    R = torch.cat((torch.cat((cw - ((w0**2)*(cw - 1))/wnorm_sq_unsqueezed,
                              - (w2*sw)/wnorm_unsqueezed - (w0*w1*(cw - 1))/wnorm_sq_unsqueezed,
                              (w1*sw)/wnorm_unsqueezed - (w0*w2*(cw - 1))/wnorm_sq_unsqueezed), dim=2)
                   , torch.cat(((w2*sw)/wnorm_unsqueezed - (w0*w1*(cw - 1))/wnorm_sq_unsqueezed,
                                cw - ((w1**2)*(cw - 1))/wnorm_sq_unsqueezed,
                               - (w0*sw)/wnorm_unsqueezed - (w1*w2*(cw - 1))/wnorm_sq_unsqueezed), dim=2)
                   , torch.cat((-(w1*sw)/wnorm_unsqueezed - (w0*w2*(cw - 1))/wnorm_sq_unsqueezed,
                                (w0*sw)/wnorm_unsqueezed - (w1*w2*(cw - 1))/wnorm_sq_unsqueezed,
                                cw - ((w2**2)*(cw - 1))/wnorm_sq_unsqueezed), dim=2))
                  , dim=1)
    R[wnorm < eps] = torch.eye(3).to(Input) + W[wnorm < eps] + 1/2*W[wnorm < eps]@W[wnorm < eps]
                              
    return R


def exp_so3_from_screw(S):
    n = S.shape[0]
    if S.shape == (n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3

    T = torch.cat([torch.cat([exp_so3(w), v], dim=2),
                   torch.zeros(n, 1, 4).to(S)], dim=1)
    T[:, -1, -1] = 1
    return T


def exp_se3(S):
    n = S.shape[0]
    if S.shape==(n, 4, 4):
        S1 = skew(S[:, :3, :3]).clone()
        S2 = S[:, 0:3, 3].clone()
        S = torch.cat([S1, S2], dim=1)
    # shape(S) = (n,6,1)
    w = S[:, :3]  # dim= n,3
    v = S[:, 3:].unsqueeze(-1)  # dim= n,3
    wsqr = torch.tensordot(w, w, dims=([1], [1]))[[range(n), range(n)]]  # dim = (n)
    wsqr_unsqueezed = wsqr.unsqueeze(-1).unsqueeze(-1)  # dim = (n,1)
    wnorm = torch.sqrt(wsqr)  # dim = (n)
    wnorm_unsqueezed = torch.sqrt(wsqr_unsqueezed)  # dim - (n,1)
    wnorm_inv = 1/wnorm_unsqueezed # dim = (n)
    cw = torch.cos(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    sw = torch.sin(wnorm).view(-1, 1).unsqueeze(-1)  # (dim = n,1)
    
    eps = 1e-014
    W = skew(w)
    P = torch.eye(3).to(S) + (1-cw)*(wnorm_inv**2)*W + \
        (wnorm_unsqueezed - sw)*(wnorm_inv**3)*torch.matmul(W, W)  # n,3,3
    P[wnorm < eps] = torch.eye(3).to(S)
    T = torch.cat([torch.cat([exp_so3(w), P@v], dim=2), (torch.zeros(n, 1, 4).to(S))], dim=1)
    T[:, -1, -1] = 1
    return T


def inverse_SE3(T):
    n = T.shape[0]
    R = T[:, 0:3, 0:3]  # n,3,3
    p = T[:, 0:3, 3].unsqueeze(-1)  # n,3,1
    T_inv = torch.cat([torch.cat([R.transpose(1, 2), (-R.transpose(1, 2))@p], dim=2),
                       torch.zeros(n, 1, 4).to(T)], dim=1)
    T_inv[:, -1, -1] = 1
    return T_inv


def large_Ad(T):
    n = T.shape[0]
    R = T[:, 0:3, 0:3]  # n,3,3
    p = T[:, 0:3, 3]  # n,3
    AdT = torch.cat([torch.cat([R, torch.zeros(n, 3, 3).to(T)], dim=2),
                     torch.cat([skew(p)@R, R], dim=2)], dim=1)
    return AdT


def small_ad(V):
    # shape(V) = (n,6)
    n = V.shape[0]
    w = V[:, :3]
    v = V[:, 3:]
    wskew = skew(w)
    vskew = skew(v)
    adV = torch.cat([torch.cat([wskew, torch.zeros(n, 3, 3).to(V)], dim=2),
                     torch.cat([vskew, wskew], dim=2)], dim=1)
    return adV

def Lie_bracket(u, v):
    if u.shape[1:] == (3,):
        u = skew(u)
    elif u.shape[1:] == (6,):
        u = screw_bracket(u)

    if v.shape[1:] == (3,):
        v = skew(v)
    elif v.shape[1:] == (6,):
        v = screw_bracket(v)

    return u @ v - v @ u

#여기부터는 SE3 참고

# def log_SO3(R):
#     # w1,w2,w3 are all (n) dim
#     # omega (n,3,3) dim
#     eps = 1e-14
#     trace = torch.sum(R[:, range(3), range(3)], dim=1)  # n
#     tracecheck = torch.sum(trace == -1)
#     # if tracecheck > 0: 
#     #     # print('Track check error occured in Lie.log_SO3')
#     #     # return 'trace error'
    
#     #     #assert tracecheck <= 0, 'error: theta is pi.'
        
#     #     theta = np.pi
#     #     # assume w[0]>0
#     #     w1_sqr = (1+R[0,0])/2
#     #     w2_sqr = (1+R[1,1])/2
#     #     w3_sqr = (1+R[2,2])/2
        
#     #     w1 = np.sqrt(w1_sqr)
#     #     if R[0,1] > 0:
#     #        w2 = np.sqrt(w2_sqr)
#     #     elif R[0,1] ==0 :
#     #        w2 = 0
#     #     else :
#     #        w2 = -np.sqrt(w2_sqr)
#     #     if R[0,2] > 0:
#     #        w3 = np.sqrt(w3_sqr)
#     #     elif R[0,2]==0:
#     #        w3 = 0
#     #     else :
#     #        w3 = -np.sqrt(w3_sqr)
           
#     #     if torch.sum(torch.isnan(w1)) > 0:
#     #         print('w1 =',w1)
#     #     if torch.sum(torch.isnan(w2)) > 0:
#     #         print('w2 =',w2)
#     #     if torch.sum(torch.isnan(w3)) > 0:
#     #         print('w3 =',w3)
        
#     #     omega = skew(np.array([[w1],[w2],[w3]]))*theta
        
#     #     if torch.sum(torch.isnan(omega)) > 0:
#     #         print('omega =',omega)
        
        
#     eps = 1e-6
#     input = (trace-1)/2
#     input = torch.clamp(input, eps, 1-eps)
#     theta = torch.acos(input).unsqueeze(-1).unsqueeze(-1)
#     omega = (theta/(2*torch.sin(theta)))*(R-R.transpose(1,2))
#     omega[trace>=(3-eps)] = 1/2*(R[trace>=(3-eps)]-(R[trace>=(3-eps)]).transpose(1,2))
    
#     if torch.sum(torch.isnan(omega)) > 0:
#             print("omega has Nan")
    
#     return omega

def log_SO3_T(T):
    #dim T = n,4,4
    R = T[:,0:3,0:3] # dim n,3,3
    p = T[:,0:3,3].unsqueeze(-1) # dim n,3,1
    n = T.shape[0]
    W = log_SO3(R) #n,3,3
    
    if isinstance(W, str):
        #print(W)
        return 'trace error'
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
    v = p #n,3,1
    return torch.cat([torch.cat([W,v],dim=2),torch.zeros(n,1,4).to(T)],dim=1) #n,4,4
    

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

def convert_SO3_to_quaternion(R):
    #dim(R) = n,3,3
    W = log_SO3(R) # n,3,3
    w = skew(W) #n,3
    theta_1dim = torch.sqrt(torch.sum(w**2,dim=1))
    theta = theta_1dim.unsqueeze(-1) # n,1
    w_hat = w/theta # n,3
    w_hat[theta_1dim<1.0e-016] = 0
    return torch.cat([w_hat[:,0].unsqueeze(-1)*torch.sin(theta/2),
                      w_hat[:,1].unsqueeze(-1)*torch.sin(theta/2),
                      w_hat[:,2].unsqueeze(-1)*torch.sin(theta/2),
                      torch.cos(theta/2)], dim=1)
