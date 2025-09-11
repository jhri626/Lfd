import torch
import torch.nn as nn
import numpy as np
from scipy import interpolate
from scipy import integrate
from scipy.optimize import fsolve
import scipy
import math
import matplotlib.pyplot as plt
import agent.utils.Lie_old as lie
from torch.autograd.functional import jvp

dtype = torch.float

def q_to_x(thetaphi):
    if thetaphi.shape == (2,):
        theta = thetaphi[0]
        phi = thetaphi[1]
        ct = torch.cos(theta)
        st = torch.sin(theta)
        cp = torch.cos(phi)
        sp = torch.sin(phi)
        x = torch.tensor([st * cp, st * sp, ct]).to(thetaphi)
    else:
        theta = thetaphi[..., 0].unsqueeze(-1)
        phi = thetaphi[..., 1].unsqueeze(-1)
        ct = torch.cos(theta)
        st = torch.sin(theta)
        cp = torch.cos(phi)
        sp = torch.sin(phi)
        x = torch.cat([st * cp, st * sp, ct], dim=-1).to(thetaphi)
    return x.type(torch.DoubleTensor).to(thetaphi)

def qdot_to_xdot(qdot, q):
    if q.shape[-1] == 3:
        q = x_to_q(q)
    pf = get_jacobian(q)
    return (pf @ qdot.unsqueeze(-1)).squeeze(-1)

def xdot_to_qdot(xdot, q):
    eps = 1e-7
    if q.shape[-1] == 3:
        q = x_to_q(q)
    pf = get_jacobian(q)
    pf_t = pf.transpose(1, 2)
    pf_t_pf = pf_t  @ pf
    pf_t_pf = torch.clamp(pf_t_pf, min=eps)
    pf_t_pf_inv = torch.inverse(pf_t_pf)
    mult = pf_t_pf_inv @ pf_t
    return (mult @ xdot.unsqueeze(-1)).squeeze(-1)
    
def xdot_projection(xdot, x):
    if x.shape[-1] == 2:
        x = q_to_x(x)
    xcos = torch.einsum('ni, ni -> n', x, xdot).unsqueeze(-1)
    xdot_proj = xdot - xcos * x
    return xdot_proj

def x_to_q(x):
    eps = 1e-7
    if x.shape == (3,):
        ct = x[2]
        assert 1 > ct > -1, 'theta is equal to pi, infinite solution'
        theta = torch.acos(ct)
        st = torch.sqrt(1 - ct ** 2)
        st = torch.clip(st, min = eps, max=1-eps)
        sp = x[1] / st
        cp = x[0] / st
        if sp >= 0:
            phi = torch.acos(cp)
        else:
            phi = 2 * math.pi - torch.acos(cp)
        thetaphi = torch.tensor([theta, phi], dtype=dtype).to(x)
        

    else:
        n = x.shape[0]

        ct = torch.clip(x[:, 2].unsqueeze(-1), min = -1+eps, max=1-eps)
        assert (sum(sum(ct < -1)) + sum(sum(ct > 1))) == 0, 'theta is equal to pi, infinite solution'
                
        theta = torch.acos(ct)
        phi = torch.atan2(x[:,1], x[:,0]).unsqueeze(-1)
        phi = torch.where(phi < 0, phi + 2*math.pi, phi)

        thetaphi = torch.cat([theta, phi], dim=1)
        # breakpoint()
    return thetaphi

def get_jacobian(input_arg):
    n = input_arg.shape[0]
    if input_arg.shape == (n, 3):
        thetaphi = x_to_q(input_arg)
    elif input_arg.shape == (n, 2):
        thetaphi = input_arg
    else:
        print(input_arg.shape)
        return
    theta = thetaphi[:, 0].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    phi = thetaphi[:, 1].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    ct = torch.cos(theta)
    st = torch.sin(theta)
    cp = torch.cos(phi)
    sp = torch.sin(phi)
    J = torch.cat([torch.cat([ct * cp, -st * sp], dim=2),
                   torch.cat([ct * sp, st * cp], dim=2),
                   torch.cat([-st, torch.zeros(n, 1, 1).to(input_arg)], dim=2)],
                  dim=1)
    return J

def get_jacobian_derivative(input_arg):
    n = input_arg.shape[0]
    if input_arg.shape == (n, 3):
        thetaphi = x_to_q(input_arg)
    elif input_arg.shape == (n, 2):
        thetaphi = input_arg
    else:
        print(input_arg.shape)
        return
    theta = thetaphi[:, 0].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    phi = thetaphi[:, 1].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    ct = torch.cos(theta)
    st = torch.sin(theta)
    cp = torch.cos(phi)
    sp = torch.sin(phi)
    J_t = torch.cat([torch.cat([-st * cp, -ct * sp], dim=2),
                   torch.cat([-st * sp, ct * cp], dim=2),
                   torch.cat([-ct, torch.zeros(n, 1, 1).to(input_arg)], dim=2)],
                  dim=1).unsqueeze(1)
    J_p = torch.cat([torch.cat([ct * -sp, -st * cp], dim=2),
                   torch.cat([ct * cp, st * -sp], dim=2),
                   torch.cat([torch.zeros(n, 1, 1).to(input_arg), torch.zeros(n, 1, 1).to(input_arg)], dim=2)],
                  dim=1).unsqueeze(1)
    dJdq = torch.cat([J_t, J_p], dim=1) # n, 2, 3, 2
    return dJdq



def get_jacobian_dot(q, qdot):
    n = q.shape[0]
    if q.shape == (n, 3):
        thetaphi = x_to_q(q)
    elif q.shape == (n, 2):
        thetaphi = q
    else:
        print(q.shape)
        return
    dJdq = get_jacobian_derivative(q)
    Jdot = torch.einsum('nijk,ni->njk', dJdq, qdot)
    return Jdot

def get_Riemannian_metric(input_arg):
    n = input_arg.shape[0]
    if input_arg.shape == (n, 3):
        thetaphi = x_to_q(input_arg)
    elif input_arg.shape == (n, 2):
        thetaphi = input_arg
    else:
        print(input_arg.shape)
        return
    theta = thetaphi[:, 0].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    # phi = thetaphi[:, 1].unsqueeze(-1).unsqueeze(-1).to(input_arg)
    ones = torch.ones(n, 1, 1).to(input_arg)
    zeros = torch.zeros(n, 1, 1).to(input_arg)
    st_2 = torch.sin(theta)**2
    G = torch.cat([
        torch.cat([ones, zeros], dim=2),
        torch.cat([zeros, st_2], dim=2)
    ], dim=1)
    return G

def gaussian_sampling(qtraj, std, batch_size):
    num_timesteps = qtraj.shape[0]
    X = q_to_x(qtraj)
    traj_samples = X[torch.randint(0, num_timesteps, [batch_size])].unsqueeze(-1).to(qtraj)
    Gaussian_v = torch.empty(batch_size, 2, 1, 1).normal_(mean=0, std=std).to(qtraj)
    e1_temp = torch.empty(batch_size, 3, 1).normal_(mean=0, std=1).to(qtraj)
    e1_temp2 = e1_temp - traj_samples @ (traj_samples.transpose(1, 2)) @ e1_temp
    e1_temp3 = torch.sqrt(torch.sum(e1_temp2 ** 2, 1)).unsqueeze(-1)
    e1 = e1_temp2 / e1_temp3
    e2 = torch.cross(traj_samples, e1)
    d_v = Gaussian_v[:, 0] * e1 + Gaussian_v[:, 1] * e2
    d_v_norm = torch.sqrt(torch.sum(d_v ** 2, 1)).unsqueeze(-1)
    Random_X = (traj_samples * torch.cos(d_v_norm) +
                d_v / d_v_norm * torch.sin(d_v_norm)).view(batch_size, 3)
    Random_q = x_to_q(Random_X)
    return Random_q.detach()


def uniform_sampling(batch_size, return_local=True):
    eps = 0.001
    Random_ball = torch.from_numpy(np.random.normal(0, 1, [int(batch_size * 2), 3])).to(dtype)
    Random_ball_norm = torch.sqrt(torch.sum(Random_ball ** 2, dim=1).unsqueeze(-1))
    Random_ball = Random_ball[Random_ball_norm>eps][:batch_size]
    Random_ball_norm = Random_ball_norm[Random_ball_norm>eps][:batch_size]
    Random_X = Random_ball / Random_ball_norm
    if return_local:
        Random_q = x_to_q(Random_X)
        return Random_q.detach()
    else:
        return Random_X


def grid_sampling(batch):
    # canonical Fibonacci Lattice
    # source
    # https://extremelearning.com.au/how-to-evenly-distribute-points-on-a-sphere-more-effectively-than-the-canonical-fibonacci-lattice/
    
    n = batch
    goldenRatio = (1 + 5**0.5)/2
    i = torch.arange(0, n)
    theta = 2 *torch.pi * i / goldenRatio
    phi = torch.arccos(1 - 2*(i+0.5)/n)
    x, y, z = torch.cos(theta) * torch.sin(phi), torch.sin(theta) * torch.sin(phi), torch.cos(phi)
    grid = torch.cat([x.unsqueeze(1),y.unsqueeze(1),z.unsqueeze(1)], dim=1) # [batch, 3]
    return grid


def Sphere_spline(xpoints, ypoints, vpoints, T, delta, num_timesteps):
    af = 10
    deltaT = delta * T
    aa = np.linspace(0, af, xpoints.shape[0])
    tt = np.linspace(0, deltaT, vpoints.shape[0])
    tck1 = interpolate.splrep(aa, xpoints)
    tck2 = interpolate.splrep(aa, ypoints)
    tckV = interpolate.splrep(tt, vpoints)

    t_total = np.linspace(0, T, num_timesteps + 1)
    timesteps = (t_total[:-1] + t_total[1:]) / 2

    def splfun(a):
        x = interpolate.splev(a, tck1)
        y = interpolate.splev(a, tck2)
        return x, y

    def splder(a):
        dx = interpolate.splev(a, tck1, der=1)
        dy = interpolate.splev(a, tck2, der=1)
        return dx, dy

    def lenfun(a):
        dx, dy = splder(a)
        dl = np.sqrt(dx ** 2 + dy ** 2)
        return dl

    def len_int(a):
        length = integrate.quad(lenfun, 0, a)[0]
        return length

    total_length = len_int(af)

    ####2. max_speed calculation
    # max_speed = (total_length/
    #             (T + -1/((1-delta)*(1-delta)*T*T*3)*(T-deltaT)*(T-deltaT)*(T-deltaT)))

    def V(t):
        # speed = np.zeros(t.shape)

        # speed[t<=deltaT] = interpolate.splev(t[t<=deltaT],tck1)
        if t <= deltaT:
            speed = interpolate.splev(t, tckV)
        # c = Vpoints[-1]/((T-deltaT)**2)
        # speed[t>deltaT] = -c*(t[t>deltaT]-deltaT)+Vpoints[-1]
        else:
            # slope = interpolate.splev(deltaT,tck1,der=1)
            c = vpoints[-1] / ((T - deltaT) ** 2)
            speed = -c * ((t - deltaT) ** 2) + vpoints[-1]
        return speed

    def l(t):
        # distance = np.zeros(t.shape)
        # for i in range(t.shape):
        #    distance(i)
        distance = integrate.quad(V, 0, t)[0]
        # distance = np.zeros(t.shape)
        # distance[t<=deltaT] = max_speed*t[t<=deltaT]
        # distance[t>deltaT] = (max_speed*t[t>deltaT] + -(max_speed)/((1-delta)*(1-delta)*T*T*3)*
        #                      (t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT)*(t[t>deltaT]-deltaT))
        return distance

    lt_final = l(T)
    V_mul = total_length / lt_final

    def V2(t):
        speed = V_mul * V(t)
        return speed

    def V2mat(t):
        speed = np.zeros(t.shape)

        speed[t <= deltaT] = interpolate.splev(t[t <= deltaT], tckV)
        c = vpoints[-1] / ((T - deltaT) ** 2)
        speed[t > deltaT] = -c * ((t[t > deltaT] - deltaT) ** 2) + vpoints[-1]
        return V_mul * speed  # V_mul*V(t)

    def l2(t):
        distance = integrate.quad(V2, 0, t)[0]
        return distance

    # print(timesteps)
    # print(V2mat(timesteps))
    # print(V2(T))

    #### 3. matching t vs x,y
    def xy_at_t(t):
        # 3-1. t vs l(t)
        lt = l2(t)

        # print(V2(t))
        # print(lt)
        # 3-2. l(t) vs a
        def len_res(a):
            return len_int(a) - lt

        a = fsolve(len_res, (af / 2))
        x, y = splfun(a)
        return a, x, y

    xx = np.zeros([num_timesteps])
    yy = np.zeros([num_timesteps])
    aa = np.zeros([num_timesteps])
    for i in range(num_timesteps):
        print('\r(spline) current timestep = {:3}/{:3}'.format(i + 1, num_timesteps), end=' ')
        aa[i], xx[i], yy[i] = xy_at_t(timesteps[i])
    print('\r(spline) Calcualtion finished.', end=' ')

    #### 4. asigning speed to the spline
    dx, dy = splder(aa)
    dx_normalized = dx / np.sqrt(dx ** 2 + dy ** 2)
    dy_normalized = dy / np.sqrt(dx ** 2 + dy ** 2)
    dx_f = dx_normalized * V2mat(timesteps)
    dy_f = dy_normalized * V2mat(timesteps)

    #### 5. xtraj and x_dot
    qtraj = torch.zeros(num_timesteps, 2)
    q_dot = torch.zeros(num_timesteps, 2)

    qtraj[:, 0] = torch.from_numpy(xx)
    qtraj[:, 1] = torch.from_numpy(yy)
    q_dot[:, 0] = torch.from_numpy(dx_f)
    q_dot[:, 1] = torch.from_numpy(dy_f)
    qtraj = qtraj.detach()
    q_dot = q_dot.detach()
    # Xtraj = q_to_x(qtraj)
    return qtraj, q_dot  # , Xtraj


def q_line_traj(q1, q2, T, delta, num_timesteps):
    timesteps = torch.linspace(0, T, num_timesteps + 1)

    qinit = torch.tensor([q1[0], q2[0]])
    qfinal = torch.tensor([q1[1], q2[1]])
    q_dot = torch.zeros(num_timesteps, 2)
    qtraj = torch.zeros(num_timesteps, 2)
    deltaT = delta * T
    direction = (qfinal - qinit) / torch.norm((qfinal - qinit), 2)
    max_speed = (torch.norm((qfinal - qinit), 2) /
                 (T + -1 / ((1 - delta) * (1 - delta) * T * T * 3) * (T - deltaT) * (T - deltaT) * (T - deltaT)))

    def V(t):
        if t <= deltaT:
            speed = max_speed
        else:
            speed = -(max_speed) / ((1 - delta) * (1 - delta) * T * T) * (t - deltaT) * (t - deltaT) + max_speed
        return speed

    def s(t):
        if t <= deltaT:
            distance = max_speed * t
        else:
            distance = (max_speed * t + -(max_speed) / ((1 - delta) * (1 - delta) * T * T * 3) *
                        (t - deltaT) * (t - deltaT) * (t - deltaT))
        return distance

    for i in range(num_timesteps):
        t = (timesteps[i] + timesteps[i + 1]) / 2
        vel_current = V(t)
        dist_current = s(t)
        q_dot[i, 0] = vel_current * direction[0]
        q_dot[i, 1] = vel_current * direction[1]

        qtraj[i, 0] = qinit[0] + dist_current * direction[0]
        qtraj[i, 1] = qinit[1] + dist_current * direction[1]

    return qtraj, q_dot


def traj_plot(qtraj, qdot, *args):  # , q1, q2):
    plt.rcParams["figure.figsize"] = (10, 5)
    plt.axis([0, 2 * math.pi, 0, math.pi])
    qtrajj = qtraj.cpu().detach().numpy()
    qdott = qdot.cpu().detach().numpy()
    # plt.quiver(qtrajj[:,1],qtrajj[:,0],qdott[:,1],qdott[:,0])
    plt.plot(qtrajj[:, 1], qtrajj[:, 0])
    if len(args) == 2:
        plt.plot(args[1][0], args[0][0], 'bo')
        plt.plot(args[1][1], args[0][1], 'cs')

    return


def line_traj_storage(traj_number):
    if traj_number == 1:
        q1 = [0.2, 1.3]
        q2 = [1.2, 2.5]

    elif traj_number == 2:
        q1 = [1.2, 2.3]
        q2 = [1.0, 3.0]

    elif traj_number == 3:
        q1 = [0.2, 1.3]
        q2 = [1.2, 2.0]

    elif traj_number == 4:
        q1 = [0.1, 1.5]
        q2 = [0.1, 1.5]

    else:
        print('Wrong traj_number!')
        assert (False)

    return q1, q2


def spline_traj_storage(traj_number):
    if traj_number == 1:
        xpoints = np.array([1.25, 0.45, 0.25, 0.45, 0.75, 1.15]) + 0.3
        ypoints = np.array([5.9 - 0.5, 6.0 - 0.5, 5.5 - 0.5, 5.0 - 0.5, 4.7 - 0.5, 4.5 - 0.5])
        Vpoints = np.array([1.5, 1.7, 1.9, 2.3, 1.5])
    elif traj_number == 2:
        xpoints = np.array([5.9, 6.0, 5.5, 5.0, 4.7, 4.5]) - 3.5
        ypoints = np.array([1.25, 0.65, 0.45, 0.65, 0.85, 1.15]) + 0.3
        Vpoints = np.array([1.5, 1.7, 1.9, 2.3, 1.5])

    else:
        print('Wrong traj_number!')
        assert (False)

    return xpoints, ypoints, Vpoints


def sphere_traj(traj_type, *args, **kwargs):
    if traj_type == 'line':
        print("traj_type = " + traj_type)
        traj_number = args[0]
        print("traj_number = " + str(traj_number))
        print(
            "Note that input must be in the form of (traj_type, traj_number, t_final, delta, num_timesteps, **kwargs)")

        t_final = args[1]
        delta = args[2]
        num_timesteps = args[3]
        q1, q2 = line_traj_storage(traj_number)
        qtraj, q_dot = q_line_traj(q1, q2, t_final, delta, num_timesteps)

    elif traj_type == 'spline':
        print("traj_type = " + traj_type)
        traj_number = args[0]
        t_final = args[1]
        delta = args[2]
        num_timesteps = args[3]
        print("traj_number = " + str(traj_number))
        print(
            "Note that input must be in the form of (traj_type, traj_number, t_final, delta, num_timesteps, **kwargs)")

        q1, q2, Vpoints = spline_traj_storage(traj_number)
        qtraj, q_dot = Sphere_spline(q1, q2, Vpoints, t_final, delta, num_timesteps)

    elif traj_type == 'custom_line':
        print("traj_type = " + traj_type)
        print("Note that input must be in the form of (traj_type, q1, q2, t_final, delta, num_timesteps)")
        q1 = args[0]
        q2 = args[1]
        t_final = args[2]
        delta = args[3]
        num_timesteps = args[4]
        qtraj, q_dot = q_line_traj(q1, q2, t_final, delta, num_timesteps)

    elif traj_type == 'custom_spline':
        print("traj_type = " + traj_type)
        print(
            "Note that input must be in the form of (traj_type, xpoints, ypoints, Vpoints, t_final, delta, num_timesteps)")

        q1 = args[0]
        q2 = args[1]
        Vpoints = args[2]
        t_final = args[3]
        delta = args[4]
        num_timesteps = args[5]
        qtraj, q_dot = Sphere_spline(q1, q2, Vpoints, t_final, delta, num_timesteps)

    else:
        print('traj_type must be one of: \'line\', \'spline\', \'custom_line\', \'custom_spline\' ')
        assert (False)

    if len(kwargs) != 0:
        if kwargs['plot'] == True:
            #             traj_plot(qtraj, q_dot, q1, q2)
            plot_f_sphere(qtraj, q_dot, q1, q2)

    thetaphi_final = torch.tensor([q1[-1], q2[-1]], dtype=dtype)
    Xstable = q_to_x(thetaphi_final)

    return qtraj, q_dot, Xstable


def get_SO3_from_w_theta(w, theta):
    w = torch.tensor(w, dtype=torch.float).unsqueeze(0)
    wnorm = torch.norm(w)
    what = w / wnorm
    return lie.exp_so3(what * theta).squeeze(0).numpy()

def get_geodesic_dist(x1, x2):
    eps = 1e-10
    x1x2 = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    theta = torch.arccos(torch.clip(x1x2, min=-1+eps, max=1-eps))  # (n, 1)
    return theta
    
def geodesic_sphere(x1, x2, t):
    if x1.shape[1] == 2:
        x1 = q_to_x(x1)
        x2 = q_to_x(x2)
    eps = 1e-7
    x1x2 = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    theta = torch.arccos(x1x2)  # (n, 1)
    term1 = torch.cos(theta * t) * x1
    
    temp1 = x2 - (x1x2 * x1)
    temp1_norm = torch.clamp(torch.norm(temp1, dim=1).unsqueeze(-1), min=eps)   
    temp2 = temp1 / temp1_norm
    term2 = torch.sin(theta * t) * temp2

    return term1 + term2

def vel_geo_sphere(x1, x2, t):
    # Might not be used..
    eps = 1e-7
    x1x2 = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    theta = torch.arccos(x1x2)  # (n, 1)
    term1 = -theta * torch.sin(theta * t) * x1
    
    temp1 = x2 - (x1x2 * x1)
    temp1_norm = torch.clamp(torch.norm(temp1, dim=1).unsqueeze(-1), min=eps)
    temp2 = temp1 / temp1_norm
    term2 = theta * torch.cos(theta * t) * temp2
    
    return term1 + term2

def vel_geo_0_sphere(x1, x2):
    eps = 1e-10
    x1x2 = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    x1x2 = torch.clip(x1x2, min= -1 + eps, max = 1 - eps)
    theta = torch.arccos(x1x2)  # (n, 1)
    temp1 = x2 - (x1x2 * x1)
    temp1_norm = torch.clamp(torch.norm(temp1, dim=1).unsqueeze(-1), min=eps)
    temp2 = temp1 / temp1_norm
    return theta * temp2

def get_closest_point_sphere(x, traj, index=True):
    x_traj_dist = torch.einsum('ni, mi -> nm', x, traj)
    index_closest = torch.argmax(x_traj_dist, dim=1)
    if index:
        return traj[index_closest], index_closest
    else:
        return traj[index_closest]

def get_distance_sphere(x, traj, index=True):
    x_traj_dist = torch.einsum('ni, mi -> nm', x, traj)
    return x_traj_dist 
    

def parallel_transport(x1, x2, V):
    eps = 1e-10
    x1x2_cross = torch.cross(x1, x2)
    x1x2_dot = torch.einsum('ni, ni -> n', x1, x2).unsqueeze(-1)
    x1x2_dot = torch.clip(x1x2_dot, min=-1 + eps, max= 1 - eps)
    theta = torch.arccos(x1x2_dot)
    w = x1x2_cross * theta
    Rot = lie.exp_so3(w)
    return torch.einsum('nij, nj -> ni', Rot, V)

def cvf_sphere(xsample, eta, xtraj, xdottraj):
    eps = 1e-6
    if xsample.shape[-1] == 2:
        xsample = q_to_x(xsample)
    xtraj_closest, index_closest = get_closest_point_sphere(xsample, xtraj, index=True)
    xdottraj_closest = xdottraj[index_closest]
    if eta < 1e30:
        V1 = parallel_transport(xtraj_closest, xsample, xdottraj_closest)
    if eta > 0:
        vel = vel_geo_0_sphere(xsample, xtraj_closest)
    # Only parallel transport case (eta = 0)
    if eta == 0: 
        return V1
    # Only contraction case
    if eta > 1e30:
        return vel
    
    if type(eta) != torch.Tensor:
        eta = torch.zeros(len(xsample), 1).to(xtraj) + eta
        # beta = torch.zeros(len(xsample), 1).to(xtraj) + beta
    elif len(eta.shape) == 1:
        eta = eta.unsqueeze(1).to(xtraj)
        # beta = beta.unsqueeze(1)
    # print(f'xsample.sum() = {xsample.sum()}')
    # print(f'(V1.sum(), vel.sum()) {V1.sum()}, {vel.sum()}')
    return V1 + eta * vel

def exp_sphere(x, v):
    # if input is quaternion (n, 2) convert to (n, 3)
    if x.shape[-1] == 2:
        x = q_to_x(x)  # (n, 3)

    # match dimensions for broadcasting
    if len(v.shape) == 3 and len(x.shape) == 2:
        x = x.unsqueeze(-1)  # (n, 1, 3)

    vnorm = torch.norm(v, dim=-1, keepdim=True)  # (n, 1) or (n, s, 1)
    mask = vnorm < 1e-12  # boolean mask


    v_div = torch.where(mask, torch.zeros_like(v), v / vnorm)
    par = torch.cos(vnorm) * x
    vert = torch.sin(vnorm) * v_div
    # x_new = torch.cos(vnorm) * x + torch.sin(vnorm) * v_div
    x_new = par + vert

    x_new = torch.where(mask, x, x_new)

    return x_new


def tangent_gaussian_sampling(q, std, sample_size):
    if q.shape[-1] == 2:
        x = q_to_x(q)
    elif q.shape[-1] == 3:
        x = q
    if len(x.shape) == 1:
        squeezed = True
        x = x.unsqueeze(0)
    nx = len(x)
    x = x.unsqueeze(1) # n, 1, 3
    vsample_3d = torch.empty(nx, sample_size, 3).to(q).normal_(mean=0, std=std)
    aligned_norm = (torch.sum(x * vsample_3d, dim=-1)).unsqueeze(-1) # n, s, 1
    vsample_tangent = vsample_3d - (x * aligned_norm) # n, s, 3
    xsample = exp_sphere(x, vsample_tangent)
    if squeezed:
        xsample = xsample.squeeze(0)
    return xsample

def cvf_gaussian_sphere(xsample, eta, xtraj, xdottraj, std=0.1, sample_size=100, xsample_sample=None):
    if xsample_sample is None:
        xsample_sample = tangent_gaussian_sampling(xsample, std, sample_size)
    list_Vsample_sample_vec = []
    if type(eta) != torch.Tensor:
        for x_temp in xsample_sample.view(-1, 3).split(10000):
            Vsample_sample_vec = cvf_sphere(x_temp, eta, xtraj, xdottraj)
            list_Vsample_sample_vec.append(Vsample_sample_vec)
    else:
        eta_long = eta.reshape(-1, 1).repeat(1, sample_size).reshape(-1).to(xtraj)
        for (x_temp, eta_temp) in zip(xsample_sample.view(-1, 3).split(10000), eta_long.split(10000)):
            Vsample_sample_vec = cvf_sphere(x_temp, eta_temp, xtraj, xdottraj)
            list_Vsample_sample_vec.append(Vsample_sample_vec)
    Vsample_sample_vec = torch.cat(list_Vsample_sample_vec, dim=0)        
    V_samples_sample = Vsample_sample_vec.view(len(xsample), sample_size, 3)
    V_sample_mean = V_samples_sample.mean(dim=1)
    return V_sample_mean



## fixed version
def get_xddot(model, q):
    x = q_to_x(q)
    qdot = model(x)
    
    def model_for_jvp(q):
        x = q_to_x(q)
        out = model(x)
        return out
    
    qddot = jvp(model_for_jvp, q, qdot)[1]
    
    Jdot = get_jacobian_dot(q, qdot)
    J = get_jacobian(q)
    xddot = (Jdot @ q.unsqueeze(2) + J @ qddot.unsqueeze(2)).squeeze(2)
    return xddot

## original version
# def get_xddot(model, q):
#     qdot = model(q)
#     qddot = jvp(model, q, qdot)
#     Jdot = get_jacobian_dot(q, qdot)
#     J = get_jacobian(q)
#     print('qddot :', len(qddot))
#     print('qddot :', qddot[0].shape)
#     xddot = (Jdot @ q.unsqueeze(2) + J @ qddot[0].unsqueeze(2)).squeeze(2)
#     return xddot

def get_qddot(model, q):
    qdot = model(q)
    qddot = jvp(model, q, qdot)
    return qddot


###################################################################################################
#################################### functions for LASA sphere ####################################

import numpy as np

def q_to_x_numpy(thetaphi):
    """
    Convert spherical coordinates (theta, phi) to Cartesian coordinates on S^2.

    Args:
        thetaphi: shape (2,) for a single point [theta, phi],
                  or (..., 2) for a batch of points.

    Returns:
        x: Cartesian coordinates, shape (3,) for single input,
           or (..., 3) for batch input.
    """
    thetaphi = np.asarray(thetaphi)

    if thetaphi.shape == (2,):
        theta = thetaphi[0]
        phi = thetaphi[1]
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        x = np.array([st * cp, st * sp, ct], dtype=np.float64)
    else:
        theta = thetaphi[..., 0][..., np.newaxis]
        phi = thetaphi[..., 1][..., np.newaxis]
        ct = np.cos(theta)
        st = np.sin(theta)
        cp = np.cos(phi)
        sp = np.sin(phi)
        x = np.concatenate([st * cp, st * sp, ct], axis=-1).astype(np.float64)

    return x



def xtraj_to_xdottraj(xtraj, dt):
    qtraj = x_to_q(xtraj)
    qdottraj = (qtraj[1:,:] - qtraj[:-1,:])/dt
    qdottraj = torch.cat([qdottraj, torch.zeros([1,qdottraj.shape[1]])], dim=0)
    xdottraj = qdot_to_xdot(qdottraj, qtraj)
    return xdottraj

def generate_trajectory_sphere(x_input, model, eta=1, time_step=110, dt=0.03, model_type=None):
    # input : [batch, 3 or 2]
    if x_input.shape[-1] == 2:
        x_input = q_to_x(x_input)
    
    x_now  = x_input
    xtraj_list = [x_now.unsqueeze(1)]
    qtraj_list = [x_to_q(x_now).unsqueeze(1)]
    for j in range(time_step):
        q_dot = model(x_now.to(x_input), eta=eta)
        if model_type in ['lieflow', 'seds', 'sddm', 'rsds', 'bc-deepovec']:
            max_speed = 1
            q_dot = q_dot / torch.where(q_dot.norm(dim=1)>max_speed, q_dot.norm(dim=1)/max_speed, 1.).unsqueeze(1).repeat(1, q_dot.shape[1])
        q_now = x_to_q(x_now)
        q_next = q_now + dt * q_dot     # [batch, 2]
        x_now = q_to_x(q_next).to(torch.float) # [batch, 3]
        xtraj_list.append(x_now.unsqueeze(1).detach())
        qtraj_list.append(q_next.unsqueeze(1).detach())
    xtraj = torch.cat(xtraj_list, dim=1)
    qtraj = torch.cat(qtraj_list, dim=1)
    
    return xtraj, qtraj


def vec_3dim_to_2dim(x, vec):
    Jac = get_jacobian(x)
    return torch.einsum('nij, ni -> nj', Jac, vec)





# ============================ 2nd order ============================

        


