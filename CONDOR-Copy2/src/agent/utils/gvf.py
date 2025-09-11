from agent.utils.distance import riemann_anisotropic_distance_S2
from agent.utils.S2_functions import *

def cvf_sphere_2nd_order(xsample, eta, xtraj, xdottraj, dist_radius):
    
    eps = 1e-6
    check = False
    batch_list=[]
    
    B = xsample.shape[0]
    T = xtraj.shape[1]
    D = xtraj.shape[2]
    
    xsample_pos = xsample[:, :2].float()
    xtraj = xtraj.squeeze(0)
    xdottraj = xdottraj.squeeze(0)
    
    xsample_pos = q_to_x(xsample_pos)
    xtraj = q_to_x(xtraj)
    xdottraj = qdot_to_xdot(xdottraj, xtraj)
    
    # verterized distance
    distance_pos = get_distance_sphere(xsample_pos, xtraj, index=False)
    distance_pos = torch.ones_like(distance_pos) - distance_pos
    
    
    top_n_argmin = []
    for b in range(B):
        idx = torch.nonzero(distance_pos[b] < dist_radius[b], as_tuple=False).squeeze(1)
        
        if idx.shape[0] == 0:
            iter = 0
            new_idx = np.array([])
            batch_list.append(b)
            check = True
            while new_idx.size == 0: 
                new_idx_tuple = np.where(distance_pos[b].cpu().numpy() < np.power(1.01, iter) * dist_radius[b].cpu().numpy())
                new_idx = new_idx_tuple[0]
                iter += 1
            idx = torch.from_numpy(new_idx).to(distance_pos.device)
            # print(idx)
        
        top_n_argmin.append(idx)
    
    xsample_vel = qdot_to_xdot(xsample[:, 2:].float(), xsample_pos)
    xsample_expanded = torch.cat([xsample_pos, xsample_vel], dim=-1)
    
    
    index_closest = torch.empty(B, dtype=torch.int32, device=xsample.device)
    
    
    candidate_counts = [len(idx) for idx in top_n_argmin]
    unique_counts = list(set(candidate_counts))
    
    for count in unique_counts:
    
        batch_indices = [i for i, c in enumerate(candidate_counts) if c == count]
        
        if len(batch_indices) == 1:
    
            b = batch_indices[0]
            idx = top_n_argmin[b]
            
            cand_pos = xtraj[idx, :3]
            cand_vel = xdottraj[idx, :3]
            full_state = torch.cat([cand_pos, cand_vel], dim=1).unsqueeze(0)  # (1, K, 6)
            
            sample_state = xsample_expanded[b:b+1].expand(1, count, 6).contiguous()
            
            dist_bt, _, _ = riemann_anisotropic_distance_S2(sample_state, full_state)
            k_best = torch.argmin(dist_bt, dim=1).item()
            index_closest[b] = idx[k_best]
        
        else:
            
            batch_tensor = torch.tensor(batch_indices, device=xsample.device)
            group_size = len(batch_indices)
            
            
            all_cand_pos = torch.stack([xtraj[top_n_argmin[b], :3] for b in batch_indices])  # (group_size, count, 3)
            all_cand_vel = torch.stack([xdottraj[top_n_argmin[b], :3] for b in batch_indices])  # (group_size, count, 3)
            all_full_states = torch.cat([all_cand_pos, all_cand_vel], dim=-1)  # (group_size, count, 6)
            
            
            all_sample_states = xsample_expanded[batch_indices].unsqueeze(1).expand(group_size, count, 6).contiguous()
            
            
            dist_bt, _, _ = riemann_anisotropic_distance_S2(all_sample_states, all_full_states)  # (group_size, count)
            
            
            k_best_group = torch.argmin(dist_bt, dim=1)  # (group_size,)
            
            for i, b in enumerate(batch_indices):
                idx = top_n_argmin[b]
                index_closest[b] = idx[k_best_group[i]]
    
    
    xtraj_closest = xtraj[index_closest]
    xdottraj_closest = xdottraj[index_closest]
    
    if eta < 1e30:
        V1 = parallel_transport(xtraj_closest, xsample_pos, xdottraj_closest)
    if eta > 0:
        vel = vel_geo_0_sphere(xsample_pos, xtraj_closest)
    
    if eta == 0: 
        return V1
    if eta > 1e30:
        return vel
    
    if type(eta) != torch.Tensor:
        eta = torch.zeros(len(xsample), 1).to(xtraj) + eta
    elif len(eta.shape) == 1:
        eta = eta.unsqueeze(1).to(xtraj)
    
    dist = torch.einsum('bd, bd -> b', xsample_pos, xtraj_closest)
    dist = torch.ones_like(dist) - dist
    
    # if check:
    #     print(batch_list)

    return V1 + eta * vel, dist , check, batch_list[0] if batch_list else None