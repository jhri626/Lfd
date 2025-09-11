import torch
from agent.utils.S2_functions import xdot_to_qdot, q_to_x

# =============================
# Math / geometry helpers
# =============================

def _normalize(x: torch.Tensor, dim: int = -1, eps: float = 1e-12) -> torch.Tensor:
    """L2-normalize along a dimension with numerical safety."""
    n = torch.linalg.norm(x, dim=dim, keepdim=True).clamp_min(eps)
    return x / n

def alpha_for_target_coverage_torch(r_max: torch.Tensor, coverage: float = 0.8):
    """
    Compute scale alpha to achieve desired sphere area coverage (fraction).
    coverage in (0,1).

    Derivation:
      Let Z* be the latitude (z=Z*) bounding the used area on S^2.
      cap area fraction f = (1 - Z*) / 2,  coverage = 1 - f  =>  Z* = 2*coverage - 1.
      Inverse stereographic radius relation: R = sqrt((1+Z*)/(1-Z*)).
      We want (r_max / alpha) = R  =>  alpha = r_max / sqrt((1+Z*)/(1-Z*)).
    """
    if not (0.0 < coverage < 1.0):
        raise ValueError("coverage must be in (0,1).")
    Z_star = 2.0 * coverage - 1.0
    if not (-1.0 < Z_star < 1.0):
        raise ValueError("Computed Z* is out of (-1,1).")
    s = (1.0 + Z_star) / (1.0 - Z_star)

    # r_max can be scalar or (B,)
    r_max = torch.as_tensor(r_max)
    alpha = r_max / torch.sqrt(torch.as_tensor(s, dtype=r_max.dtype, device=r_max.device))
    Z_star_t = torch.as_tensor(Z_star, dtype=r_max.dtype, device=r_max.device)
    return alpha, Z_star_t

def inverse_stereographic_projection_batch_torch(
    xy: torch.Tensor,
    alpha: torch.Tensor,
    center=(0.0, 0.0),
    R: torch.Tensor | None = None
) -> torch.Tensor:
    """
    Batched inverse stereographic projection onto S^2.

    Args:
        xy: (B, T, 2)
        alpha: (B,) or scalar tensor > 0
        center: (cx, cy) float pair
        R: optional (B,3,3) or (3,3) rotation(s), applied after mapping

    Returns:
        pts: (B, T, 3) unit vectors on S^2
    """
    if xy.ndim == 2 and xy.shape[1] == 2:  # upgrade to batch
        xy = xy.unsqueeze(0)

    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("xy must be (B,T,2) or (T,2).")

    B, T, _ = xy.shape
    device, dtype = xy.device, xy.dtype

    alpha = torch.as_tensor(alpha, dtype=dtype, device=device)
    if alpha.ndim == 0:
        alpha = alpha.repeat(B)
    if alpha.shape != (B,):
        raise ValueError("alpha must be scalar or (B,)")

    cx, cy = float(center[0]), float(center[1])
    u = (xy[..., 0] - cx) / alpha[:, None]
    v = (xy[..., 1] - cy) / alpha[:, None]
    r2 = u * u + v * v
    denom = r2 + 1.0

    X = 2.0 * u / denom
    Y = 2.0 * v / denom
    Z = (r2 - 1.0) / denom
    pts = torch.stack([X, Y, Z], dim=-1)

    # Optional rotation(s): row-vector application -> pts @ R^T
    if R is not None:
        if R.ndim == 2:
            if R.shape != (3, 3):
                raise ValueError("R must be (3,3) or (B,3,3).")
            pts = pts @ R.t()
        elif R.ndim == 3:
            if R.shape != (B, 3, 3):
                raise ValueError("R must be (3,3) or (B,3,3).")
            pts = torch.einsum("btj,bij->bti", pts, R.transpose(1, 2))
        else:
            raise ValueError("R must be (3,3) or (B,3,3).")

    # Ensure unit-length
    pts = _normalize(pts, dim=-1)
    return pts  # (B,T,3)

def hat_torch(v: torch.Tensor) -> torch.Tensor:
    """Skew-symmetric matrix (hat operator): (...,3) -> (...,3,3)."""
    v = v.view(*v.shape[:-1], 3)
    zeros = torch.zeros_like(v[..., 0])
    vx = torch.stack([
        torch.stack([ zeros,      -v[..., 2],  v[..., 1]], dim=-1),
        torch.stack([ v[..., 2],  zeros,      -v[..., 0]], dim=-1),
        torch.stack([-v[..., 1],  v[..., 0],  zeros     ], dim=-1),
    ], dim=-2)
    return vx

def axis_angle_to_SO3_torch(axis: torch.Tensor, theta: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """
    Rodrigues formula for rotation about 'axis' by 'theta'.
    axis: (...,3), theta: (...) -> (...,3,3)
    """
    axis = _normalize(axis, dim=-1, eps=eps)
    theta = theta.unsqueeze(-1) if theta.ndim == axis.ndim - 1 else theta
    omega = axis * theta
    omega_hat = hat_torch(omega)
    th = torch.linalg.norm(omega, dim=-1, keepdim=True).clamp_min(eps)

    A = torch.sin(th) / th
    B = (1.0 - torch.cos(th)) / (th ** 2)

    I = torch.eye(3, dtype=axis.dtype, device=axis.device).expand(*axis.shape[:-1], 3, 3)
    R = I + A[..., None] * omega_hat + B[..., None] * (omega_hat @ omega_hat)
    return R

def align_a_to_b_single(a: torch.Tensor, b: torch.Tensor, eps: float = 1e-9) -> torch.Tensor:
    """
    Rotation R s.t. R @ a_hat ? b_hat. a,b are (3,) tensors. Robust for a?��b.
    Returns (3,3).
    """
    a = _normalize(a, dim=-1, eps=eps)
    b = _normalize(b, dim=-1, eps=eps)
    dot = torch.dot(a, b).clamp(-1.0, 1.0)

    if torch.isclose(dot, torch.tensor(1.0, dtype=a.dtype, device=a.device), atol=1e-7):
        return torch.eye(3, dtype=a.dtype, device=a.device)

    if torch.isclose(dot, torch.tensor(-1.0, dtype=a.dtype, device=a.device), atol=1e-7):
        aux = torch.tensor([1.0, 0.0, 0.0], dtype=a.dtype, device=a.device)
        if torch.abs(torch.dot(a, aux)) > 0.9:
            aux = torch.tensor([0.0, 1.0, 0.0], dtype=a.dtype, device=a.device)
        axis = torch.cross(a, aux)
        axis = _normalize(axis, dim=-1, eps=eps)
        return axis_angle_to_SO3_torch(axis, torch.tensor(torch.pi, dtype=a.dtype, device=a.device))

    v = torch.cross(a, b)
    s = torch.linalg.norm(v).clamp_min(eps)
    vx = hat_torch(v)
    I = torch.eye(3, dtype=a.dtype, device=a.device)
    R = I + vx + (vx @ vx) * ((1.0 - dot) / (s ** 2))
    return R

def representative_direction_mean_or_pca(pts: torch.Tensor, mean_tol: float = 1e-6) -> torch.Tensor:
    """
    Representative direction for a single trajectory on S^2.
    Prefer mean; fallback to PCA principal component when mean is ~0.
    pts: (T,3) -> returns (3,)
    """
    m = pts.mean(dim=0)
    nm = torch.linalg.norm(m)
    if nm > mean_tol:
        return m / nm
    X = pts - pts.mean(dim=0, keepdim=True)
    C = (X.t() @ X) / max(pts.shape[0] - 1, 1)
    vals, vecs = torch.linalg.eigh(C)
    principal = vecs[:, torch.argmax(vals)]
    return _normalize(principal, dim=0)

# =============================
# x �� S^2  ->  q = (theta, phi)
# =============================

def x_to_q_torch(x: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    """
    Convert Cartesian points on S^2 to spherical coordinates q=(theta, phi).
    theta �� [0, pi], phi �� [0, 2��). Works with (T,3) or (B,T,3).

    This mirrors the logic you shared previously, but vectorized.
    """
    if x.ndim == 1 and x.shape[0] == 3:
        x = x.unsqueeze(0)  # (1,3)

    if x.ndim == 2 and x.shape[1] == 3:
        # (T,3)
        ct = x[:, 2].clamp(-1+eps, 1-eps)
        theta = torch.arccos(ct)
        st = torch.sqrt(torch.clamp(1 - ct * ct, min=eps))
        sp = (x[:, 1] / st).clamp(-1+eps, 1-eps)
        cp = (x[:, 0] / st).clamp(-1+eps, 1-eps)
        phi = torch.arccos(cp)
        phi = torch.where(x[:, 1] >= 0, phi, -phi + 2 * torch.pi)
        return torch.stack([theta, phi], dim=-1)

    elif x.ndim == 3 and x.shape[-1] == 3:
        # (B,T,3)
        ct = x[..., 2].clamp(-1+eps, 1-eps)
        theta = torch.arccos(ct)
        st = torch.sqrt(torch.clamp(1 - ct * ct, min=eps))
        sp = (x[..., 1] / st).clamp(-1+eps, 1-eps)
        cp = (x[..., 0] / st).clamp(-1+eps, 1-eps)
        phi = torch.arccos(cp)
        phi = torch.where(x[..., 1] >= 0, phi, -phi + 2 * torch.pi)
        return torch.stack([theta, phi], dim=-1)

    else:
        raise ValueError("x must be (T,3), (B,T,3), or (3,)")

# =============================
# End-to-end: R^2 curve -> q-trajectory on S^2
# =============================

def map_R2_curve_to_S2_q_torch(
    xy: torch.Tensor,
    *,
    coverage: float = 0.8,
    center: tuple[float, float] = (0.0, 0.0),
    align_front: bool = True,
    front_target: torch.Tensor | None = None,
    R_user: torch.Tensor | None = None,
    return_intermediates: bool = False
):
    """
    Map an arbitrary 2D curve xy (T,2) or batch (B,T,2) to a trajectory on S^2,
    and return spherical coordinates q=(theta, phi).

    Steps:
      1) alpha from desired coverage (per-trajectory)
      2) inverse stereographic projection -> x in S^2
      3) optional front alignment to 'front_target'
      4) optional user rotation R_user
      5) convert x -> q

    Returns:
        q_traj: (T,2) or (B,T,2)
        If return_intermediates:
            dict with keys:
              'x_traj'   : (B,T,3) mapped Cartesian on S^2
              'alpha'    : (B,) alpha values
              'Z_star'   : scalar Z* (same for all batch if same coverage)
              'R_total'  : (B,3,3) total rotations applied (column convention)
    """
    # Ensure batched
    single = False
    if not isinstance(xy, torch.Tensor):
        xy = torch.from_numpy(xy).to(torch.get_default_dtype())

    if xy.ndim == 2 and xy.shape[1] == 2:
        xy = xy.unsqueeze(0)  # (1,T,2)
        single = True
    if xy.ndim != 3 or xy.shape[-1] != 2:
        raise ValueError("xy must be (T,2) or (B,T,2).")

    B, T, _ = xy.shape
    device, dtype = xy.device, xy.dtype
    
    # 1) per-trajectory r_max and alpha
    center_t = torch.as_tensor(center, device=device, dtype=dtype)
    r = torch.linalg.norm(xy - center_t, dim=-1)         # (B,T)
    r_max = r.max(dim=1).values                           # (B,)
    if torch.any(r_max <= 0):
        raise ValueError("Some trajectories have zero radius; cannot define alpha.")
    alpha, Z_star = alpha_for_target_coverage_torch(r_max, coverage=coverage)  # (B,), scalar

    # 2) inverse stereographic projection (no rotation yet)
    x_traj = inverse_stereographic_projection_batch_torch(xy, alpha=alpha, center=center, R=None)  # (B,T,3)

    # 3) optional front alignment (per batch)
    R_total = torch.eye(3, dtype=dtype, device=device).repeat(B, 1, 1)
    if align_front:
        if front_target is None:
            front_target = torch.tensor([0.0, 0.0, 1.0], dtype=dtype, device=device)
        front_target = _normalize(front_target, dim=-1)
        print("front",front_target)
        # compute batch rotations and apply
        R_list = []
        for b in range(B):
            a_dir = representative_direction_mean_or_pca(x_traj[b])  # (3,)
            Rb = align_a_to_b_single(a_dir, front_target)            # (3,3)
            
            R_list.append(Rb)
        R_front = torch.stack(R_list, dim=0)                         # (B,3,3)
        print("R_front",R_front.shape,"xtraj",x_traj.shape)
        x_traj = torch.einsum("btj,bji->bti",R_front, x_traj.transpose(1,2))
        R_total = torch.einsum("bij,bjk->bik", R_front, R_total)

    # 4) optional user rotation(s)
    if R_user is not None:
        if R_user.ndim == 2:
            if R_user.shape != (3, 3):
                raise ValueError("R_user must be (3,3) or (B,3,3).")
            x_traj = x_traj @ R_user.t()
            R_total = torch.einsum("ij,bjk->bik", R_user, R_total)
        elif R_user.ndim == 3:
            if R_user.shape != (B, 3, 3):
                raise ValueError("R_user must be (3,3) or (B,3,3).")
            x_traj = torch.einsum("btj,bji->bti",R_front, x_traj.transpose(1,2))
            R_total = torch.einsum("bij,bjk->bik", R_user, R_total)
        else:
            raise ValueError("R_user must be (3,3) or (B,3,3).")

    # 5) x -> q
    x_traj = x_traj.transpose(1, 2)
    q_traj = x_to_q_torch(x_traj)  # (B,T,2)

    if single:
        q_traj = q_traj.squeeze(0)     # (T,2)
        x_traj = x_traj.squeeze(0)     # (T,3)
        alpha = alpha.squeeze(0)       # ()
        # Z_star identical across batch for uniform coverage
        if return_intermediates:
            return q_traj, {
                "x_traj": x_traj,
                "alpha": alpha,
                "Z_star": Z_star,
                "R_total": R_total.squeeze(0)
            }
        else:
            return q_traj

    if return_intermediates:
        return q_traj, {
            "x_traj": x_traj,
            "alpha": alpha,
            "Z_star": Z_star,
            "R_total": R_total
        }
    return q_traj


def qtraj_to_qdot_forward_consistent(q_traj: torch.Tensor, dt: float) -> torch.Tensor:
    """
    Compute q_dot from q_traj using FORWARD difference on Cartesian x on S^2,
    then map xdot -> qdot via your xdot_to_qdot (least-squares).
    This is consistent with manifold integration using exp_sphere.

    Args:
        q_traj : (T,2) or (B,T,2)  with q = [theta, phi]
        dt     : scalar timestep

    Returns:
        q_dot  : same shape as q_traj
                 forward-diff at t=0..T-2, and zero at the last step (like your Euclidean code)
    """
    # Standardize shape to (B,T,2)
    single = False
    if q_traj.ndim == 2 and q_traj.shape[1] == 2:
        q = q_traj.unsqueeze(0)
        single = True
    elif q_traj.ndim == 3 and q_traj.shape[-1] == 2:
        q = q_traj
    else:
        raise ValueError("q_traj must be (T,2) or (B,T,2).")

    B, T, _ = q.shape
    device, dtype = q.device, q.dtype

    # 1) q -> x on S^2 (uses your function)
    x = q_to_x(q)  # (B,T,3), unit-norm

    # 2) xdot via FORWARD difference, last step = 0 (Euclidean-style)
    xdot = torch.zeros(B, T, 3, dtype=dtype, device=device)
    if T >= 2:
        xdot[:, :-1, :] = (x[:, 1:, :] - x[:, :-1, :]) / dt
        # xdot[:, -1, :] = 0  # already zero-initialized

    # 3) Ensure tangency: project xdot to T_x S^2
    xdot = xdot - (xdot * x).sum(dim=-1, keepdim=True) * x

    # 4) Map xdot -> qdot using your least-squares routine
    xdot_flat = xdot.reshape(-1, 3)
    q_flat    = q.reshape(-1, 2)
    qdot_flat = xdot_to_qdot(xdot_flat, q_flat)  # (B*T,2)
    q_dot     = qdot_flat.view(B, T, 2)

    # 5) Return with original shape
    if single:
        return q_dot.squeeze(0)  # (T,2)
    return q_dot                 # (B,T,2)
