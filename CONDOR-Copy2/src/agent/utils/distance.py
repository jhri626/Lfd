import numpy as np
import torch
from agent.utils.S2_functions import parallel_transport
# from agent.utils.SE3_functions import invSE3, largeAdjoint, logSO3, Tdot_to_Vb, screw_bracket
# from utils.SE3_functions import invSE3, largeAdjoint, logSO3, Tdot_to_Vb, screw_bracket
import time


def angle_between(u, v, degrees=False):
    """
    u, v: np.ndarray of shape (B, T, 2)
    degrees: if True, return angles in degrees
    returns: np.ndarray of shape (B, T, 1)
    """
    # Dot product over last axis (B, T)
    dot = np.sum(u * v, axis=2)

    # Norms (B, T)
    norm_u = np.linalg.norm(u, axis=2)
    norm_v = np.linalg.norm(v, axis=2)

    # Avoid division by zero
    cos_theta = np.clip(dot / (norm_u * norm_v + 1e-8), -1.0, 1.0)

    # Angle in radians
    angle = np.arccos(cos_theta)  # shape: (B, T)

    if degrees:
        angle = np.degrees(angle)

    # Expand to (B, T, 1)
    return angle

def _build_A(v_ref, beta_mag=1.0, alpha_dir=10.0, sigma=0.5, eps=1e-12, tau=1e-6):
    """
    Construct a 2x2 positive-definite, symmetric anisotropic weight matrix A(v_ref).

    Design goals:
    - Keep a quadratic (bilinear) form for velocities: <dv, A dv>.
    - Penalize perpendicular differences (to v_ref) more than parallel differences.
    - Smoothly turn off the extra perpendicular penalty when ||v_ref|| is near zero.

    Parameters
    ----------
    v_ref : (..., 2) array
        Reference velocity vectors defining "direction".
    beta_mag : float
        Baseline weight for velocity magnitude differences.
    alpha_dir : float
        Strength of direction-mismatch penalty.
    sigma : float
        Smoothing scale controlling how fast the extra penalty decays near zero speed.
    eps : float
        Numerical stabilizer to avoid division by zero when normalizing.
    tau : float
        Hard threshold to fully turn off the anisotropic term when ||v_ref|| < tau.

    Returns
    -------
    A : (..., 2, 2) array
        Anisotropic positive-definite matrices.
    """
    v_ref = np.asarray(v_ref)
    s = np.linalg.norm(v_ref, axis=-1, keepdims=True)  # (..., 1)

    # Default: isotropic baseline beta_mag * I
    I = np.eye(2, dtype=v_ref.dtype)
    A_iso = beta_mag * I

    # If speed is too small, return isotropic (no direction information)
    small = (s[..., 0] < tau)
    # Prepare output array
    A = np.broadcast_to(A_iso, (*v_ref.shape[:-1], 2, 2)).copy()

    if np.all(small):
        return A  # all isotropic

    # For non-small speeds, build anisotropic weights
    idx = ~small
    v_sel = v_ref[idx]                         # (N, 2)
    s_sel = np.linalg.norm(v_sel, axis=-1)     # (N,)
    u = v_sel / (s_sel[:, None] + eps)         # (N, 2), unit direction

    # Projections
    P_par = u[:, :, None] @ u[:, None, :]      # (N, 2, 2)
    P_perp = np.eye(2)[None, :, :] - P_par     # (N, 2, 2)

    # Extra perpendicular weight approximating alpha_dir * (1 - cos ��)
    # small-angle match: extra ? (alpha_dir / 2) * ��^2 ? (alpha_dir / (2 s^2)) * ||dv_perp||^2
    # Use a smooth decay near zero: s2 / (s2 + sigma^2)^2
    s2 = s_sel**2
    extra_perp = (alpha_dir / 2.0) * (s2 / (s2 + sigma**2)**2)  # (N,)
    a_par = beta_mag                                            # scalar
    a_perp = beta_mag + extra_perp                              # (N,)

    A_aniso = a_par * P_par + a_perp[:, None, None] * P_perp    # (N, 2, 2)
    A[idx] = A_aniso
    return A


def riemann_anisotropic_distance(seq1, seq2,
                                 lambda_horiz=1.0,
                                 lambda_vert=0.1,   # default 1e-1
                                 alpha_dir=1.0, # default 10.0
                                 beta_mag=1.0,
                                 sigma=0.001,
                                 eps=1e-12,
                                 tau=1e-6,
                                 v_ref_mode="mean"):
    """
    Compute a direction-sensitive *Riemannian* quadratic distance per time step.

    This replaces the non-quadratic (1 - cos) term by a quadratic anisotropic form:
        <dv, A(v_ref) dv> = a_par * ||dv_parallel||^2 + a_perp * ||dv_perp||^2
    with A(v_ref) symmetric positive-definite.

    Parameters
    ----------
    seq1, seq2 : array-like, shape (B, T, 4)
        Each time step holds [x, y, vx, vy].
    lambda_horiz : float
        Weight for position differences.
    lambda_vert : float
        Weight for the velocity-quadratic form.
    alpha_dir : float
        Direction-mismatch strength (controls extra weight on perpendicular differences).
    beta_mag : float
        Baseline magnitude difference weight.
    sigma : float
        Smoothing scale for turning off direction penalty near zero speed.
    eps : float
        Numerical stabilizer.
    tau : float
        Hard threshold below which anisotropy is disabled.
    v_ref_mode : {"mean", "seq1", "seq2"}
        How to choose the reference velocity v_ref.

    Returns
    -------
    distances : ndarray, shape (B, T)
        Per-time-step quadratic distances.
    pos_dist : ndarray, shape (B, T)
        Position part (squared).
    vel_dist : ndarray, shape (B, T)
        Velocity part (quadratic form with A(v_ref)).
    """
    seq1 = np.asarray(seq1); seq2 = np.asarray(seq2)
    B, T, D = seq1.shape
    # assert seq2.shape =/ (B, T, D)
    # assert D == 4

    # Split positions and velocities
    pos1 = seq1[:, :, :2]   # (B, T, 2)
    pos2 = seq2[:, :, :2]   # (B, T, 2)
    vel1 = seq1[:, :, 2:]   # (B, T, 2)
    vel2 = seq2[:, :, 2:]   # (B, T, 2)

    # Position term (squared Euclidean)
    dpos = pos1 - pos2
    pos_dist = np.sum(dpos**2, axis=2)  # (B, T)

    # Velocity difference
    dv = vel1 - vel2  # (B, T, 2)

    # Choose reference velocity
    if v_ref_mode == "mean":
        v_ref = 0.5 * (vel1 + vel2)    # (B, T, 2)
    elif v_ref_mode == "seq1":
        v_ref = vel1
    elif v_ref_mode == "seq2":
        v_ref = vel2
    else:
        raise ValueError("v_ref_mode must be one of {'mean','seq1','seq2'}")

    # Build A(v_ref) for each (B, T)
    # time_metric = time.time()
    A = _build_A(v_ref, beta_mag=beta_mag, alpha_dir=alpha_dir,
                 sigma=sigma, eps=eps, tau=tau)  # (B, T, 2, 2)
    # print("Metric Construction Time:", time.time() - time_metric)
    # time_dist = time.time()
    # Quadratic form: vel_dist[i,t] = dv[i,t]^T A[i,t] dv[i,t]
    dv_col = dv[..., None]                 # (B, T, 2, 1)
    tmp = np.matmul(A, dv_col)             # (B, T, 2, 1)
    vel_dist = np.matmul(dv_col.transpose(0,1,3,2), tmp)[..., 0, 0]  # (B, T)

    distances = lambda_horiz * pos_dist + lambda_vert * vel_dist
    # print("Distance Computation Time:", time.time() - time_dist)
    return distances, pos_dist, vel_dist


# ---------- small helpers (torch) ----------

def _safe_normalize(x: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Normalize along the last dim with numerical safety."""
    n = torch.linalg.norm(x, dim=-1, keepdim=True).clamp_min(eps)
    return x / n

def _project_tangent(x: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Project v onto the tangent plane at x on S^2: v_tan = v - <v, x> x.
    x, v: (..., 3)
    """
    return v - (torch.sum(v * x, dim=-1, keepdim=True) * x)

def _geodesic_angle(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """
    Geodesic angle theta in [0, pi] between unit vectors x, y in R^3.
    x, y: (..., 3) (assumed unit or near-unit)
    """
    c = torch.sum(x * y, dim=-1).clamp(-1.0, 1.0)
    return torch.arccos(c)

def _build_A_tangent3_torch(
    v_ref: torch.Tensor,
    x_anchor: torch.Tensor,
    beta_mag: float = 1.0,
    alpha_dir: float = 10.0,
    sigma: float = 0.5,
    eps: float = 1e-12,
    tau: float = 1e-6,
) -> torch.Tensor:
    """
    Build an SPD matrix A acting on the tangent plane at x_anchor (S^2 ⊂ R^3).

    Args:
        v_ref: (..., 3) reference velocity (in tangent at x_anchor)
        x_anchor: (..., 3) anchor points on S^2
    Returns:
        A: (..., 3, 3) SPD matrices s.t. <dv, A dv> = a_par||dv_par||^2 + a_perp||dv_perp||^2
    """
    device = x_anchor.device
    dtype = x_anchor.dtype

    x_anchor = _safe_normalize(x_anchor, eps=eps)
    # ensure tangency of v_ref
    v_ref = _project_tangent(x_anchor, v_ref)

    # tangent projector P_tan = I - x x^T
    I3 = torch.eye(3, dtype=dtype, device=device)
    P_tan = I3.expand(*x_anchor.shape[:-1], 3, 3) - x_anchor.unsqueeze(-1) * x_anchor.unsqueeze(-2)  # (...,3,3)

    # default isotropic in tangent: beta_mag * P_tan
    A = beta_mag * P_tan

    # speed and small-speed mask
    s = torch.linalg.norm(v_ref, dim=-1, keepdim=True)             # (...,1)
    small = (s[..., 0] < tau)
    if torch.all(small):
        return A  # fully isotropic where speeds are tiny

    # anisotropy for non-small speeds
    idx = ~small
    v_sel = v_ref[idx]                                             # (N,3)
    x_sel = x_anchor[idx]                                          # (N,3)
    P_tan_sel = P_tan[idx]                                         # (N,3,3)

    s_sel = torch.linalg.norm(v_sel, dim=-1, keepdim=True)         # (N,1)
    u = v_sel / (s_sel + eps)                                      # (N,3) unit direction in tangent

    # parallel / perpendicular projectors inside the tangent space
    P_par = u.unsqueeze(-1) @ u.unsqueeze(-2)                      # (N,3,3)
    P_perp = P_tan_sel - P_par                                     # (N,3,3)

    s2 = (s_sel[..., 0] ** 2)                                      # (N,)
    extra_perp = (alpha_dir / 2.0) * (s2 / (s2 + sigma**2)**2)     # (N,)
    a_par = beta_mag
    a_perp = beta_mag + extra_perp                                 # (N,)

    A_sel = a_par * P_par + a_perp.unsqueeze(-1).unsqueeze(-1) * P_perp
    A[idx] = A_sel
    return A


# ---------- main distance on S^2 (torch) ----------

def riemann_anisotropic_distance_S2(
    seq1: torch.Tensor,
    seq2: torch.Tensor,
    lambda_horiz: float = 1.0,
    lambda_vert: float = 1e-1,  #5e-3
    alpha_dir: float = 1.0,
    beta_mag: float = 1.0,
    sigma: float = 0.001,
    eps: float = 1e-12,
    tau: float = 1e-12,
    v_ref_mode: str = "mean",   # {"mean","seq1","seq2"}
    anchor: str = "seq2",       # {"seq1","seq2"}
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Direction-sensitive Riemannian quadratic distance on S^2 (positions + velocities), PyTorch.

    Inputs:
        seq1, seq2: (B, T, 6) where each step stores [x(3), v(3)], with x ∈ S^2, v ∈ T_x S^2 (approximately).

    Returns:
        distances: (B, T) = lambda_horiz * theta^2 + lambda_vert * <dv, A dv>
        pos_dist : (B, T) = theta^2
        vel_dist : (B, T) = quadratic form in the anchor tangent space

    Notes:
        - Uses existing 'parallel_transport(x1, x2, V)' from your code for velocity alignment.
        - All intermediate x are normalized and all v are projected to tangent for robustness.
    """
    assert seq1.size(-1) == 6, "Expected (B,T,6) inputs"

    device = seq1.device
    dtype = seq1.dtype
    B, T, _ = seq1.shape

    # split and clean inputs
    x1 = _safe_normalize(seq1[..., :3], eps=eps)              # (B,T,3)
    x2 = _safe_normalize(seq2[..., :3], eps=eps)              # (B,T,3)
    v1 = _project_tangent(x1, seq1[..., 3:])                  # (B,T,3)
    v2 = _project_tangent(x2, seq2[..., 3:])                  # (B,T,3)

    # position term: theta^2
    theta = _geodesic_angle(x1, x2)                           # (B,T)
    pos_dist = theta ** 2
    # time_transport = time.time()
    # choose anchor tangent space and parallel-transport the other side
    if anchor == "seq1":
        xA = x1
        # flatten to (B*T,3) for calling provided parallel_transport
        x2f, x1f, v2f = x2.reshape(-1, 3), x1.reshape(-1, 3), v2.reshape(-1, 3)
        v2_at_A_flat = parallel_transport(x2f, x1f, v2f)      # uses your existing function
        v2_at_A = v2_at_A_flat.view(B, T, 3)
        v1_at_A = v1
    elif anchor == "seq2":
        xA = x2
        x1f, x2f, v1f = x1.reshape(-1, 3), x2.reshape(-1, 3), v1.reshape(-1, 3)
        v1_at_A_flat = parallel_transport(x1f, x2f, v1f)      # uses your existing function
        v1_at_A = v1_at_A_flat.view(B, T, 3)
        v2_at_A = v2
    else:
        raise ValueError("anchor must be 'seq1' or 'seq2'")

    # print("Parallel Transport Time:", time.time() - time_transport)
    # time_metric = time.time()

    # choose reference velocity for anisotropic weighting
    if v_ref_mode == "mean":
        v_ref = 0.5 * (v1_at_A + v2_at_A)
    elif v_ref_mode == "seq1":
        v_ref = v1_at_A
    elif v_ref_mode == "seq2":
        v_ref = v2_at_A
    else:
        raise ValueError("v_ref_mode must be one of {'mean','seq1','seq2'}")

    # velocity difference in anchor tangent space (and safety projection)
    dv = _project_tangent(xA, (v1_at_A - v2_at_A))            # (B,T,3)

    # build anisotropic SPD matrix on tangent(xA)
    A = _build_A_tangent3_torch(
        v_ref, xA, beta_mag=beta_mag, alpha_dir=alpha_dir, sigma=sigma, eps=eps, tau=tau
    )                                                         # (B,T,3,3)
    # print("Metric Construction Time:", time.time() - time_metric)

    # quadratic form: dv^T A dv
    time_dist = time.time()
    dv_col = dv.unsqueeze(-1)                                 # (B,T,3,1)
    tmp = torch.matmul(A, dv_col)                             # (B,T,3,1)
    vel_dist = torch.matmul(dv_col.transpose(-2, -1), tmp)[..., 0, 0]  # (B,T)
    

    distances = lambda_horiz * pos_dist + lambda_vert * vel_dist
    # print("Distance Computation Time:", time.time() - time_dist)
    return distances, pos_dist, vel_dist
    




# ---------- anisotropic SPD on se(3) (6x6) ----------
import torch

def _build_A_se3(
    xi_ref: torch.Tensor,
    *,
    # rotational block hyper-parameters
    beta_rot: float = 1.0,
    alpha_rot: float = 10.0,
    sigma_rot: float = 0.5,
    enable_rot_aniso: bool = True,
    # translational block hyper-parameters
    beta_trans: float = 1.0,
    alpha_trans: float = 10.0,
    sigma_trans: float = 0.5,
    enable_trans_aniso: bool = True,
    # numerics
    eps: float = 1e-12,
    tau: float = 1e-6,
) -> torch.Tensor:
    """
    Build a 6x6 SPD matrix A for se(3) twist xi = [w(3), v(3)].

    Design:
      A = diag(A_rot(omega_ref), A_trans(v_ref)),  no cross-coupling blocks.
      Each 3x3 block is either isotropic beta*I or anisotropic
      with parallel/perpendicular weights relative to the reference direction.

    Args:
        xi_ref: (..., 6) reference twist = [omega_ref(3), v_ref(3)]
        beta_rot: baseline weight for rotational magnitude
        alpha_rot: extra penalty for rotational components perpendicular to omega_ref
        sigma_rot: smoothing scale near ||omega_ref|| ~ 0 for rotational block
        enable_rot_aniso: enable anisotropy on rotational block
        beta_trans: baseline weight for translational magnitude
        alpha_trans: extra penalty for translational components perpendicular to v_ref
        sigma_trans: smoothing scale near ||v_ref|| ~ 0 for translational block
        enable_trans_aniso: enable anisotropy on translational block
        eps, tau: numerical stabilizers

    Returns:
        A: (..., 6, 6) SPD matrix
    """
    device, dtype = xi_ref.device, xi_ref.dtype
    *lead, _ = xi_ref.shape
    A = torch.zeros(*lead, 6, 6, device=device, dtype=dtype)

    # split reference twist
    w_ref = xi_ref[..., :3]   # (...,3)
    v_ref = xi_ref[..., 3:]   # (...,3)

    # ---- rotational block ----
    I3r = torch.eye(3, device=device, dtype=dtype)
    A_rot = I3r * beta_rot

    if enable_rot_aniso:
        s = torch.linalg.norm(w_ref, dim=-1, keepdim=True)             # (...,1)
        small = (s[..., 0] < tau)
        if (~small).any():
            idx = ~small
            w_sel = w_ref[idx]
            s_sel = torch.linalg.norm(w_sel, dim=-1, keepdim=True)     # (N,1)
            u = w_sel / (s_sel + eps)                                  # (N,3)
            P_par  = u.unsqueeze(-1) @ u.unsqueeze(-2)                 # (N,3,3)
            I3     = torch.eye(3, device=device, dtype=dtype).expand_as(P_par)
            P_perp = I3 - P_par
            s2 = (s_sel.squeeze(-1) ** 2)                              # (N,)
            # same small-angle shape as in your S^2/R^2 code
            extra_perp = (alpha_rot / 2.0) * (s2 / (s2 + sigma_rot**2)**2)  # (N,)
            a_par  = beta_rot
            a_perp = beta_rot + extra_perp.unsqueeze(-1).unsqueeze(-1)      # (N,1,1)
            A_rot_sel = a_par * P_par + a_perp * P_perp                      # (N,3,3)

            # scatter into A_rot per batch/time index
            # Start with isotropic, then write anisotropic at idx
            # We build a full tensor for A_rot to ease assignment
            A_rot_full = A_rot.expand(*lead, 3, 3).clone()
            A_rot_full[idx] = A_rot_sel
            A[..., 0:3, 0:3] = A_rot_full
        else:
            A[..., 0:3, 0:3] = A_rot
    else:
        A[..., 0:3, 0:3] = A_rot

    # ---- translational block ----
    I3t = torch.eye(3, device=device, dtype=dtype)
    A_trans = I3t * beta_trans

    if enable_trans_aniso:
        s = torch.linalg.norm(v_ref, dim=-1, keepdim=True)             # (...,1)
        small = (s[..., 0] < tau)
        if (~small).any():
            idx = ~small
            v_sel = v_ref[idx]
            s_sel = torch.linalg.norm(v_sel, dim=-1, keepdim=True)     # (N,1)
            u = v_sel / (s_sel + eps)                                  # (N,3)
            P_par  = u.unsqueeze(-1) @ u.unsqueeze(-2)                 # (N,3,3)
            I3     = torch.eye(3, device=device, dtype=dtype).expand_as(P_par)
            P_perp = I3 - P_par
            s2 = (s_sel.squeeze(-1) ** 2)                              # (N,)
            extra_perp = (alpha_trans / 2.0) * (s2 / (s2 + sigma_trans**2)**2)  # (N,)
            a_par  = beta_trans
            a_perp = beta_trans + extra_perp.unsqueeze(-1).unsqueeze(-1)
            A_trans_sel = a_par * P_par + a_perp * P_perp              # (N,3,3)

            A_trans_full = A_trans.expand(*lead, 3, 3).clone()
            A_trans_full[idx] = A_trans_sel
            A[..., 3:6, 3:6] = A_trans_full
        else:
            A[..., 3:6, 3:6] = A_trans
    else:
        A[..., 3:6, 3:6] = A_trans

    # cross-coupling blocks (rot-trans) are kept zero for simplicity and SPD
    # If you later need coupling, ensure symmetry and SPD (e.g., via Schur complement).

    return A



# ---------- parallel transport (left-invariant) ----------
def parallel_transport_twist_SE3(xi2: torch.Tensor,
                                 T1: torch.Tensor,
                                 T2: torch.Tensor) -> torch.Tensor:
    """
    Parallel transport twist xi2 ∈ T_{T2}SE(3) to T_{T1}SE(3)
    using left-invariant metric: PT = Ad_{T1^{-1} T2} xi2.

    xi2 : (B, T, 6)
    T1  : (B, T, 4, 4)
    T2  : (B, T, 4, 4)
    returns: (B, T, 6)
    """
    # relative transform T_rel = T1^{-1} T2
    T1_inv = invSE3(T1.reshape(-1,4,4)).reshape(*T1.shape)
    T_rel  = torch.matmul(T1_inv, T2)  # (B,T,4,4)
    
    if torch.any(torch.isnan(T_rel)):
        raise ValueError("T_rel contains NaNs")

    # Ad_{T_rel} ∈ R^{6x6}
    Ad = largeAdjoint(T_rel.reshape(-1,4,4)).reshape(*T_rel.shape[:-2], 6, 6)  # (B,T,6,6)

    xi2_col = xi2.unsqueeze(-1)                        # (B,T,6,1)
    xi2_at_T1 = torch.matmul(Ad, xi2_col)[..., 0]      # (B,T,6)
    return xi2_at_T1


# ---------- pose distance on SE(3) ----------
def _pose_distance_SE3(T1: torch.Tensor, T2: torch.Tensor,
                       lambda_rot: float = 1.0,
                       lambda_pos: float = 1.0) -> torch.Tensor:
    """
    Squared pose distance per time:
    d^2 = lambda_rot * ||log(R1^T R2)||_F^2 + lambda_pos * ||p1 - p2||^2

    T1, T2: (B, T, 4, 4)
    returns: (B, T)
    """
    R1, p1 = T1[..., :3, :3], T1[..., :3, 3]
    R2, p2 = T2[..., :3, :3], T2[..., :3, 3]

    R_rel = torch.matmul(R1.transpose(-1, -2), R2)  # (B,T,3,3)
    so3   = logSO3(R_rel.reshape(-1,3,3)).reshape(*R_rel.shape)  # use provided logSO3
    dR2   = torch.linalg.matrix_norm(so3, dim=(-2, -1))**2       # ||log||_F^2
    dp2   = torch.sum((p1 - p2)**2, dim=-1)
    return lambda_rot * dR2 + lambda_pos * dp2


# ---------- main metric with PT ----------
def riemann_anisotropic_distance_SE3_PT(
    Tseq1: torch.Tensor, Tseq2: torch.Tensor,
    *,
    # You can pass either Vb (body twist) directly, or Tdot; if Tdot is given we convert to Vb.
    Vb1: torch.Tensor | None = None,
    Vb2: torch.Tensor | None = None,
    Tdot1: torch.Tensor | None = None,
    Tdot2: torch.Tensor | None = None,
    lambda_rot: float = 1.0,
    lambda_pos: float = 10.0,
    lambda_vel: float = 1e-1,
    alpha_dir: float = 10.0,
    beta_mag: float = 1.0,
    sigma: float = 0.5,
    eps: float = 1e-12,
    tau: float = 1e-6,
    v_ref_mode: str = "mean"  # {"mean","seq1","seq2"}
):
    """
    Direction-sensitive Riemannian distance on SE(3) with proper parallel transport.

    Inputs
    ------
    Tseq1, Tseq2 : (B, T, 4, 4) poses
    Vb1, Vb2     : (B, T, 6) body twists at Tseq1 / Tseq2
    Tdot1, Tdot2 : (B, T, 4, 4) pose time-derivatives (if Vb* not provided)

    Returns
    -------
    distances : (B, T)
    pose_part : (B, T)  = lambda_rot*||log(R1^T R2)||^2 + lambda_pos*||p1-p2||^2
    vel_part  : (B, T)  = <Δξ, A(ξ_ref) Δξ>  with ξ in T_{Tseq1}SE(3)
    """
    B, T, _, _ = Tseq1.shape

    # 1) pose part
    pose_part = _pose_distance_SE3(Tseq1, Tseq2, lambda_rot=lambda_rot, lambda_pos=lambda_pos)

    # 2) get body twists
    if Vb1 is None or Vb2 is None:
        assert (Tdot1 is not None) and (Tdot2 is not None), "Provide Vb* or Tdot*."
    if Vb1 is None:
        Vb1 = Tdot_to_Vb(Tdot1.reshape(-1,4,4), Tseq1.reshape(-1,4,4)).reshape(B,T,4,4)
        Vb1 = screw_bracket(Vb1).reshape(B,T,6)  # [w,v]
    if Vb2 is None:
        Vb2 = Tdot_to_Vb(Tdot2.reshape(-1,4,4), Tseq2.reshape(-1,4,4)).reshape(B,T,4,4)
        Vb2 = screw_bracket(Vb2).reshape(B,T,6)

    # 3) parallel transport xi2 to T1
    
    xi2_at_T1 = parallel_transport_twist_SE3(Vb2, Tseq1, Tseq2)  # (B,T,6)

    # 4) reference twist for anisotropy
    if v_ref_mode == "mean":
        
        xi_ref = 0.5 * (Vb1 + xi2_at_T1)
    elif v_ref_mode == "seq1":
        xi_ref = Vb1
    elif v_ref_mode == "seq2":
        xi_ref = xi2_at_T1
    else:
        raise ValueError("v_ref_mode must be one of {'mean','seq1','seq2'}")

    # 5) anisotropic SPD A(ξ_ref)

    A = _build_A_se3(
    xi_ref,
    beta_rot=1.0, alpha_rot=10.0, sigma_rot=0.3, enable_rot_aniso=True,
    beta_trans=1.0, alpha_trans=10.0, sigma_trans=0.5, enable_trans_aniso=True
)
    

    # 6) velocity difference in same tangent space T_{T1}SE(3)
    dxi = (Vb1 - xi2_at_T1).unsqueeze(-1)           # (B,T,6,1)
    tmp = torch.matmul(A, dxi)                      # (B,T,6,1)
    vel_part = torch.matmul(dxi.transpose(-2, -1), tmp)[..., 0, 0]  # (B,T)
    if torch.isnan(vel_part).any():
        print("vel_part is NaN")
    distances = pose_part + lambda_vel * vel_part
    return distances, pose_part, vel_part
