import numpy as np
import torch
from agent.utils.S2_functions import parallel_transport


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




import numpy as np

def sasaki_R2(seq1, seq2, lambda_horiz=1.0, lambda_vert=1e-1,
              alpha_dir=10.0, beta_mag=1.0, eps=1e-12):
    """
    Compute a direction-sensitive distance between two batches of 2D position-velocity sequences.

    Parameters
    ----------
    seq1, seq2 : array-like, shape (B, T, 4)
        Two batches of sequences. Each time step holds [x, y, vx, vy].
    lambda_horiz : float
        Weight for position (horizontal) differences. Multiplies ||?x||^2.
    lambda_vert : float
        Overall weight for the velocity (vertical) distance term (for backward compatibility).
        Multiplies the combined velocity distance: beta_mag*||?v||^2 + alpha_dir*(1 - cos?).
    alpha_dir : float
        Direction-mismatch weight ? in the velocity distance.
    beta_mag : float
        Magnitude-difference weight ? in the velocity distance.
    eps : float
        Small constant for numerical stability (e.g., when velocity norms are near zero).

    Returns
    -------
    distances : ndarray, shape (B, T)
        Per-time-step distance values for each pair of sequences.
        (If you prefer a single value per sequence, apply a reduction like sum/mean along axis=1.)
    """

    # Basic checks
    seq1 = np.asarray(seq1)
    seq2 = np.asarray(seq2)
    B, T, D = seq1.shape
    # print(seq1.shape, seq2.shape)
    # assert seq2.shape == (B, T, D), "Both sequences must have the same shape (B, T, 4)."
    # assert D == 4, "Input must have 4 channels per time step: [x, y, vx, vy]."

    # 1) Split positions and velocities
    pos1 = seq1[:, :, :2]   # (B, T, 2)
    pos2 = seq2[:, :, :2]   # (B, T, 2)
    vel1 = seq1[:, :, 2:]   # (B, T, 2)
    vel2 = seq2[:, :, 2:]   # (B, T, 2)

    # 2) Position distance: squared Euclidean per time step
    dpos = pos1 - pos2                      # (B, T, 2)
    pos_dist = np.sum(dpos**2, axis=2)      # (B, T)

    # 3) Velocity distance:
    #    beta_mag * ||v - w||^2 + alpha_dir * (1 - cos(theta))
    dv = vel1 - vel2
    diff_mag2 = np.sum(dv**2, axis=2)       # (B, T)

    # Cosine similarity with safe handling for zero vectors
    dot = np.sum(vel1 * vel2, axis=2)                       # (B, T)
    n1 = np.linalg.norm(vel1, axis=2)                       # (B, T)
    n2 = np.linalg.norm(vel2, axis=2)                       # (B, T)
    denom = np.maximum(n1 * n2, eps)                        # avoid division by zero
    cos_sim = dot / denom                                   # (B, T)

    # Handle undefined directions explicitly:
    both_zero = (n1 < eps) & (n2 < eps)                     # both velocities ~ zero
    one_zero  = ((n1 < eps) ^ (n2 < eps))                   # exactly one is ~ zero

    # Default cosine dissimilarity
    cos_dissim = 1.0 - cos_sim

    # If both are zero, direction mismatch is 0 and magnitude diff is 0
    cos_dissim[both_zero] = 0.0
    diff_mag2[both_zero]  = 0.0

    # If exactly one is zero, treat direction as maximally mismatched (penalty = 1)
    cos_dissim[one_zero] = 1.0
    # diff_mag2 remains as computed (equals the nonzero norm squared)

    vel_dist = beta_mag * diff_mag2 + alpha_dir * cos_dissim  # (B, T)

    # 4) Combine position and velocity parts
    distances = lambda_horiz * pos_dist + lambda_vert * vel_dist  # (B, T)

    return distances , pos_dist, vel_dist





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
                                 lambda_vert=0.1,
                                 alpha_dir=10.0,
                                 beta_mag=1.0,
                                 sigma=0.5,
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
    A = _build_A(v_ref, beta_mag=beta_mag, alpha_dir=alpha_dir,
                 sigma=sigma, eps=eps, tau=tau)  # (B, T, 2, 2)

    # Quadratic form: vel_dist[i,t] = dv[i,t]^T A[i,t] dv[i,t]
    dv_col = dv[..., None]                 # (B, T, 2, 1)
    tmp = np.matmul(A, dv_col)             # (B, T, 2, 1)
    vel_dist = np.matmul(dv_col.transpose(0,1,3,2), tmp)[..., 0, 0]  # (B, T)

    distances = lambda_horiz * pos_dist + lambda_vert * vel_dist
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
    *,
    lambda_horiz: float = 1.0,
    lambda_vert: float = 1e-2,  #5e-3
    alpha_dir: float = 10.0,
    beta_mag: float = 1.0,
    sigma: float = 0.5,
    eps: float = 1e-12,
    tau: float = 1e-6,
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

    # quadratic form: dv^T A dv
    dv_col = dv.unsqueeze(-1)                                 # (B,T,3,1)
    tmp = torch.matmul(A, dv_col)                             # (B,T,3,1)
    vel_dist = torch.matmul(dv_col.transpose(-2, -1), tmp)[..., 0, 0]  # (B,T)
    

    distances = lambda_horiz * pos_dist + lambda_vert * vel_dist
    return distances, pos_dist, vel_dist
