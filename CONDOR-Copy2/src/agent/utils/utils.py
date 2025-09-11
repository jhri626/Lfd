import numpy as np
from scipy.signal import butter, filtfilt
import os
import torch
from typing import List, Union

from agent.utils.S2_functions import *




def _to_numpy(a):
    """Convert torch.Tensor or list to numpy array."""
    try:
        import torch
        if isinstance(a, torch.Tensor):
            return a.detach().cpu().numpy()
    except ImportError:
        pass
    return np.asarray(a)

def lpf_demonstrations(demo_array, fs=1.0, L_min=20, order=4):
    """
    Apply LPF to demonstrations array of shape (N, T, D).
    Returns filtered array of the same shape.
    """
    N, T, D = demo_array.shape
    demo_f = np.zeros_like(demo_array)
    for n in range(N):
        for d in range(D):
            demo_f[n, :, d] = butter_lpf_uniform(demo_array[n, :, d],
                                                fs=fs, L_min=L_min, order=order)
    return demo_f


def butter_lpf_uniform(signal, fs=1.0, L_min=20, order=4):
    """
    Apply zero-phase Butterworth LPF to a 1D signal.
    signal : 1D array (length T)
    fs     : sampling frequency (1/dt). If dt=1, set fs=1.
    L_min  : smooth out variations shorter than L_min samples
    order  : filter order
    """
    fc = fs / float(L_min)
    wn = fc / (0.5 * fs)
    if not (0 < wn < 1):
        raise ValueError("Cutoff must satisfy 0 < fc < fs/2.")
    b, a = butter(order, wn, btype="low", analog=False)
    return filtfilt(b, a, signal, method="pad")

def create_directories(results_path):
    """
    Creates the requested directory and subfolders
    """
    try:
        if not os.path.exists(results_path + 'images/'):
            os.makedirs(results_path + 'images/')
            os.makedirs(results_path + 'stats/')
        print('Results directory created:', results_path)
    except FileExistsError:
        print('Results directory already exists:', results_path)
        
        
def generate_random_initial_states(num_states, params, demonstrations, mode='init', sampling_std=0.1, r_min=0.05, r_max=0.2):
    """
    Generate initial states based on mode
    mode='init': random states near starting point (existing logic)
    mode='grid': 25 points in a 5x5 grid between (-1,-1) and (1,1)
    mode='all': random sampling from all demonstration points with gaussian noise
    """
    if mode == 'init':
        # Original logic: random states near starting point
        demo = demonstrations[0]  # Single demonstration
        start_point = demo[0, :params.workspace_dimensions]  # First time step, position only
        
        print(f"Demonstration start point: {start_point}")
        print(f"Sampling radius around start point: {sampling_std}")
        
        # Generate random positions near the start point
        random_positions = []
        dim = start_point.shape[0]
        
        for i in range(num_states):
            direction = np.random.normal(0, 1, size=dim)
            direction /= np.linalg.norm(direction)
            u = np.random.uniform(r_min**dim, r_max**dim)
            radius = u ** (1.0 / dim)
            # Generate random offset within the sampling radius
            random_position = start_point + radius * direction
            random_positions.append(random_position)
        
        random_positions = np.array(random_positions)
        
    elif mode == 'grid':
        # Grid mode: 25 points in 5x5 grid between (-1,-1) and (1,1)
        print("Generating 5x5 grid points between (-1,-1) and (1,1)")
        
        if params.workspace_dimensions == 2:
            # Create 5x5 grid
            x = np.linspace(-1, 1, 5)
            y = np.linspace(-1, 1, 5)
            xx, yy = np.meshgrid(x, y)
            
            # Flatten to get 25 points
            grid_x = xx.flatten()
            grid_y = yy.flatten()
            random_positions = np.column_stack([grid_x, grid_y])
            
            print(f"Generated 25 grid points (using first {min(num_states, 25)} points)")
            # If num_states < 25, use only the first num_states points
            # If num_states > 25, repeat the pattern or use only 25
            if num_states <= 25:
                random_positions = random_positions[:num_states]
            else:
                print(f"Warning: num_states ({num_states}) > 25, using only 25 grid points")
                random_positions = random_positions[:25]
                
        else:
            raise ValueError(f"Grid mode currently only supports 2D workspace, got {params.workspace_dimensions}D")
    
    elif mode == 'all':
        # Random sampling from all demonstration points with gaussian noise
        print(f"Generating {num_states} states by random sampling from all demonstration points with gaussian noise")
        print(f"Gaussian noise std: {sampling_std}")
        
        # Collect all positions from all demonstrations
        all_positions = []
        for demo in demonstrations:
            # Extract positions (first workspace_dimensions columns) from all time steps
            positions = demo[:, :params.workspace_dimensions]
            all_positions.append(positions)
        
        # Concatenate all positions
        all_positions = np.vstack(all_positions)  # Shape: (total_points, workspace_dimensions)
        print(f"Total demonstration points available: {len(all_positions)}")
        
        # Random sampling from all demonstration points
        random_positions = []
        for i in range(num_states):
            # Randomly select a point from all demonstration points
            random_idx = np.random.randint(0, len(all_positions))
            selected_point = all_positions[random_idx].copy()
            
            # Add gaussian noise
            gaussian_noise = np.random.normal(
                loc=0.0, 
                scale=sampling_std, 
                size=params.workspace_dimensions
            )
            
            # Add noise to selected point
            noisy_position = selected_point + gaussian_noise
            random_positions.append(noisy_position)
        
        random_positions = np.array(random_positions)
        
        print(f"Sample of selected demonstration points (before noise):")
        sample_indices = np.random.choice(len(all_positions), min(3, len(all_positions)), replace=False)
        for i, idx in enumerate(sample_indices):
            print(f"  Demo point {idx}: {all_positions[idx]}")
        print("\n")
    
    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'init', 'grid', or 'all'")
    
    # Add velocities if second order system
    if params.dynamical_system_order == 1:
        # First order: only positions
        initial_states = random_positions
    elif params.dynamical_system_order == 2:
        # Second order: positions + velocities (start with zero velocity)
        random_velocities = np.zeros((len(random_positions), params.workspace_dimensions))
        initial_states = np.concatenate([random_positions, random_velocities], axis=1)
    else:
        raise ValueError(f"Unsupported dynamical system order: {params.dynamical_system_order}")
    
    print(f"Generated {len(initial_states)} initial states with shape: {initial_states.shape}")
    if mode == 'grid':
        print("Grid points:")
        for i, state in enumerate(initial_states):
            if params.dynamical_system_order == 1:
                print(f"  Point {i+1}: ({state[0]:.2f}, {state[1]:.2f})")
            else:
                print(f"  Point {i+1}: pos=({state[0]:.2f}, {state[1]:.2f}), vel=({state[2]:.2f}, {state[3]:.2f})")
        print("\n")
    elif mode == 'all':
        print("Sample of generated initial states (with noise):")
        for i in range(min(5, len(initial_states))):
            if params.dynamical_system_order == 1:
                print(f"  State {i+1}: pos=({initial_states[i][0]:.3f}, {initial_states[i][1]:.3f})")
            else:
                print(f"  State {i+1}: pos=({initial_states[i][0]:.3f}, {initial_states[i][1]:.3f}), vel=({initial_states[i][2]:.3f}, {initial_states[i][3]:.3f})")
        print("\n")
    else:
        print("Sample initial states:")
        for i in range(min(3, len(initial_states))):
            print(f"  State {i+1}: {initial_states[i]}")
        print("\n")
    return initial_states




@torch.no_grad()
def generate_random_initial_states_s2(
    num_states: int,
    params,
    demonstrations: Union[List[torch.Tensor], torch.Tensor],  # list of (T,2) or (B,T,2)
    mode: str = 'init',
    sampling_std: float = 0.1,
    r_min: float = 0.05,
    r_max: float = 0.15,
    verbose: bool = True,
) -> torch.Tensor:
    """
    Generate initial states on S^2 in q=(theta,phi) (and optionally qdot) form.

    Modes:
      - 'init': random states near the first demo's start point using tangent-plane sampling + exp map
      - 'grid': 5x5 grid in the start-point tangent plane (geodesic square), then exp map
      - 'all' : pick random demo frames and perturb on their tangent planes with Gaussian noise

    Returns:
      initial_states:
        - if params.dynamical_system_order == 1: (N, 2)           -> q only
        - if params.dynamical_system_order == 2: (N, 4)           -> [q, q_dot] with q_dot = 0
    """
    assert hasattr(params, "dynamical_system_order"), "params.dynamical_system_order is required"
    assert hasattr(params, "workspace_dimensions"), "params.workspace_dimensions is required"
    if params.workspace_dimensions != 2:
        raise ValueError(f"S^2 version expects workspace_dimensions==2 (q=theta,phi), got {params.workspace_dimensions}")

    # ---- collect device/dtype from demonstrations
    def _as_tensor(x) -> torch.Tensor:
        if isinstance(x, torch.Tensor):
            return x
        import numpy as np
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x)
        raise TypeError("demonstrations must be a list/array/tensor of q-trajectories")

    if isinstance(demonstrations, list):
        demos = [ _as_tensor(d).detach() for d in demonstrations ]
        device = demos[0].device
        dtype  = demos[0].dtype
    else:
        demos = [ _as_tensor(demonstrations).detach() ]
        device = demos[0].device
        dtype  = demos[0].dtype

    # Helper: ensure (T,2) tensor
    def _ensure_T2(t: torch.Tensor) -> torch.Tensor:
        if t.ndim == 2 and t.shape[-1] == 2:
            return t
        if t.ndim == 3 and t.shape[-1] == 2 and t.shape[0] == 1:
            return t.squeeze(0)
        raise ValueError("Each demo must be shaped (T,2) or (1,T,2)")

    demos = [ _ensure_T2(d).to(device=device, dtype=dtype) for d in demos ]

    # -----------------------------
    # tangent helpers
    # -----------------------------
    def _tangent_basis(x: torch.Tensor, eps: float = 1e-9):
        """Build an orthonormal basis (e1,e2) for T_x S^2 given x (3,)."""
        x = x / torch.linalg.norm(x)
        # choose an auxiliary vector not parallel to x
        aux = torch.tensor([1.0, 0.0, 0.0], dtype=x.dtype, device=x.device)
        if torch.abs(torch.dot(x, aux)) > 0.9:
            aux = torch.tensor([0.0, 1.0, 0.0], dtype=x.dtype, device=x.device)
        e1 = aux - torch.dot(aux, x) * x
        n1 = torch.linalg.norm(e1).clamp_min(eps)
        e1 = e1 / n1
        e2 = torch.cross(x, e1)
        e2 = e2 / torch.linalg.norm(e2).clamp_min(eps)
        return e1, e2

    def _sample_uniform_annulus(N: int, rmin: float, rmax: float, device, dtype):
        """Sample radii uniformly in area in 2D (sqrt of uniform on [rmin^2, rmax^2]) and angles uniform."""
        u = torch.rand(N, device=device, dtype=dtype)
        r = torch.sqrt((rmax**2 - rmin**2) * u + rmin**2)
        theta = 2 * torch.pi * torch.rand(N, device=device, dtype=dtype)
        return r, theta

    # -----------------------------
    # mode implementations
    # -----------------------------
    if mode == 'init':
        # Use the first demonstration's start point as center on S^2
        q0 = demos[0][0]                      # (2,)
        x0 = q_to_x(q0.unsqueeze(0)).squeeze(0)  # (3,)
        e1, e2 = _tangent_basis(x0)
        

        r, ang = _sample_uniform_annulus(num_states, r_min, r_max, device, dtype)
        # build tangent vectors v_i = r_i (cos a e1 + sin a e2)
        v = (r.unsqueeze(1) * (torch.cos(ang).unsqueeze(1) * e1 + torch.sin(ang).unsqueeze(1) * e2))  # (N,3)
        # v = torch.zeros_like(v)

        x0_rep = x0.unsqueeze(0).repeat(num_states, 1)  # (N,3)
        x_init = exp_sphere(x0_rep, v)                  # (N,3)
        q_init = x_to_q(x_init)                         # (N,2)

        if verbose:
            print(f"[S2:init] center q0 = {q0.tolist()}, r in [{r_min}, {r_max}] (geodesic)")

        random_positions_q = q_init

    elif mode == 'grid':
        # Global q-grid on S^2: latitude full range, longitude in [pi, 3*pi/2]
        if verbose:
            print("[S2:grid] Generating global q-grid (lat full, lon in [pi, 3*pi/2])")

        # ---------- parameters ----------
        eps = 1e-6  # small margin to avoid pole singularities
        lon_min = 1.5 * math.pi
        lon_max = 2 * math.pi

        # Decide grid resolution (H x W) based on num_states
        # Heuristic: make H and W as "square" as possible
        H = int(math.ceil(math.sqrt(num_states)))
        W = int(math.ceil(num_states / H))

        # Toggle this if you want equal-area bands instead of uniform-lat spacing
        equal_area = False  # set True for near equal-area latitude bands

        # ---------- build latitude vector ----------
        if equal_area:
            # Equal-area: sample u = sin(lat) uniformly in [-1,1]
            # Then lat = arcsin(u). Avoid exact -1 and 1 with eps.
            u = torch.linspace(-1.0 + eps, 1.0 - eps, steps=H, device=device, dtype=dtype)
            lats = torch.arcsin(u)
        else:
            # Uniform in latitude
            lat_min =  math.pi/6 + eps
            lat_max =  2 * math.pi/3 - eps
            lats = torch.linspace(lat_min, lat_max, steps=H, device=device, dtype=dtype)

        # ---------- build longitude vector ----------
        lons = torch.linspace(lon_min, lon_max, steps=W, device=device, dtype=dtype)

        # ---------- mesh and pack to q-grid ----------
        a, b = torch.meshgrid(lats, lons, indexing="ij")             # a: lat, b: lon  -> shapes (H, W)
        q_grid = torch.stack([a, b], dim=-1).reshape(-1, 2)          # (H*W, 2)

        # ---------- match num_states behavior ----------
        total = q_grid.shape[0]
        if num_states <= total:
            random_positions_q = q_grid[:num_states]
        else:
            if verbose:
                print(f"[S2:grid] Warning: num_states ({num_states}) > H*W ({total}), using only {total} grid points")
            random_positions_q = q_grid  # (total, 2)


    elif mode == 'all':
        # Sample from all demo frames, then perturb on tangent with Gaussian noise (std in geodesic radians)
        if verbose:
            print(f"[S2:all] Sampling {num_states} states from all demos with tangent Gaussian noise (std={sampling_std})")

        # Concatenate all q's
        all_q_list = [d for d in demos]               # list of (T,2)
        lengths = [q.shape[0] for q in all_q_list]
        total_T = sum(lengths)
        all_q = torch.cat(all_q_list, dim=0)          # (total_T, 2)

        # Random indices
        idx = torch.randint(low=0, high=total_T, size=(num_states,), device=device)
        q_sel = all_q[idx]                             # (N,2)

        # Map to x and build tangent noise (Gaussian in R^3 then project)
        x_sel = q_to_x(q_sel)                          # (N,3)
        g = torch.randn(num_states, 3, device=device, dtype=dtype)
        g_tan = g - (g * x_sel).sum(dim=-1, keepdim=True) * x_sel
        g_tan = g_tan / torch.linalg.norm(g_tan, dim=-1, keepdim=True).clamp_min(1e-12)

        # Radii ~ |N(0, sampling_std)|, or clip to [r_min, r_max] if you want bounds:
        # Here: pure Gaussian magnitude on geodesic
        radii = torch.abs(torch.randn(num_states, device=device, dtype=dtype) * sampling_std)
        v = radii.unsqueeze(1) * g_tan                 # (N,3) tangent displacements

        x_init = exp_sphere(x_sel, v)
        q_init = x_to_q(x_init)

        random_positions_q = q_init

        if verbose:
            print(f"[S2:all] Total frames available: {total_T}")

    else:
        raise ValueError(f"Unknown mode: {mode}. Use 'init', 'grid', or 'all'")

    # -----------------------------
    # pack initial states
    # -----------------------------
    if params.dynamical_system_order == 1:
        initial_states = random_positions_q  # (N,2)
    elif params.dynamical_system_order == 2:
        zeros_qdot = torch.zeros(random_positions_q.shape[0], 2, dtype=dtype, device=device)
        initial_states = torch.cat([random_positions_q, zeros_qdot], dim=1)  # (N,4) -> [q, qdot=0]
    else:
        raise ValueError(f"Unsupported dynamical system order: {params.dynamical_system_order}")

    if verbose:
        print(f"[S2] Generated {initial_states.shape[0]} initial states, shape={tuple(initial_states.shape)}")
        if mode == 'grid':
            for i in range(initial_states.shape[0]):
                if params.dynamical_system_order == 1:
                    th, ph = initial_states[i]
                    print(f"  Point {i+1}: q=({float(th):.3f}, {float(ph):.3f})")
                else:
                    th, ph, thd, phd = initial_states[i]
                    print(f"  Point {i+1}: q=({float(th):.3f},{float(ph):.3f}), qdot=({float(thd):.3f},{float(phd):.3f})")
        else:
            show_n = min(5, initial_states.shape[0])
            print("[S2] Sample initial states:")
            for i in range(show_n):
                if params.dynamical_system_order == 1:
                    th, ph = initial_states[i]
                    print(f"  State {i+1}: q=({float(th):.3f}, {float(ph):.3f})")
                else:
                    th, ph, thd, phd = initial_states[i]
                    print(f"  State {i+1}: q=({float(th):.3f}, {float(ph):.3f}), qdot=({float(thd):.3f}, {float(phd):.3f})")

    return initial_states
    
    
#============================ s2 filter ======================
import torch
import torch.nn.functional as F
from math import pi

# ========== Utilities ==========
def normalize_rows_torch(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Row-wise normalization to unit length."""
    n = torch.linalg.norm(X, dim=-1, keepdim=True).clamp_min(eps)
    return X / n

# ========== Fréchet mean on S^2 using user-provided log/exp ==========
@torch.no_grad()
def frechet_mean_with_user_funcs(X: torch.Tensor, max_iter: int = 10, tol: float = 1e-10) -> torch.Tensor:
    """
    Compute Fréchet mean on S^2 for a set of unit vectors X using vel_geo_0_sphere (log) and exp_sphere (exp).
    X: (N,3) unit vectors
    returns m: (3,) unit vector
    """
    m = normalize_rows_torch(X.mean(dim=0, keepdim=True))[0]  # init with normalized arithmetic mean
    N = X.shape[0]

    for _ in range(max_iter):
        mN = m.unsqueeze(0).expand(N, 3)              # (N,3) repeat pole
        V  = vel_geo_0_sphere(mN, X)                  # log_m(X_i) using user's function
        delta = V.mean(dim=0, keepdim=True)           # (1,3)
        if torch.linalg.norm(delta).item() < tol:
            break
        m = exp_sphere(m.unsqueeze(0), delta).squeeze(0)  # update pole on S^2
        m = normalize_rows_torch(m.unsqueeze(0))[0]
    return m

# ========== FIR low-pass (windowed-sinc) and zero-phase application ==========
def design_fir_lowpass_torch(fc: float, fs: float, numtaps: int = 101,
                             window: str = "hamming",
                             device=None, dtype=None) -> torch.Tensor:
    """
    Create a windowed-sinc low-pass FIR kernel (Hamming window).
    fc: cutoff frequency (Hz), fs: sampling rate (Hz), numtaps must be odd.
    """
    if numtaps % 2 == 0:
        raise ValueError("numtaps must be odd for a symmetric FIR kernel.")
    if not (0.0 < fc < 0.5 * fs):
        raise ValueError("Cutoff must satisfy 0 < fc < fs/2.")

    n = torch.arange(numtaps, device=device, dtype=dtype)
    mid = (numtaps - 1) / 2.0
    fc_norm = fc / fs  # cycles per sample in [0, 0.5]

    # Ideal low-pass impulse response: 2*fc_norm*sinc(2*fc_norm*(n-mid)); torch.sinc uses sin(pi x)/(pi x)
    h = 2.0 * fc_norm * torch.sinc(2.0 * fc_norm * (n - mid))

    if window.lower() == "hamming":
        w = 0.54 - 0.46 * torch.cos(2.0 * pi * n / (numtaps - 1))
        h = h * w
    else:
        raise NotImplementedError("Only 'hamming' window is implemented.")

    h = h / h.sum()  # DC normalization
    return h

def apply_zero_phase_fir_torch(data: torch.Tensor, h: torch.Tensor) -> torch.Tensor:
    """
    Apply zero-phase FIR to multi-channel series via forward-backward depthwise conv1d.
    data: (T, D)  time-major; D channels are filtered independently.
    h:    (M,)    FIR kernel
    returns: (T, D)
    """
    T, D = data.shape
    M = h.numel()

    # Prepare input as (B=1, C=D, T)
    xx = data.t().unsqueeze(0)  # (1, D, T)

    # Depthwise conv kernel (C,1,M). Flip h to realize convolution (conv1d is correlation by default).
    kernel = h.flip(0).view(1, 1, M).repeat(D, 1, 1)
    pad = (M - 1) // 2

    # Forward pass with reflect padding
    y_f = F.conv1d(F.pad(xx, (pad, pad), mode="reflect"), kernel, groups=D)

    # Backward pass (filtfilt-like)
    y_rev = torch.flip(y_f, dims=[-1])
    y_b = F.conv1d(F.pad(y_rev, (pad, pad), mode="reflect"), kernel, groups=D)
    y_fb = torch.flip(y_b, dims=[-1])

    return y_fb.squeeze(0).t()

# ========== Main pipeline using your functions ==========
@torch.no_grad()
def s2_lowpass_filter(
    X_or_Q: torch.Tensor,
    fs: float,
    fc: float,
    numtaps: int = 101,
    refine_mean: bool = False,
    return_q: bool = False,
    dtype: torch.dtype = torch.float64,
    device: torch.device | None = None,
):
    """
    Low-pass filter a trajectory on S^2 using user's q/x converters + log/exp implementations.

    Parameters
    ----------
    X_or_Q : (N,3) or (N,2)
        Trajectory either as unit vectors on S^2 (xyz) or spherical coords (theta, phi).
    fs : float
        Sampling rate (Hz) = 1/dt.
    fc : float
        Cutoff frequency (Hz), must satisfy 0 < fc < fs/2.
    numtaps : int
        Odd FIR length; larger -> narrower transition band (default 101).
    refine_mean : bool
        If True, do one more mean-refinement pass (useful for highly curved paths).
    return_q : bool
        If True, also return filtered (theta, phi).
    dtype, device
        Compute dtype/device; geometry is more stable with float64.

    Returns
    -------
    X_filt : (N,3) filtered unit vectors on S^2
    (optional) Q_filt : (N,2) filtered spherical coords if return_q=True or input was (N,2)
    """
    # Cast and convert input
    xx = torch.as_tensor(X_or_Q, dtype=dtype, device=device)
    xx = xx.squeeze(0)
    input_was_q = (xx.shape[-1] == 2)
    if xx.shape[-1] == 2:
        X = q_to_x(xx)                      # uses your function
    elif xx.shape[-1] == 3:
        X = xx
    else:
        raise ValueError("X_or_Q must have shape (N,2) or (N,3).")

    X = normalize_rows_torch(X)

    # Pole (Fréchet mean) using your vel_geo_0_sphere / exp_sphere
    m = frechet_mean_with_user_funcs(X)     # (3,)
    mN = m.unsqueeze(0).expand_as(X)        # (N,3)

    # Log-map to tangent space at m, FIR zero-phase, exp-map back
    V = vel_geo_0_sphere(mN, X)             # log_m(X)
    h = design_fir_lowpass_torch(fc, fs, numtaps=numtaps, device=X.device, dtype=X.dtype)
    V_lp = apply_zero_phase_fir_torch(V, h)
    X_filt = exp_sphere(mN, V_lp)           # exp_m(V_lp)
    X_filt = normalize_rows_torch(X_filt)

    # Optional one-step refinement for very curved data
    if refine_mean:
        m2 = frechet_mean_with_user_funcs(X_filt)
        m2N = m2.unsqueeze(0).expand_as(X_filt)
        V2 = vel_geo_0_sphere(m2N, X_filt)
        V2_lp = apply_zero_phase_fir_torch(V2, h)
        X_filt = exp_sphere(m2N, V2_lp)
        X_filt = normalize_rows_torch(X_filt)

    if return_q or input_was_q:
        Q_filt = x_to_q(X_filt)             # uses your function
        Q_filt = Q_filt.unsqueeze(0).float()
        return X_filt, Q_filt
    return X_filt



def step_geodesic_dist(X: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    """Compute per-step geodesic distances on S^2.
    X: (N,3) unit vectors
    returns: (N-1,) geodesic distances in radians
    """
    x1 = X[:-1]
    x2 = X[1:]
    dots = (x1 * x2).sum(dim=-1).clamp(-1.0 + eps, 1.0 - eps)
    return torch.arccos(dots)

def cumulative_arclength(d: torch.Tensor) -> torch.Tensor:
    """Cumulative arc length from per-step distances.
    d: (N-1,)
    returns: (N,) with s[0]=0
    """
    s = torch.zeros(d.numel() + 1, dtype=d.dtype, device=d.device)
    s[1:] = torch.cumsum(d, dim=0)
    return s



def rematch_speed_profile(X_filt: torch.Tensor, X_orig: torch.Tensor) -> torch.Tensor:
    """Reparametrize filtered path so that its cumulative length matches the original's timestamps.
    Returns a path with same length and strongly preserved per-frame geodesic distances.
    """
    if X_filt.ndim == 3 and X_filt.size(0) == 1:
        X_filt = X_filt.squeeze(0)
    if X_orig.ndim == 3 and X_orig.size(0) == 1:
        X_orig = X_orig.squeeze(0)

    # 2) Safety checks
    if not (X_filt.ndim == 2 and X_filt.size(-1) == 3):
        raise ValueError(f"X_filt must be (N,3); got {tuple(X_filt.shape)}")
    if not (X_orig.ndim == 2 and X_orig.size(-1) == 3):
        raise ValueError(f"X_orig must be (N,3); got {tuple(X_orig.shape)}")
    
    N = X_filt.shape[0]
    # cumulative lengths
    d_f = step_geodesic_dist(X_filt); s_f = cumulative_arclength(d_f)
    d_o = step_geodesic_dist(X_orig); s_o = cumulative_arclength(d_o)
    s_tot_f = s_f[-1].clamp_min(1e-12)

    # original arc positions projected to filtered arc domain
    # target positions on filtered path that correspond to original s_o
    s_target = (s_o / s_o[-1].clamp_min(1e-12)) * s_tot_f

    # allocate
    Y = torch.empty_like(X_filt)
    Y[0] = X_filt[0]; Y[-1] = X_filt[-1]

    j = 0
    for k in range(1, N-1):
        st = s_target[k]
        while j < N-2 and s_f[j+1] < st:
            j += 1
        seg_len = (s_f[j+1] - s_f[j]).clamp_min(1e-12)
        tau = (st - s_f[j]) / seg_len
        Y[k:k+1] = geodesic_sphere(X_filt[j:j+1], X_filt[j+1:j+2], tau.view(1,1))
    return Y

def resample_equal_arclength(X: torch.Tensor) -> torch.Tensor:
    """Resample a filtered S^2 path to equal geodesic spacing using SLERP between neighbors.
    X: (N,3) unit vectors (filtered)
    returns: (N,3) equal-spacing resampled path
    """
    N = X.shape[0]
    d = step_geodesic_dist(X)             # (N-1,)
    s = cumulative_arclength(d)           # (N,)
    s_tot = s[-1]
    if s_tot.item() == 0.0:
        return X.clone()

    # target arc positions
    s_target = torch.linspace(0.0, float(s_tot), N, device=X.device, dtype=X.dtype)

    # output
    Y = torch.empty_like(X)
    Y[0] = X[0]
    Y[-1] = X[-1]

    # for each target s, find segment [i, i+1] with s[i] <= s_t <= s[i+1]
    # and interpolate via SLERP using your geodesic_sphere
    j = 0
    for k in range(1, N-1):
        st = s_target[k]
        # advance j so that s[j] <= st <= s[j+1]
        while j < N-2 and s[j+1] < st:
            j += 1
        seg_len = (s[j+1] - s[j]).clamp_min(1e-12)
        tau = (st - s[j]) / seg_len
        # geodesic_sphere expects batched inputs; wrap with batch dim
        Y[k:k+1] = geodesic_sphere(X[j:j+1], X[j+1:j+2], tau.view(1,1))
    return Y



def lpf_then_rematch_q(q_traj, fs, fc, numtaps=101, refine_mean=True):
    """
    Apply S^2 low-pass to q-trajectory, then reparametrize to match original arc-length timing.
    Returns:
        q_filt_matched: (N,2)
        X_filt_matched: (N,3)
    """
    # Convert original to X for timing profile
    X_orig = q_to_x(q_traj)

    # Low-pass on S^2 (uses your functions internally)
    X_filt, q_filt = s2_lowpass_filter(
        q_traj, fs=fs, fc=fc, numtaps=numtaps,
        refine_mean=refine_mean, return_q=True
    )

    # Reparametrize filtered path to match original timing
    X_filt_matched = rematch_speed_profile(X_filt, X_orig)

    # Convert back to q if needed
    q_filt_matched = x_to_q(X_filt_matched)
    q_filt_matched = q_filt_matched.unsqueeze(0).float()

    return q_filt_matched, X_filt_matched
