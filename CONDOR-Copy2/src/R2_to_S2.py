# Requirements:
# - numpy
# - matplotlib
import importlib
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from simple_parsing import ArgumentParser
from data_preprocessing.data_preprocessor import DataPreprocessor
from agent.utils.utils import lpf_demonstrations 
import os


# -----------------------------
# Geometry helpers
# -----------------------------
def alpha_for_target_coverage(r_max, coverage=0.8):
    """
    Compute scale alpha given desired sphere coverage (area fraction).
    coverage in (0,1). For example, coverage=0.8 means ~80% of sphere area.

    Relationship:
      coverage = 1 - f, where f is the spherical cap area fraction at the north pole.
      Z* = 1 - 2f = 2*coverage - 1
      alpha = r_max / sqrt((1+Z*)/(1-Z*))
    """
    if not (0.0 < coverage < 1.0):
        raise ValueError("coverage must be in (0,1).")
    Z_star = 2.0 * coverage - 1.0
    if not (-1.0 < Z_star < 1.0):
        raise ValueError("Computed Z* is out of (-1,1). Check coverage.")
    s = (1.0 + Z_star) / (1.0 - Z_star)
    return r_max / np.sqrt(s), Z_star


def inverse_stereographic_projection(xy, alpha=1.0, center=(0.0, 0.0), R=None):
    """
    Map 2D points to the unit sphere S^2 via inverse stereographic projection (north pole).

    Parameters
    ----------
    xy : (N, 2)
    alpha : float > 0
    center : (mu_x, mu_y)
    R : optional (3,3) rotation matrix; applied after mapping

    Returns
    -------
    pts : (N, 3) on S^2
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")
    if alpha <= 0:
        raise ValueError("alpha must be positive.")

    u = (xy[:, 0] - center[0]) / alpha
    v = (xy[:, 1] - center[1]) / alpha
    r2 = u * u + v * v
    denom = r2 + 1.0
    X = 2.0 * u / denom
    Y = 2.0 * v / denom
    Z = (r2 - 1.0) / denom
    pts = np.stack([X, Y, Z], axis=1)

    if R is not None:
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("R must be a 3x3 rotation matrix.")
        # Apply rotation to row-vectors: p' = p @ R^T
        pts = pts @ R.T
    return pts


# -----------------------------
# Rotation utilities (front alignment)
# -----------------------------
def hat(v):
    """Skew-symmetric matrix (hat operator): R^3 -> so(3)."""
    vx, vy, vz = np.asarray(v, dtype=float).reshape(3)
    return np.array([[ 0.0, -vz,  vy],
                     [ vz,  0.0, -vx],
                     [-vy,  vx,  0.0]])


def axis_angle(axis, theta):
    """Rotation matrix via Rodrigues around 'axis' by 'theta' radians."""
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        raise ValueError("axis must be non-zero.")
    x, y, z = axis / n
    c, s = np.cos(theta), np.sin(theta)
    C = 1.0 - c
    return np.array([
        [c + x*x*C,     x*y*C - z*s,  x*z*C + y*s],
        [y*x*C + z*s,   c + y*y*C,    y*z*C - x*s],
        [z*x*C - y*s,   z*y*C + x*s,  c + z*z*C]
    ])


def align_a_to_b(a, b):
    """
    Return rotation matrix R such that R @ a_hat = b_hat.
    Robust for a≈b and a≈-b.
    """
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        raise ValueError("Input vectors must be non-zero.")
    a = a / na; b = b / nb
    dot = float(np.dot(a, b))
    if np.isclose(dot, 1.0):
        return np.eye(3)
    if np.isclose(dot, -1.0):
        aux = np.array([1.0, 0.0, 0.0]) if abs(a[0]) < 0.9 else np.array([0.0, 1.0, 0.0])
        axis = np.cross(a, aux); axis /= np.linalg.norm(axis)
        return axis_angle(axis, np.pi)
    v = np.cross(a, b)
    s = np.linalg.norm(v)
    vx = np.array([[0.0, -v[2],  v[1]],
                   [v[2],  0.0, -v[0]],
                   [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * ((1.0 - dot) / (s ** 2))


def representative_direction(pts, mean_tol=1e-6):
    """
    Representative direction from points on S^2.
    Prefer mean; fallback to first principal component if mean ~ 0.
    """
    m = np.mean(pts, axis=0)
    nm = np.linalg.norm(m)
    if nm > mean_tol:
        return m / nm
    X = pts - np.mean(pts, axis=0)
    C = (X.T @ X) / max(len(pts) - 1, 1)
    vals, vecs = np.linalg.eigh(C)
    principal = vecs[:, np.argmax(vals)]
    return principal / np.linalg.norm(principal)


# -----------------------------
# so(3) <-> SO(3)
# -----------------------------
def vecToso3(axis, theta):
    """
    Convert axis & angle to skew-symmetric so(3) matrix.
    axis : (3,), normalized internally
    theta : float (rad)
    """
    axis = np.asarray(axis, dtype=float)
    n = np.linalg.norm(axis)
    if n == 0:
        raise ValueError("axis must be non-zero.")
    omega = (theta / n) * axis  # axis-angle vector
    return hat(omega)


def so3ToSO3(omega=None, axis=None, theta=None, eps=1e-12):
    """
    Exponential map from so(3) to SO(3).

    Modes:
      - so3ToSO3(omega=<axis-angle vector>)  # omega = axis * angle
      - so3ToSO3(axis=<axis>, theta=<angle>)
    """
    if omega is not None:
        omega = np.asarray(omega, dtype=float).reshape(3)
        theta_val = np.linalg.norm(omega)
        omega_hat = hat(omega)
    elif axis is not None and theta is not None:
        axis = np.asarray(axis, dtype=float).reshape(3)
        n = np.linalg.norm(axis)
        if n == 0:
            raise ValueError("axis must be non-zero.")
        theta_val = float(theta)
        omega_hat = hat((theta_val / n) * axis)
    else:
        raise ValueError("Provide either omega, or (axis and theta).")

    if theta_val < 1e-10:
        return np.eye(3)

    if theta_val < 1e-4:
        A = 1.0 - (theta_val**2) / 6.0 + (theta_val**4) / 120.0
        B = 0.5 - (theta_val**2) / 24.0 + (theta_val**4) / 720.0
    else:
        A = np.sin(theta_val) / theta_val
        B = (1.0 - np.cos(theta_val)) / (theta_val**2 + eps)

    R = np.eye(3) + A * omega_hat + B * (omega_hat @ omega_hat)
    return R


def axisAngleToSO3(axis, theta):
    """Convenience wrapper."""
    return so3ToSO3(axis=axis, theta=theta)


# -----------------------------
# Plot helpers
# -----------------------------
def set_axes_equal_3d(ax):
    """Set equal scale so that the sphere looks like a sphere."""
    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()
    x_range = abs(x_limits[1] - x_limits[0]); x_mid = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0]); y_mid = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0]); z_mid = np.mean(z_limits)
    R = 0.5 * max([x_range, y_range, z_range])
    ax.set_xlim3d([x_mid - R, x_mid + R])
    ax.set_ylim3d([y_mid - R, y_mid + R])
    ax.set_zlim3d([z_mid - R, z_mid + R])


def is_interactive_backend():
    backend = matplotlib.get_backend().lower()
    interactive = {"qt5agg", "tkagg", "macosx", "gtk3agg", "wxagg"}
    return any(b in backend for b in interactive)


def plot_on_sphere(
    pts,
    show_unit_sphere=True,
    connect_as_line=True,
    title=None,
    view=None,
    save_path=None,
    show_plot=True,
    # surface appearance
    surface_alpha=1.0,
    surface_color="lightblue",
    light_dir=(0.25, 0.35, 0.9),
    ambient=0.65,
    diffuse=0.45,
    add_wire=True,
    wire_alpha=0.15,
    wire_lw=0.3,
    # trajectory appearance
    traj_lift=1.003,
    traj_color="red",
    traj_linewidth=2.5,
    traj_zorder=10,
    # pole markers
    show_poles=True,
    pole_radius=1.06,
    pole_size=70
):
    """
    Plot mapped points on S^2 with custom per-vertex lighting (bright but 3D),
    lifted trajectory to avoid z-fighting, and optional pole markers.
    """
    pts = np.asarray(pts, dtype=float)
    if pts.ndim != 2 or pts.shape[1] != 3:
        raise ValueError("pts must have shape (N, 3).")

    fig = plt.figure(figsize=(7, 7))
    ax = fig.add_subplot(111, projection='3d')

    if show_unit_sphere:
        # Sphere mesh
        nu, nv = 160, 80
        u = np.linspace(0, 2*np.pi, nu)
        v = np.linspace(0, np.pi, nv)
        xs = np.outer(np.cos(u), np.sin(v))
        ys = np.outer(np.sin(u), np.sin(v))
        zs = np.outer(np.ones_like(u), np.cos(v))

        # Custom lighting (Lambertian): intensity = ambient + diffuse * max(0, n·l)
        N = np.stack([xs, ys, zs], axis=-1)
        N /= np.clip(np.linalg.norm(N, axis=-1, keepdims=True), 1e-12, None)
        L = np.asarray(light_dir, dtype=float); L /= np.linalg.norm(L)
        lambert = np.maximum(0.0, N @ L)
        intensity = np.clip(ambient + diffuse * lambert, 0.0, 1.0)
        base_rgb = np.asarray(matplotlib.colors.to_rgb(surface_color), dtype=float).reshape(1, 1, 3)
        facecolors = intensity[..., None] * base_rgb
        if surface_alpha < 1.0:
            fc = np.concatenate([facecolors, np.full((*facecolors.shape[:2], 1), surface_alpha)], axis=-1)
        else:
            fc = facecolors

        ax.plot_surface(xs, ys, zs, facecolors=fc, rstride=1, cstride=1,
                        linewidth=0, antialiased=True, shade=False, zorder=0)

        if add_wire:
            u_w = np.linspace(0, 2*np.pi, 24)
            v_w = np.linspace(0, np.pi, 12)
            xw = np.outer(np.cos(u_w), np.sin(v_w))
            yw = np.outer(np.sin(u_w), np.sin(v_w))
            zw = np.outer(np.ones_like(u_w), np.cos(v_w))
            ax.plot_wireframe(xw, yw, zw, color="k", linewidth=wire_lw, alpha=wire_alpha, zorder=1)

    # Lift trajectory slightly off the sphere to avoid z-fighting
    lift = 1.0 if (traj_lift is None or traj_lift <= 0) else float(traj_lift)
    pts_plot = pts * lift

    if connect_as_line and pts.shape[0] >= 2:
        ax.plot(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
                linewidth=traj_linewidth, color=traj_color, zorder=traj_zorder)
    else:
        ax.scatter(pts_plot[:, 0], pts_plot[:, 1], pts_plot[:, 2],
                   s=20, color=traj_color, zorder=traj_zorder)

    # Optional pole markers (lifted beyond surface so they are always visible)
    if show_poles:
        npole = np.array([0.0, 0.0, 1.0]) * pole_radius
        spole = np.array([0.0, 0.0, -1.0]) * pole_radius
        ax.scatter([npole[0]], [npole[1]], [npole[2]],
                   s=pole_size, color='blue', marker='o', label='North Pole', zorder=15)
        ax.scatter([spole[0]], [spole[1]], [spole[2]],
                   s=pole_size, color='green', marker='o', label='South Pole', zorder=15)

    # Camera
    if view is not None:
        elev, azim = view
        ax.view_init(elev=elev, azim=azim)

    ax.set_xlabel("X"); ax.set_ylabel("Y"); ax.set_zlabel("Z")
    if title is not None:
        ax.set_title(title)

    set_axes_equal_3d(ax)
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to: {save_path}")

    if is_interactive_backend():
        if show_plot:
            print("[INFO] Showing figure window...")
            plt.show()
        else:
            print("[INFO] Skipping GUI show (--no_show).")
            plt.close(fig)
    else:
        print("[INFO] Non-interactive backend; figure not shown (saved if save_path given).")
        plt.close(fig)


# -----------------------------
# End-to-end helper (with auto front alignment)
# -----------------------------
def map_and_plot_to_s2(xy, coverage=0.8, center=(0.0, 0.0), R=None,
                       show_unit_sphere=True, connect_as_line=True, title=None,
                       align_front=True, front_target=np.array([0.0, 0.0, 1.0]),
                       view=None, save_path="R2_to_S2_plot.png", show_plot=True,
                       **plot_kwargs):
    """
    Pipeline:
      1) Compute alpha for desired coverage.
      2) Map xy to S^2.
      3) Optional front alignment to 'front_target'.
      4) Optional user rotation R (applied after front alignment).
      5) Plot & save/show.

    Returns: pts_s2, alpha, Z_star, R_total
    """
    xy = np.asarray(xy, dtype=float)
    if xy.ndim != 2 or xy.shape[1] != 2:
        raise ValueError("xy must have shape (N, 2).")

    r = np.linalg.norm(xy - np.array(center), axis=1)
    r_max = float(np.max(r))
    if r_max == 0.0:
        raise ValueError("All points are identical; cannot define a radius.")

    alpha, Z_star = alpha_for_target_coverage(r_max, coverage=coverage)
    pts_s2 = inverse_stereographic_projection(xy, alpha=alpha, center=center, R=None)

    R_total = np.eye(3)
    if align_front:
        a_dir = representative_direction(pts_s2)
        R_front = align_a_to_b(a_dir, front_target)
        pts_s2 = pts_s2 @ R_front.T
        R_total = R_front @ R_total

    if R is not None:
        R = np.asarray(R, dtype=float)
        if R.shape != (3, 3):
            raise ValueError("R must be a 3x3 rotation matrix.")
        pts_s2 = pts_s2 @ R       # <--- consistent application
        R_total = R @ R_total

    ttl = title if title is not None else (
        f"Inverse stereographic projection (coverage≈{coverage*100:.0f}%, Z*={Z_star:.2f}, alpha={alpha:.3f})"
    )

    plot_on_sphere(
        pts_s2,
        show_unit_sphere=show_unit_sphere,
        connect_as_line=connect_as_line,
        title=ttl,
        view=view,
        save_path=save_path,
        show_plot=show_plot,
        **plot_kwargs
    )

    return pts_s2, alpha, Z_star, R_total


# -----------------------------
# Example usage / CLI
# -----------------------------
if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--params', type=str, default='2nd_order_2D', help='Parameter file name')
    parser.add_argument('--coverage', type=float, default=0.20, help='Sphere area usage in (0,1). 0.5=half sphere, 0.8=80%')
    parser.add_argument('--center_x', type=float, default=0.0, help='Center X to subtract before scaling')
    parser.add_argument('--center_y', type=float, default=0.0, help='Center Y to subtract before scaling')
    parser.add_argument('--no_align_front', action='store_true', help='Disable automatic front alignment')
    parser.add_argument('--view_elev', type=float, default=None, help='Camera elevation (deg)')
    parser.add_argument('--view_azim', type=float, default=None, help='Camera azimuth (deg)')
    parser.add_argument('--save_path', type=str, default='S2_Plot', help='Path to save the figure')
    parser.add_argument('--no_show', action='store_true', help='Do not show GUI window (headless)')
    args = parser.parse_args()

    # Load data
    Params = getattr(importlib.import_module('params.' + args.params), 'Params')
    data_preprocessor = DataPreprocessor(params=Params, verbose=True)
    data = data_preprocessor.run()

    demonstrations_norm = data['demonstrations train'][:, :, :, 0]
    demonstrations_norm = lpf_demonstrations(
        demonstrations_norm,
        fs=1.0,   # unit-spaced samples
        L_min=30,
        order=4
    )
    xy_traj = demonstrations_norm[0, :, :]  # (T, 2)

    # Optional: user-defined rotation (example)
    front_target = np.array([0.8, -1.2, 0.7])
    front_target /= np.linalg.norm(front_target)
    R_user = so3ToSO3(axis=front_target, theta=np.deg2rad(300.0))

    save_path = args.save_path + '/' + Params.selected_primitives_ids + '/R2_to_S2_plot.png'
    if not os.path.exists(save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

    pts_s2, alpha, Z_star, R_total = map_and_plot_to_s2(
        xy_traj,
        coverage=float(args.coverage),
        center=(args.center_x, args.center_y),
        R=R_user,                                 # applied after front alignment
        show_unit_sphere=True,
        connect_as_line=True,
        title=f"Trajectory on S^2 (~{args.coverage*100:.0f}% coverage)",
        align_front=not args.no_align_front,
        front_target=front_target,
        view=(args.view_elev, args.view_azim) if (args.view_elev is not None and args.view_azim is not None) else None,
        save_path=save_path,
        show_plot=not args.no_show,
        # plot style knobs (tweak as you like):
        surface_alpha=1.0,
        surface_color="lightblue",
        light_dir=(0.25, 0.35, 0.9),
        ambient=0.65,
        diffuse=0.45,
        add_wire=True,
        wire_alpha=0.6,
        wire_lw=0.3,
        traj_lift=1.003,
        traj_color="red",
        traj_linewidth=2.5,
        show_poles=True,
        pole_radius=1.06,
        pole_size=70
    )

    print(f"[INFO] alpha={alpha:.6f}, Z*={Z_star:.6f}")
    print(f"[INFO] Rotation matrix (R_total):\n{R_total}")
