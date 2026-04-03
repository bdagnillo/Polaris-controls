"""plot.py — load simulation data from data/ and display all figures.

Run after simulate.py has written the .npz files:
    python simulate.py   # produces data/*.npz
    python plot.py       # reads data/*.npz and shows figures

Figures that depend on a missing data file are silently skipped.
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


DATA_DIR = "data"


def _load(name: str):
    """Return dict from npz, or None if the file doesn't exist."""
    path = f"{DATA_DIR}/{name}.npz"
    try:
        return dict(np.load(path))
    except FileNotFoundError:
        print(f"[plot] {path} not found — skipping dependent figures.")
        return None


# ── load all datasets ─────────────────────────────────────────────────────────

cl      = _load("cl")
ol      = _load("ol")
dist_cl = _load("dist_cl")
dist_ol = _load("dist_ol")

sweep_results = []
for path in sorted(glob.glob(f"{DATA_DIR}/sweep_*.npz")):
    d = dict(np.load(path))
    # support both old key ("p") and new key ("phi")
    key = "phi" if "phi" in d else "p"
    sweep_results.append((d["t"], d[key], int(d.get("phi_deg", d.get("p_deg", 0)))))

if cl is None:
    raise SystemExit("[plot] cl.npz is required — run simulate.py first.")

# ── unpack closed-loop arrays ─────────────────────────────────────────────────

t    = cl["t"]
nu   = cl["nu"]
xi   = cl["xi"]

u, v, w         = nu[:, 0], nu[:, 1], nu[:, 2]
p, q, r         = nu[:, 3], nu[:, 4], nu[:, 5]
phi, theta, psi = nu[:, 6], nu[:, 7], nu[:, 8]
x, y, z         = nu[:, 9], nu[:, 10], nu[:, 11]

Vmag = cl["Vmag"]
Wmag = cl["Wmag"]

phi_ref  = float(cl["phi_ref"]) if "phi_ref" in cl else None
p_ref    = cl["p_ref"]
e_p      = cl["e_p"]
m_cx     = cl["m_cx"]
delta_cx = cl["delta_cx"]

alpha_array = cl["alpha"]
beta_array  = cl["beta"]
q_dyn       = cl["q_dyn"]

Fx, Fy, Fz = cl["Fx"], cl["Fy"], cl["Fz"]
Mx, My, Mz = cl["Mx"], cl["My"], cl["Mz"]

E_kin, E_pot, E_tot = cl["E_kin"], cl["E_pot"], cl["E_tot"]

# disturbance histories (shape N×3: columns = Mx, My, Mz)
dist_hist_cl = dist_cl["dist"] if dist_cl is not None and "dist" in dist_cl else None
dist_hist_ol = dist_ol["dist"] if dist_ol is not None and "dist" in dist_ol else None


def _overlay_disturbance(ax, t_d, dist_hist, color="gray", alpha=0.35):
    """Add disturbance roll moment as a shaded fill on a twin y-axis."""
    if dist_hist is None:
        return
    ax2 = ax.twinx()
    ax2.fill_between(t_d, dist_hist[:, 0], 0,
                     color=color, alpha=alpha, label="Disturbance Mx")
    ax2.set_ylabel("Disturbance Mx [N·m]", color=color)
    ax2.tick_params(axis="y", labelcolor=color)
    # keep zero centred so the fill is easy to read alongside the primary axis
    lim = max(abs(dist_hist[:, 0]).max(), 1e-3)
    ax2.set_ylim(-lim * 3, lim * 3)
    ax2.legend(loc="upper right")


# ── Fig 1: 3D trajectory ──────────────────────────────────────────────────────

fig = plt.figure()
ax = fig.add_subplot(111, projection="3d")
ax.plot(x, y, z, linewidth=1.5, label="Trajectory")

step = 50
scale = 10.0
for i in range(0, len(t), step):
    sp = np.sin(psi[i]); cp = np.cos(psi[i])
    st = np.sin(theta[i]); ct = np.cos(theta[i])
    sr = np.sin(phi[i]); cr = np.cos(phi[i])
    Cba_row0 = np.array([cp * ct,
                         cp * st * sr + sp * cr,
                         -cp * st * cr + sp * sr])
    ax.quiver(x[i], y[i], z[i],
              scale * Cba_row0[0], scale * Cba_row0[1], scale * Cba_row0[2],
              length=1.0, normalize=False)

ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
ax.set_title("3D Trajectory")
ax.legend()
ax.view_init(elev=-35, azim=-63)


# ── Fig 2: Kinematics ─────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Kinematics")

axes[0, 0].plot(t, u, label="u"); axes[0, 0].plot(t, v, label="v"); axes[0, 0].plot(t, w, label="w")
axes[0, 0].set(xlabel="Time [s]", ylabel="[m/s]", title="Linear velocities"); axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(t, x, label="x"); axes[0, 1].plot(t, y, label="y"); axes[0, 1].plot(t, z, label="z")
axes[0, 1].set(xlabel="Time [s]", ylabel="[m]", title="Positions"); axes[0, 1].legend(); axes[0, 1].grid(True)

axes[1, 0].plot(t, Vmag)
axes[1, 0].set(xlabel="Time [s]", ylabel="|V| [m/s]", title="Speed magnitude"); axes[1, 0].grid(True)

axes[1, 1].plot(t, E_kin, label="Kinetic"); axes[1, 1].plot(t, E_pot, label="Potential"); axes[1, 1].plot(t, E_tot, label="Total")
axes[1, 1].set(xlabel="Time [s]", ylabel="Energy [J]", title="Energy"); axes[1, 1].legend(); axes[1, 1].grid(True)


# ── Fig 3: Attitude & Aerodynamics ───────────────────────────────────────────

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Attitude & Aerodynamics")

axes[0, 0].plot(t, p, label="p"); axes[0, 0].plot(t, q, label="q"); axes[0, 0].plot(t, r, label="r")
axes[0, 0].set(xlabel="Time [s]", ylabel="[rad/s]", title="Angular rates"); axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(t, phi * 180/np.pi, label="φ")
axes[0, 1].plot(t, theta * 180/np.pi, label="θ")
axes[0, 1].plot(t, psi * 180/np.pi, label="ψ")
if phi_ref is not None:
    axes[0, 1].axhline(np.degrees(phi_ref), color="C0", linestyle="--", linewidth=1.0, label="φ_ref")
axes[0, 1].set(xlabel="Time [s]", ylabel="[deg]", title="Euler angles"); axes[0, 1].legend(); axes[0, 1].grid(True)

axes[0, 2].plot(t, Wmag)
axes[0, 2].set(xlabel="Time [s]", ylabel="|ω| [rad/s]", title="Angular rate magnitude"); axes[0, 2].grid(True)

axes[1, 0].plot(t, alpha_array * 180/np.pi)
axes[1, 0].set(xlabel="Time [s]", ylabel="α [deg]", title="Angle of attack"); axes[1, 0].grid(True)

axes[1, 1].plot(t, beta_array * 180/np.pi)
axes[1, 1].set(xlabel="Time [s]", ylabel="β [deg]", title="Sideslip angle"); axes[1, 1].grid(True)

axes[1, 2].plot(t, q_dyn)
axes[1, 2].set(xlabel="Time [s]", ylabel="q [Pa]", title="Dynamic pressure"); axes[1, 2].grid(True)


# ── Fig 4: Forces & Moments ───────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(9, 7))
fig.suptitle("Forces & Moments")

axes[0].plot(t, Fx, label="Fx"); axes[0].plot(t, Fy, label="Fy"); axes[0].plot(t, Fz, label="Fz")
axes[0].set(xlabel="Time [s]", ylabel="Forces [N]", title="Aero + thrust + weight"); axes[0].legend(); axes[0].grid(True)

axes[1].plot(t, Mx, label="Mx"); axes[1].plot(t, My, label="My"); axes[1].plot(t, Mz, label="Mz")
axes[1].set(xlabel="Time [s]", ylabel="Moments [N·m]", title="Roll, pitch, yaw moments"); axes[1].legend(); axes[1].grid(True)


# ── Fig 5: Roll control ───────────────────────────────────────────────────────

idx_zoom = (t >= 4.0) & (t <= 8.0)
fig, axes = plt.subplots(2, 3, figsize=(14, 8))
fig.suptitle("Roll Control")

axes[0, 0].plot(t, phi * 180/np.pi, label="φ")
if phi_ref is not None:
    axes[0, 0].axhline(np.degrees(phi_ref), color="k", linestyle="--", linewidth=1.0, label="φ_ref")
axes[0, 0].set(xlabel="Time [s]", ylabel="[deg]", title="Roll angle tracking"); axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(t, p * 180/np.pi, label="p")
axes[0, 1].plot(t, p_ref * 180/np.pi, "--", label="p_ref (outer loop cmd)")
axes[0, 1].set(xlabel="Time [s]", ylabel="[deg/s]", title="Inner-loop roll rate"); axes[0, 1].legend(); axes[0, 1].grid(True)

axes[0, 2].plot(t, e_p * 180/np.pi)
axes[0, 2].set(xlabel="Time [s]", ylabel="e_p [deg/s]", title="Roll rate error (inner)"); axes[0, 2].grid(True)

axes[1, 0].plot(t, m_cx)
axes[1, 0].set(xlabel="Time [s]", ylabel="m_cx [N·m]", title="Roll control moment"); axes[1, 0].grid(True)
if dist_hist_cl is not None:
    _overlay_disturbance(axes[1, 0], dist_cl["t"], dist_hist_cl)

axes[1, 1].plot(t, xi)
axes[1, 1].set(xlabel="Time [s]", ylabel="ξ", title="Integral state"); axes[1, 1].grid(True)

axes[1, 2].plot(phi * 180/np.pi, p * 180/np.pi)
if phi_ref is not None:
    axes[1, 2].axvline(np.degrees(phi_ref), color="k", linestyle="--", linewidth=1.0, label="φ_ref")
axes[1, 2].set(xlabel="φ [deg]", ylabel="p [deg/s]", title="Roll phase portrait"); axes[1, 2].legend(); axes[1, 2].grid(True)


# ── Fig 6: Comparisons (requires ol, dist_cl, dist_ol) ───────────────────────

if ol is not None or sweep_results or dist_cl is not None:
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Comparisons")

    # Roll angle: CL vs OL
    axes[0, 0].plot(t, phi * 180/np.pi, label="Closed-loop")
    if ol is not None:
        phi_ol = ol["nu"][:, 6]
        axes[0, 0].plot(ol["t"], phi_ol * 180/np.pi, "--", label="Open-loop")
    if phi_ref is not None:
        axes[0, 0].axhline(np.degrees(phi_ref), color="k", linestyle=":", linewidth=1.0, label="φ_ref")
    axes[0, 0].set(xlabel="Time [s]", ylabel="φ [deg]", title="Roll angle: CL vs OL"); axes[0, 0].legend(); axes[0, 0].grid(True)

    # Roll rate: CL vs OL
    axes[0, 1].plot(t, p * 180/np.pi, label="Closed-loop")
    if ol is not None:
        axes[0, 1].plot(ol["t"], ol["nu"][:, 3] * 180/np.pi, "--", label="Open-loop")
    axes[0, 1].set(xlabel="Time [s]", ylabel="p [deg/s]", title="Roll rate: CL vs OL"); axes[0, 1].legend(); axes[0, 1].grid(True)

    # Sweep
    for t_j, angle_j, deg in sweep_results:
        axes[1, 0].plot(t_j, angle_j * 180/np.pi, linewidth=1.5, label=f"φ_ref={deg}°")
    axes[1, 0].set(xlabel="Time [s]", ylabel="φ [deg]", title="Roll angle target sweep"); axes[1, 0].legend(); axes[1, 0].grid(True)

    # Disturbance rejection
    if dist_cl is not None:
        axes[1, 1].plot(dist_cl["t"], dist_cl["X"][:, 6] * 180/np.pi, label="Closed-loop")
    if dist_ol is not None:
        axes[1, 1].plot(dist_ol["t"], dist_ol["nu"][:, 6] * 180/np.pi, "--", label="Open-loop")
    if phi_ref is not None:
        axes[1, 1].axhline(np.degrees(phi_ref), color="k", linestyle=":", linewidth=1.0, label="φ_ref")
    axes[1, 1].set(xlabel="Time [s]", ylabel="φ [deg]", title="Disturbance rejection"); axes[1, 1].legend(); axes[1, 1].grid(True)
    _overlay_disturbance(axes[1, 1],
                         dist_cl["t"] if dist_cl is not None else dist_ol["t"],
                         dist_hist_cl if dist_hist_cl is not None else dist_hist_ol)


# ── Fig 7: Disturbance detail (requires dist_cl or dist_ol) ──────────────────

if dist_cl is not None or dist_ol is not None:
    fig7, ax7 = plt.subplots()
    if dist_ol is not None:
        ax7.plot(dist_ol["t"], dist_ol["nu"][:, 6] * 180/np.pi, "--", linewidth=1.5, label="Open-loop")
    if dist_cl is not None:
        ax7.plot(dist_cl["t"], dist_cl["X"][:, 6] * 180/np.pi, linewidth=1.5, label="Closed-loop")
    if phi_ref is not None:
        ax7.axhline(np.degrees(phi_ref), color="k", linestyle=":", linewidth=1.0, label="φ_ref")
    ax7.set_xlabel("Time [s]")
    ax7.set_ylabel("Roll angle φ [deg]")
    ax7.set_title("Rejection of a rolling impulse disturbance")
    ax7.grid(True)
    ax7.legend(loc="upper left")
    _overlay_disturbance(ax7,
                         dist_cl["t"] if dist_cl is not None else dist_ol["t"],
                         dist_hist_cl if dist_hist_cl is not None else dist_hist_ol)


plt.tight_layout()
plt.show()
