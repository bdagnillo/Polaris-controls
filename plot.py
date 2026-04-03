"""plot.py — load simulation data from data/ and display all figures.

Run after simulate2.py has written the .npz files:
    python simulate2.py   # produces data/*.npz
    python plot.py        # reads data/*.npz and shows figures
"""

import glob
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)


DATA_DIR = "data"


def _load(name: str) -> dict:
    path = f"{DATA_DIR}/{name}.npz"
    try:
        return dict(np.load(path))
    except FileNotFoundError:
        raise SystemExit(f"[plot] Missing {path} — run simulate2.py first.")


# ── load all datasets ─────────────────────────────────────────────────────────

cl       = _load("cl")
ol       = _load("ol")
dist_cl  = _load("dist_cl")
dist_ol  = _load("dist_ol")

# sweep files: data/sweep_10.npz, sweep_30.npz, sweep_60.npz
sweep_results = []
for path in sorted(glob.glob(f"{DATA_DIR}/sweep_*.npz")):
    d = dict(np.load(path))
    sweep_results.append((d["t"], d["p"], int(d["p_deg"])))

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

p_ref    = cl["p_ref"]
e_p      = cl["e_p"]
m_cx     = cl["m_cx"]
delta_cx = cl["delta_cx"]

alpha_array  = cl["alpha"]
beta_array   = cl["beta"]
q_dyn        = cl["q_dyn"]

Fx, Fy, Fz = cl["Fx"], cl["Fy"], cl["Fz"]
Mx, My, Mz = cl["Mx"], cl["My"], cl["Mz"]

E_kin, E_pot, E_tot = cl["E_kin"], cl["E_pot"], cl["E_tot"]

# open-loop
t_ol   = ol["t"]
nu_ol  = ol["nu"]
p_ol   = nu_ol[:, 3]
phi_ol = nu_ol[:, 6]

# disturbance
t_dist_cl = dist_cl["t"]
p_dist_cl = dist_cl["X"][:, 3]

t_dist_ol = dist_ol["t"]
p_dist_ol = dist_ol["nu"][:, 3]


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
    # First column of Cba.T = inertial x-axis expressed in body coords transposed
    # = body x-axis in inertial: [ct*cp, ct*sp*sr - st*cr, ct*sp*cr + st*sr] ... use shortcut:
    # nose_inertial = Cba(psi,theta,phi).T @ [1,0,0] = first row of Cba
    Cba_row0 = np.array([
        cp * ct,
        cp * st * sr + sp * cr,
        -cp * st * cr + sp * sr,
    ])
    nose_inertial = Cba_row0
    ax.quiver(x[i], y[i], z[i],
              scale * nose_inertial[0], scale * nose_inertial[1], scale * nose_inertial[2],
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

axes[0, 1].plot(t, phi * 180/np.pi, label="φ"); axes[0, 1].plot(t, theta * 180/np.pi, label="θ"); axes[0, 1].plot(t, psi * 180/np.pi, label="ψ")
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

axes[0, 0].plot(t, p * 180/np.pi, label="p"); axes[0, 0].plot(t, p_ref * 180/np.pi, "--", label="p_ref")
axes[0, 0].set(xlabel="Time [s]", ylabel="[deg/s]", title="Roll rate tracking"); axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(t[idx_zoom], p[idx_zoom] * 180/np.pi, label="p"); axes[0, 1].plot(t[idx_zoom], p_ref[idx_zoom] * 180/np.pi, "--", label="p_ref")
axes[0, 1].set(xlabel="Time [s]", ylabel="[deg/s]", title="Roll rate tracking (t=4–8 s)"); axes[0, 1].legend(); axes[0, 1].grid(True)

axes[0, 2].plot(t, e_p * 180/np.pi)
axes[0, 2].set(xlabel="Time [s]", ylabel="e_p [deg/s]", title="Roll rate error"); axes[0, 2].grid(True)

axes[1, 0].plot(t, m_cx)
axes[1, 0].set(xlabel="Time [s]", ylabel="m_cx [N·m]", title="Roll control moment"); axes[1, 0].grid(True)

axes[1, 1].plot(t, xi)
axes[1, 1].set(xlabel="Time [s]", ylabel="ξ", title="Integral state"); axes[1, 1].grid(True)

axes[1, 2].plot(phi * 180/np.pi, p * 180/np.pi)
axes[1, 2].set(xlabel="φ [deg]", ylabel="p [deg/s]", title="Roll phase portrait"); axes[1, 2].grid(True)


# ── Fig 6: Comparisons ────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 2, figsize=(11, 8))
fig.suptitle("Comparisons")

axes[0, 0].plot(t, p * 180/np.pi, label="Closed-loop"); axes[0, 0].plot(t_ol, p_ol * 180/np.pi, "--", label="Open-loop")
axes[0, 0].set(xlabel="Time [s]", ylabel="p [deg/s]", title="Roll rate: CL vs OL"); axes[0, 0].legend(); axes[0, 0].grid(True)

axes[0, 1].plot(t, phi * 180/np.pi, label="Closed-loop"); axes[0, 1].plot(t_ol, phi_ol * 180/np.pi, "--", label="Open-loop")
axes[0, 1].set(xlabel="Time [s]", ylabel="φ [deg]", title="Roll angle: CL vs OL"); axes[0, 1].legend(); axes[0, 1].grid(True)

for t_j, p_j, p_deg in sweep_results:
    axes[1, 0].plot(t_j, p_j * 180/np.pi, linewidth=1.5, label=f"p_ref={p_deg} °/s")
axes[1, 0].set(xlabel="Time [s]", ylabel="p [deg/s]", title="Command amplitude sweep"); axes[1, 0].legend(); axes[1, 0].grid(True)

axes[1, 1].plot(t_dist_ol, p_dist_ol * 180/np.pi, "--", label="Open-loop")
axes[1, 1].plot(t_dist_cl, p_dist_cl * 180/np.pi, label="Closed-loop PI")
axes[1, 1].set(xlabel="Time [s]", ylabel="p [deg/s]", title="Disturbance rejection"); axes[1, 1].legend(); axes[1, 1].grid(True)


# ── Fig 7: Disturbance detail ─────────────────────────────────────────────────

plt.figure()
plt.plot(t_dist_ol, p_dist_ol * 180/np.pi, "--", linewidth=1.5, label="Open-loop")
plt.plot(t_dist_cl, p_dist_cl * 180/np.pi, linewidth=1.5, label="Closed-loop PI")
plt.xlabel("Time [s]")
plt.ylabel("Roll rate p [deg/s]")
plt.title("Rejection of a rolling impulse disturbance")
plt.grid(True)
plt.legend()


plt.tight_layout()
plt.show()
