import argparse
import os
import time
import numpy as np
from scipy.integrate import solve_ivp

from disturbance_models import load as load_disturbance
from params import RocketParams, ControlParams
from dynamics import (
    ode_cl, ode_open, ode_cl_disturb, ode_open_disturb,
    event_apogee, run_closed_loop_case,
)
from analysis import (
    compute_body_air_quantities,
    reconstruct_control_history,
    compute_force_moment_history,
)

_VALID_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")


def _setup(disturbance, ivp_method):
    rocket = RocketParams()
    control = ControlParams()
    print(f"[init] RocketParams: mass={rocket.mass} kg, R={rocket.R:.4f} m, "
          f"d={rocket.d:.4f} m, l={rocket.l:.4f} m")
    print(f"[init] Inertia: Jx={rocket.Jx} kg·m², Jy={rocket.Jy} kg·m², Jz={rocket.Jz} kg·m²")
    print(f"[init] ControlParams: K_phi={control.K_phi}, Kp_p={control.Kp_p}, Ki_p={control.Ki_p}, "
          f"Kp_q={control.Kp_q}, Kp_r={control.Kp_r}")

    rocket.generate_thrust_curve("AeroTech_HP-K535W.csv")
    print("[init] Thrust curve loaded")

    if disturbance is None:
        disturbance = load_disturbance("default")
    dist_desc = getattr(disturbance, "__module__", "unknown").split(".")[-1]
    print(f"[init] Disturbance model: {dist_desc}")
    print(f"[init] ODE solver method: {ivp_method}")

    nu0 = np.zeros(12)
    nu0[7] = 2 * np.pi / 180   #set initial pitch to 2 deg
    nu0[8] = 2 * np.pi / 180   #set initial yaw to 2 deg
    print(f"[init] Initial pitch={np.degrees(nu0[7]):.1f} deg, yaw={np.degrees(nu0[8]):.1f} deg")

    xi0 = 0.0  #initial integral state
    X0 = np.concatenate([nu0, [xi0]])

    t_eval = np.linspace(0.0, 20.0, 1000)
    phi_ref_main = 45 * np.pi / 180  #target roll angle [rad]
    print(f"[init] Roll angle target: {np.degrees(phi_ref_main):.1f} deg, t_span=0–20 s, {len(t_eval)} eval points")

    return rocket, control, nu0, X0, t_eval, phi_ref_main, disturbance, dist_desc


def _run_closed_loop(X0, t_eval, phi_ref_main, rocket, control, ivp_method):
    print("\n[1/5] Running main closed-loop simulation ...")
    _t0 = time.perf_counter()
    sol_cl = solve_ivp(
        fun=ode_cl,
        args=(phi_ref_main, rocket, control),
        t_span=(0.0, 20.0),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,
        method=ivp_method,
        rtol=1e-4,
        atol=1e-6,
        max_step=0.01,
    )
    print(f"[1/5] Closed-loop done in {time.perf_counter()-_t0:.1f}s: "
          f"t_final={sol_cl.t[-1]:.2f}s, steps={sol_cl.t.size}")
    if sol_cl.t_events[0].size > 0:
        print(f"      Apogee detected at t={sol_cl.t_events[0][0]:.2f}s")

    t = sol_cl.t
    X = sol_cl.y.T  # (n_times × 13): 12 rigid-body states + PI integral
    nu = X[:, 0:12]
    xi = X[:, 12]   # PI integrator accumulator
    u, v, w = nu[:, 0], nu[:, 1], nu[:, 2]
    p = nu[:, 3]
    x = nu[:, 9]

    Vmag = np.sqrt(u**2 + v**2 + w**2)
    Wmag = np.sqrt(p**2 + nu[:, 4]**2 + nu[:, 5]**2)
    print(f"      Peak altitude={x.max():.1f}m, max speed={Vmag.max():.1f}m/s, "
          f"max |p|={np.degrees(np.abs(p).max()):.2f}deg/s")

    print("[1/5] Reconstructing control history ...")
    p_ref, e_p, e_q, e_r, m_cx, m_cy, m_cz = reconstruct_control_history(t, X, control, rocket, phi_ref_main)
    # Canard angle implied by PI law; not stored during integration so reconstructed here
    delta_cx_hist = control.Kp_p * e_p + control.Ki_p * xi  # [rad]
    print(f"      Max roll torque={np.abs(m_cx).max():.4f} N·m, "
          f"max canard deflection={np.degrees(np.abs(delta_cx_hist).max()):.4f} deg")

    print("[1/5] Computing aero quantities ...")
    alpha_array, beta_array, q_dyn, Vbody_array = compute_body_air_quantities(t, nu, rocket.rho)
    print(f"      Max AoA={np.degrees(np.abs(alpha_array).max()):.2f}deg, "
          f"max q_dyn={q_dyn.max():.1f}Pa")

    print("[1/5] Computing force/moment history ...")
    Fx, Fy, Fz, Mx, My, Mz, alpha_log = compute_force_moment_history(t, nu, rocket)
    print(f"      Max |Fx|={np.abs(Fx).max():.1f}N, max |My|={np.abs(My).max():.3f}N·m")

    return dict(
        t=t, X=X, nu=nu, xi=xi, Vmag=Vmag, Wmag=Wmag, x=x,
        p_ref=p_ref, e_p=e_p, e_q=e_q, e_r=e_r,
        m_cx=m_cx, m_cy=m_cy, m_cz=m_cz, delta_cx_hist=delta_cx_hist,
        alpha_array=alpha_array, beta_array=beta_array, q_dyn=q_dyn, Vbody_array=Vbody_array,
        Fx=Fx, Fy=Fy, Fz=Fz, Mx=Mx, My=My, Mz=Mz,
    )


def _run_open_loop(nu0, t_eval, rocket, ivp_method, Vmag, x):
    print("\n[2/5] Running open-loop (no control) simulation ...")
    _t0 = time.perf_counter()
    sol_ol = solve_ivp(
        fun=ode_open,
        args=(rocket,),
        t_span=(0.0, 20.0),
        y0=nu0,
        t_eval=t_eval,
        events=event_apogee,
        method=ivp_method,
        rtol=1e-4,
        atol=1e-6,
        max_step=0.01,
    )
    print(f"[2/5] Open-loop done in {time.perf_counter()-_t0:.1f}s: t_final={sol_ol.t[-1]:.2f}s")

    E_kin = 0.5 * rocket.mass * Vmag**2
    E_pot = rocket.mass * rocket.g * x
    E_tot = E_kin + E_pot
    print(f"      Peak total energy={E_tot.max():.1f}J  (KE={E_kin.max():.1f}J, PE={E_pot.max():.1f}J)")

    return sol_ol.t, sol_ol.y.T, E_kin, E_pot, E_tot


def _run_phi_sweep(X0, t_eval, rocket, control, ivp_method):
    print("\n[3/5] Running roll angle target sweep (phi_ref = 15, 45, 90 deg) ...")
    sweep_results = []
    for phi_deg in [15, 45, 90]:
        phi_ref_j = phi_deg * np.pi / 180
        print(f"  Sweep: phi_ref={phi_deg} deg ...")
        sol_j = run_closed_loop_case(X0, t_eval, rocket, control, phi_ref_j, ivp_method)
        phi_j = sol_j.y.T[:, 6]
        print(f"         done, t_final={sol_j.t[-1]:.2f}s, final phi={np.degrees(phi_j[-1]):.2f}deg")
        sweep_results.append((sol_j.t, phi_j, phi_deg))
    print("[3/5] Sweep done")
    return sweep_results


def _run_disturbance(X0, nu0, t_eval, phi_ref_main, disturbance, dist_desc, rocket, control, ivp_method):
    print(f"\n[4/5] Running disturbance rejection (model: {dist_desc}) ...")

    _t0 = time.perf_counter()
    sol_dist_cl = solve_ivp(
        fun=ode_cl_disturb,
        args=(phi_ref_main, disturbance, rocket, control),
        t_span=(0.0, 20.0),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,
        method=ivp_method,
        rtol=1e-4,
        atol=1e-6,
        max_step=0.01,
    )
    print(f"      Disturbed CL done in {time.perf_counter()-_t0:.1f}s: t_final={sol_dist_cl.t[-1]:.2f}s")

    _t0 = time.perf_counter()
    sol_dist_ol = solve_ivp(
        fun=ode_open_disturb,
        args=(disturbance, rocket),
        t_span=(0.0, 20.0),
        y0=nu0,
        t_eval=t_eval,
        events=event_apogee,
        method=ivp_method,
        rtol=1e-4,
        atol=1e-6,
        max_step=0.01,
    )
    print(f"      Disturbed OL done in {time.perf_counter()-_t0:.1f}s: t_final={sol_dist_ol.t[-1]:.2f}s")
    print("[4/5] Disturbance done")

    return sol_dist_cl.t, sol_dist_cl.y.T, sol_dist_ol.t, sol_dist_ol.y.T


def _save_results(cl, ol, sweep_results, dist, disturbance, phi_ref_main):
    t_ol, nu_ol, E_kin, E_pot, E_tot = ol
    t_dist_cl, X_dist_cl, t_dist_ol, nu_dist_ol = dist

    print("\n[5/5] Saving simulation data ...")
    os.makedirs("data", exist_ok=True)

    np.savez("data/cl.npz",
        t=cl["t"], X=cl["X"], nu=cl["nu"], xi=cl["xi"],
        phi_ref=np.array(phi_ref_main),
        p_ref=cl["p_ref"], e_p=cl["e_p"], e_q=cl["e_q"], e_r=cl["e_r"],
        m_cx=cl["m_cx"], m_cy=cl["m_cy"], m_cz=cl["m_cz"],
        delta_cx=cl["delta_cx_hist"],
        alpha=cl["alpha_array"], beta=cl["beta_array"], q_dyn=cl["q_dyn"], Vbody=cl["Vbody_array"],
        Fx=cl["Fx"], Fy=cl["Fy"], Fz=cl["Fz"], Mx=cl["Mx"], My=cl["My"], Mz=cl["Mz"],
        E_kin=E_kin, E_pot=E_pot, E_tot=E_tot,
        Vmag=cl["Vmag"], Wmag=cl["Wmag"],
    )
    np.savez("data/ol.npz", t=t_ol, nu=nu_ol)

    for t_j, phi_j, phi_deg in sweep_results:
        np.savez(f"data/sweep_{phi_deg}.npz", t=t_j, phi=phi_j, phi_deg=np.array(phi_deg))

    # Re-evaluate disturbance over each time grid; the ODE doesn't store it internally
    dist_hist_cl = np.array([disturbance(ti) for ti in t_dist_cl])  # (N, 3)
    dist_hist_ol = np.array([disturbance(ti) for ti in t_dist_ol])  # (N, 3)

    np.savez("data/dist_cl.npz", t=t_dist_cl, X=X_dist_cl, dist=dist_hist_cl)
    np.savez("data/dist_ol.npz", t=t_dist_ol, nu=nu_dist_ol, dist=dist_hist_ol)

    print("[5/5] Data saved to data/")


def simulate_rocket_trajectory(disturbance=None, ivp_method="BDF"):
    print("=== Polaris Roll-Control Simulation ===")

    rocket, control, nu0, X0, t_eval, phi_ref_main, disturbance, dist_desc = _setup(disturbance, ivp_method)

    cl = _run_closed_loop(X0, t_eval, phi_ref_main, rocket, control, ivp_method)
    t_ol, nu_ol, E_kin, E_pot, E_tot = _run_open_loop(nu0, t_eval, rocket, ivp_method, cl["Vmag"], cl["x"])
    sweep_results = _run_phi_sweep(X0, t_eval, rocket, control, ivp_method)
    dist = _run_disturbance(X0, nu0, t_eval, phi_ref_main, disturbance, dist_desc, rocket, control, ivp_method)

    _save_results(cl, (t_ol, nu_ol, E_kin, E_pot, E_tot), sweep_results, dist, disturbance, phi_ref_main)

    print("\n=== Simulation complete — run plot.py to visualise ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Polaris roll-control simulation")
    parser.add_argument(
        "--disturbance", "-d",
        default="default",
        metavar="MODEL",
        help="Disturbance model name (must match a file in disturbance_models/). "
             "Default: 'default'."
    )
    parser.add_argument(
        "--method", "-m",
        default="BDF",
        choices=_VALID_METHODS,
        metavar="METHOD",
        help=f"scipy solve_ivp method. Choices: {', '.join(_VALID_METHODS)}. Default: BDF."
    )
    args = parser.parse_args()
    dist_fn = load_disturbance(args.disturbance)
    simulate_rocket_trajectory(disturbance=dist_fn, ivp_method=args.method)
