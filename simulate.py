import argparse
import os
import time
import numpy as np
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from disturbance_models import load as load_disturbance

#Parameters
@dataclass
class RocketParams:
    rho: float = 1.225 #TBD
    mass: float = 5 # kg TBD
    g: float = 9.81 # m/s^2

    # all lengths in meters
    R: float = 0.0762 / 2 # rocket radius
    l: float = 1.9558 # total length maybe
    l_body: float = 0.711
    K_body: float = 1.1

    # fin geometry
    s_fin: float = 0.0635 #fin span
    gamma_c: float = 0.2915
    t_fin: float = 0.00508 #thickness
    C_r: float = 0.0889 #root chord
    C_t: float = 0.0889 - 0.0381 #tip
    c_barre: float = 0.0716
    cant: float = 0.1 * np.pi / 180

    Jx: float = 0.02
    Jy: float = 0.2 #inertia
    Jz: float = 0.2

    Tmax: float = 10.0 #max thrust
    tburn: float = 2.5
    fade_time: float = 15.0

    r_ag_x: float = 0.134

    @property 
    def A_ref(self) -> float:
        return np.pi * self.R**2

    @property
    def d(self) -> float: #rocket diameter
        return 2 * self.R

    @property
    def A_fin(self) -> float: #fin area
        return (0.0889 + (0.0889 - 0.0381)) * self.s_fin / 2

    @property
    def y_mac(self) -> float: #location related to mac
        return (self.s_fin / 3) * ((self.C_r + 2 * self.C_t) / (self.C_r + self.C_t))

    @property
    def K_TB(self) -> float: #correction factor
        return 1 + self.R / (self.R + self.s_fin)

    def __post_init__(self):
        self._J    = np.diag([self.Jx, self.Jy, self.Jz])
        self._Mmat = np.block([
            [self.mass * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)),      self._J          ]
        ])

    @property
    def J(self) -> np.ndarray: #inertia matrix
        return self._J

    @property
    def Mmat(self) -> np.ndarray: #combined mass and inertia matrix 6x6
        return self._Mmat
    
    
    def generate_thrust_curve(self, file: str) -> callable:
        thrust = np.loadtxt(file, delimiter=',', skiprows=5)
        self.thrust = lambda t: np.interp(t, thrust[:, 0], thrust[:, 1])


@dataclass
class ControlParams:
    K_phi: float = 2.0   #outer-loop P gain: roll angle error → roll rate command [rad/s per rad]

    Kp_p: float = 0.2357 #inner-loop P gain for roll rate
    Ki_p: float = 0.5656 #inner-loop I gain for roll rate

    K_theta: float = 2.0 #gain converts pitch angle error to pitch rate command
    K_psi: float = 2.0 #same for yaw

    Kp_q: float = 3.0 #proportional gain for pitch rate
    Kp_r: float = 3.0 #proportional gain for yaw rate



#Rotation helpers
def hat(vec: np.ndarray) -> np.ndarray: #cross product
    return np.array([
        [0.0, -vec[2], vec[1]],
        [vec[2], 0.0, -vec[0]],
        [-vec[1], vec[0], 0.0]
    ])


def Cba(psi: float, theta: float, phi: float) -> np.ndarray: #transform vector between frames
    return np.array([
        [
            np.cos(psi) * np.cos(theta),
            np.cos(psi) * np.sin(theta) * np.sin(phi) + np.sin(psi) * np.cos(phi),
            -np.cos(psi) * np.sin(theta) * np.cos(phi) + np.sin(psi) * np.sin(phi),
        ],
        [
            -np.sin(psi) * np.cos(theta),
            -np.sin(psi) * np.sin(theta) * np.sin(phi) + np.cos(psi) * np.cos(phi),
            np.sin(psi) * np.sin(theta) * np.cos(phi) + np.cos(psi) * np.sin(phi),
        ],
        [
            np.sin(theta),
            -np.cos(theta) * np.sin(phi),
            np.cos(theta) * np.cos(phi),
        ],
    ])


def Sba(phi: float, theta: float) -> np.ndarray: #L matrix: L @ [phi_dot, theta_dot, psi_dot] = [p, q, r]
    return np.array([
        [1.0,  0.0,              -np.sin(theta)             ],
        [0.0,  np.cos(phi),       np.sin(phi) * np.cos(theta)],
        [0.0, -np.sin(phi),       np.cos(phi) * np.cos(theta)]
    ])


def C1(delta: float) -> np.ndarray: #rotation matrix about x axis
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(delta), -np.sin(delta)], 
        [0.0, np.sin(delta), np.cos(delta)]
    ])



def canard_torque(delta_c: float, p: RocketParams, U: float, rho: float) -> float:
    """Convert canard deflection angle (rad) to roll torque (N·m).

    Uses the same lift-slope model as the fin cant roll moment in forces_and_moments.
    The roll moment coefficient per radian of deflection is:
        C_l_delta = 3 * (y_mac + R) * lift_slope / d
    so that  M_roll = 0.5 * rho * A_ref * d * C_l_delta * delta_c * U^2.

    TODO: validate / replace with measured or CFD-derived C_l_delta for the
    actual canard geometry once available.
    """
    Mach = U / 340.0
    beta_m = np.sqrt(max(1.0 - Mach**2, 1e-6))
    lift_slope = (
        2.0 * np.pi * (p.s_fin**2 / p.A_ref) /
        (1.0 + np.sqrt(1.0 + (beta_m * p.s_fin / (p.A_fin * np.cos(p.gamma_c)))**2))
    )
    C_l_delta = 3.0 * (p.y_mac + p.R) * lift_slope / p.d
    return 0.5 * rho * p.A_ref * p.d * C_l_delta * delta_c * U**2


#Forces
def forces_and_moments(
    t: float,
    psi: float,
    theta: float, #Euler angles
    phi: float,
    nu: np.ndarray,
    m_c: np.ndarray, #control moment vector
    p: RocketParams
) -> tuple[np.ndarray, float]:
    DCM = Cba(psi, theta, phi) #build orientation matrix from euler angles

    v_air_b = np.real(DCM @ nu[0:3]) #take translational velocity and rotate to give air relative velocity
    delta = np.arctan2(v_air_b[2], v_air_b[1])

    Ccb = C1(-delta) #create  x rotation matrix
    Cca = Ccb @ DCM

    Ux, Uy, Uz = v_air_b #unpack body frame velocity components
    U = np.sqrt(Ux**2 + Uy**2 + Uz**2) + 1e-6
    alpha = np.arctan2(Uz, Ux) #angle of attack
    Mach = U / 340.0
    beta = np.sqrt(max(1.0 - Mach**2, 1e-6)) 

    p_roll = nu[3]
    q_pitch = nu[4]
    r_ag = np.array([p.r_ag_x, 0.0, 0.0])

    C_N_nose = (2.0 / p.A_ref) * (p.A_ref * np.sin(-alpha)) #Normal force coefficient nose
    C_N_body = p.K_body * ((p.d * p.l_body) / p.A_ref) * np.sin(-alpha) ** 2 #body
    C_N_fins = -alpha * p.K_TB * (3.0 / 2.0) * ( #fin
        2.0 * np.pi * (p.s_fin**2 / p.A_ref) /
        (1.0 + np.sqrt(1.0 + (beta * p.s_fin / (p.A_fin * np.cos(p.gamma_c)))**2))
    )

    C_f = 0.007 #skin friction
    C_f_c = C_f * (1.0 - 0.1 * Mach**2)
    C_d_friction = C_f_c * (
        ((1.0 + 1.0 / (2.0 * (p.l / p.d))) * np.pi * p.d * p.l +
         (1.0 + 2.0 * p.t_fin / p.c_barre) * 6.0 * p.A_fin) / p.A_ref
    )

    if Mach < 1.0:
        C_d_fins = (1.0 - Mach**2) ** (-0.417) - 0.88 + 0.13 * Mach**2
    else:
        C_d_fins = 0.0

    C_d_0 = (   #baseline drag coef
        3.0 * p.t_fin * p.s_fin * C_d_fins / p.A_ref
        + (np.pi * 0.1 * p.R**2) / p.A_ref
        + C_d_friction
    )

    lift_slope_term = (
        2.0 * np.pi * (p.s_fin**2 / p.A_ref) /
        (1.0 + np.sqrt(1.0 + (beta * p.s_fin / (p.A_fin * np.cos(p.gamma_c)))**2))
    )

    C_l_f = 3.0 * (p.y_mac + p.R) * lift_slope_term * p.cant / p.d #rolling moment coef contribution due to fin cant

    geom_term = (
        0.5 * (p.C_r + p.C_t) * p.R**2 * p.s_fin
        + (p.C_r + 2.0 * p.C_t) * (1.0 / 3.0) * p.R * p.s_fin**2
        + (p.C_r + 3.0 * p.C_t) * (1.0 / 12.0) * p.s_fin**3
    )

    # Floor for damping denominators — prevents 1/U and 1/U² singularities near apogee
    # where aerodynamic damping is negligible anyway (dynamic pressure → 0)
    U_damp = max(U, 5.0)

    C_l_d = 3.0 * p_roll * (2.0 * np.pi / beta) * geom_term / (p.A_ref * p.d * U_damp) #roll damping coefficient

    N = 0.5 * p.rho * p.A_ref * (C_N_nose + C_N_body + C_N_fins) * U**2 #normal force
    D = 0.5 * p.rho * p.A_ref * C_d_0 * U**2 #drag

    m_l = np.array([  # aero roll moment vector
        0.5 * p.rho * p.A_ref * p.d * (C_l_f - C_l_d) * U**2,
        0.0,
        0.0
    ])

    f_g = np.array([-p.mass * p.g, 0.0, 0.0]) #weight
    f_d = Cca.T @ np.array([-D, 0.0, 0.0]) #drag rotated into dynamics frane

    if not hasattr(p, 'thrust'):
        raise ValueError("No thrust curve provided in RocketParams. Set p.thrust to a callable.")
    thrust = p.thrust(t)
    f_t = DCM.T @ np.array([thrust, 0.0, 0.0]) #thrust rotated into dynamics frame

    f_N = Cca.T @ np.array([0.0, N, 0.0])
    pitch_moment = hat(r_ag) @ DCM @ f_N

    C_damp_body = 0.55 * ((p.l**4 * p.R) / (p.A_ref * p.d)) * (np.abs(q_pitch) * q_pitch / U_damp**2)
    d_fin_cg = 0.825 - 0.506                                                         #pitch damping
    C_damp_fin = 0.6 * 3.0 * p.A_fin * d_fin_cg**3 * np.abs(q_pitch) * q_pitch / (p.A_ref * p.d * U_damp**2)

    m_damping = Ccb.T @ np.array([
        0.0,
        0.5 * p.rho * p.A_ref * (-C_damp_fin - C_damp_body) * p.d * U**2,
        0.0
    ])

    if not np.all(np.isfinite(m_damping)):
        m_damping = np.zeros(3)

    f_total = f_g + f_t + f_d #gravity,thrust,drag
    m_total = m_l + pitch_moment + m_damping + m_c 

    FM = np.concatenate([f_total, m_total]) #6 component vector
    return FM, alpha



#Dynamics
def ode_rocket(t: float, nu: np.ndarray, m_c: np.ndarray, p: RocketParams) -> np.ndarray: #m_c=control moment from controller


    v = nu[0:3] #vx, vy, vz inertial-frame translational velocity
    omega = nu[3:6] #p,q,r angular velocity vector in body coordinates
    phi = nu[6]
    theta = nu[7]#pulling out euler angles
    psi = nu[8]

    dangles = np.linalg.solve(Sba(phi, theta), omega) #dangles=vector of euler angle rates
    dpos = v #position derivative

    gyro = np.concatenate([np.zeros(3), hat(omega) @ (p.J @ omega)]) #last 3 elements=rotational gyroscopic term
    rhs = forces_and_moments(t, psi, theta, phi, nu, m_c, p)[0] - gyro #returns vector FM then subtract gyro
    dnu_dyn = np.linalg.solve(p.Mmat, rhs) #gives 3 translational accelerations and 3 angular accelerations

    return np.concatenate([dnu_dyn, dangles, dpos]) #complete derivative of 12 state vector


def ode_cl(t: float, X: np.ndarray, phi_ref: float, p: RocketParams, c: ControlParams) -> np.ndarray:
    nu = X[0:12] #12 rocket states
    xi = X[12]   #integral state for inner-loop roll-rate PI

    phi       = nu[6]
    roll_rate = nu[3]
    pitch_rate = nu[4]
    yaw_rate   = nu[5]
    theta = nu[7]
    psi   = nu[8]

    v_body = Cba(psi, theta, phi) @ nu[0:3]
    U = np.linalg.norm(v_body) + 1e-6

    # Outer loop: roll angle error → roll rate command
    p_ref = c.K_phi * (phi_ref - phi)

    # Pitch/yaw outer loops (unchanged)
    q_ref = c.K_theta * (0.0 - theta)
    r_ref = c.K_psi   * (0.0 - psi)

    e_p = p_ref - roll_rate   #inner-loop roll rate error
    e_q = q_ref - pitch_rate
    e_r = r_ref - yaw_rate

    dxi = e_p #integrator driven by roll rate error

    delta_cx = c.Kp_p * e_p + c.Ki_p * xi  #canard deflection [rad]
    m_cx = canard_torque(delta_cx, p, U, p.rho)
    m_cy = c.Kp_q * e_q
    m_cz = c.Kp_r * e_r

    m_c = np.array([m_cx, m_cy, m_cz])
    dnu = ode_rocket(t, nu, m_c, p)

    return np.concatenate([dnu, [dxi]])


def ode_open(t: float, nu: np.ndarray, p: RocketParams) -> np.ndarray: #no control
    return ode_rocket(t, nu, np.zeros(3), p)


def ode_cl_disturb(
    t: float,
    X: np.ndarray,
    phi_ref: float,   #target roll angle [rad]
    disturbance,      #callable: t -> 3-vector [Mx, My, Mz] N·m
    p: RocketParams,
    c: ControlParams
) -> np.ndarray:
    nu = X[0:12]
    xi = X[12]

    phi       = nu[6]
    roll_rate = nu[3]
    theta = nu[7]; psi = nu[8]
    v_body = Cba(psi, theta, phi) @ nu[0:3]
    U = np.linalg.norm(v_body) + 1e-6

    p_ref = c.K_phi * (phi_ref - phi)

    e_p = p_ref - roll_rate
    dxi = e_p

    delta_cx = c.Kp_p * e_p + c.Ki_p * xi
    m_cx = canard_torque(delta_cx, p, U, p.rho)
    m_c = np.array([m_cx, 0.0, 0.0]) + disturbance(t)

    dnu = ode_rocket(t, nu, m_c, p)
    return np.concatenate([dnu, [dxi]])


def ode_open_disturb(t: float, nu: np.ndarray, disturbance, p: RocketParams) -> np.ndarray:
    m_c = disturbance(t)
    return ode_rocket(t, nu, m_c, p)


_APOGEE_MIN_ALT = 50.0  # [m] minimum altitude before apogee detection activates

def event_apogee(t: float, X: np.ndarray, *_) -> float: #used as an event function
    nu = X[0:12]
    h  = nu[9]   # inertial x = altitude
    vx = nu[0]   # velocity whose zero crossing indicates apogee
    # Return positive (no event) until the rocket has left the pad;
    # once above the threshold hand off to vx so direction=-1 triggers correctly.
    return vx if h > _APOGEE_MIN_ALT else 1.0


event_apogee.terminal = True #stop integrating
event_apogee.direction = -1 #trigger event when function crosses zero in negative direc



#Analysis helpers
def compute_body_air_quantities(
    t: np.ndarray,
    nu_hist: np.ndarray, #history os state vectors over time
    rho: float = 1.225
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    alpha_array = np.zeros_like(t) #arrays same length as t
    beta_array = np.zeros_like(t)
    q_dyn = np.zeros_like(t)
    Vbody_array = np.zeros_like(t)

    for k in range(len(t)):
        phi = nu_hist[k, 6]
        theta = nu_hist[k, 7] #extracting euler angles at time step k
        psi = nu_hist[k, 8]

        DCM = Cba(psi, theta, phi) #construct direc cosine matrix
        v_body = DCM @ nu_hist[k, 0:3] #rotate velocity into body frame

        Ux, Vy, Uz = v_body #unpack body velocity
        Vmag = np.linalg.norm(v_body) #total airspeed magnitude

        Vbody_array[k] = Vmag #save magnitude at this instant for plotting
        alpha_array[k] = np.arctan2(Uz, Ux) #AoA
        beta_array[k] = np.arcsin(np.clip(Vy / (Vmag + 1e-6), -1.0, 1.0)) #sideslip angle
        q_dyn[k] = 0.5 * rho * Vmag**2 #dynamic pressure

    return alpha_array, beta_array, q_dyn, Vbody_array


def reconstruct_control_history(t: np.ndarray, X: np.ndarray, c: ControlParams,
                                rocket: RocketParams, phi_ref: float):
    nu = X[:, 0:12]
    xi = X[:, 12]

    phi    = nu[:, 6]
    p_rate = nu[:, 3]
    q_rate = nu[:, 4]
    r_rate = nu[:, 5]
    theta  = nu[:, 7]
    psi    = nu[:, 8]

    # Mirror the cascade structure in ode_cl
    p_ref = c.K_phi * (phi_ref - phi)   #outer loop: angle → rate command
    q_ref = c.K_theta * (0.0 - theta)
    r_ref = c.K_psi   * (0.0 - psi)

    e_p = p_ref - p_rate
    e_q = q_ref - q_rate
    e_r = r_ref - r_rate

    delta_cx = c.Kp_p * e_p + c.Ki_p * xi
    U_hist = np.array([
        np.linalg.norm(Cba(psi[k], theta[k], phi[k]) @ nu[k, 0:3]) + 1e-6
        for k in range(len(t))
    ])
    m_cx = np.array([canard_torque(delta_cx[k], rocket, U_hist[k], rocket.rho) for k in range(len(t))])
    m_cy = c.Kp_q * e_q
    m_cz = c.Kp_r * e_r

    return p_ref, e_p, e_q, e_r, m_cx, m_cy, m_cz


def compute_force_moment_history(t: np.ndarray, nu: np.ndarray, rocket: RocketParams):
    Fx = np.zeros_like(t)
    Fy = np.zeros_like(t)
    Fz = np.zeros_like(t)
    Mx = np.zeros_like(t)
    My = np.zeros_like(t)
    Mz = np.zeros_like(t)
    alpha_log = np.zeros_like(t)

    for k in range(len(t)):
        phi = nu[k, 6]
        theta = nu[k, 7]
        psi = nu[k, 8]
        m_c_k = np.zeros(3) #set control moment to zero

        FM_k, alpha_k = forces_and_moments(
            t[k], psi, theta, phi, nu[k, :], m_c_k, rocket
        )

        f_k = FM_k[0:3] #force vector 
        m_k = FM_k[3:6] #moment vector

        Fx[k], Fy[k], Fz[k] = f_k
        Mx[k], My[k], Mz[k] = m_k
        alpha_log[k] = alpha_k #at teach time, store AoA

    return Fx, Fy, Fz, Mx, My, Mz, alpha_log


def run_closed_loop_case(
    X0: np.ndarray,
    t_eval: np.ndarray,
    rocket: RocketParams,
    control: ControlParams,
    phi_ref: float,
    ivp_method: str = "BDF"
):
    return solve_ivp(
        fun=ode_cl,
        args=(phi_ref, rocket, control),
        t_span=(t_eval[0], t_eval[-1]),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,
        method=ivp_method,
        rtol=1e-4,
        atol=1e-6,
        max_step=0.01,
    )



#Main simulation
_VALID_METHODS = ("RK45", "RK23", "DOP853", "Radau", "BDF", "LSODA")

def simulate_rocket_trajectory(disturbance=None, ivp_method="BDF"):
    print("=== Polaris Roll-Control Simulation ===")

    rocket = RocketParams()
    control = ControlParams()
    print(f"[init] RocketParams: mass={rocket.mass} kg, R={rocket.R:.4f} m, "
          f"d={rocket.d:.4f} m, l={rocket.l:.4f} m")
    print(f"[init] Inertia: Jx={rocket.Jx} kg·m², Jy={rocket.Jy} kg·m², Jz={rocket.Jz} kg·m²")
    print(f"[init] ControlParams: K_phi={control.K_phi}, Kp_p={control.Kp_p}, Ki_p={control.Ki_p}, "
          f"Kp_q={control.Kp_q}, Kp_r={control.Kp_r}")

    rocket.generate_thrust_curve("AeroTech_HP-K535W.csv") #generate thrust curve function from file
    print("[init] Thrust curve loaded")

    if disturbance is None:
        disturbance = load_disturbance("default")
    dist_desc = getattr(disturbance, "__module__", "unknown").split(".")[-1]
    print(f"[init] Disturbance model: {dist_desc}")
    print(f"[init] ODE solver method: {ivp_method}")

    nu0 = np.zeros(12)
    nu0[7] = 2 * np.pi / 180   #set intiial pitch to 2
    nu0[8] = 2 * np.pi / 180   #initial yaw to 2
    print(f"[init] Initial pitch={np.degrees(nu0[7]):.1f} deg, yaw={np.degrees(nu0[8]):.1f} deg")

    xi0 = 0.0 #initial inegral state
    X0 = np.concatenate([nu0, [xi0]])

    t_eval = np.linspace(0.0, 20.0, 1000)#1000 evenly spaced points
    phi_ref_main = 45 * np.pi / 180  #target roll angle [rad]
    print(f"[init] Roll angle target: {np.degrees(phi_ref_main):.1f} deg, t_span=0–20 s, {len(t_eval)} eval points")

    #Main closed-loop run
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

    #extracting solutions
    t = sol_cl.t
    X = sol_cl.y.T #transposed to give n_times x n_states
    nu = X[:, 0:12]
    xi = X[:, 12]
    u, v, w         = nu[:, 0], nu[:, 1], nu[:, 2]
    p, q, r         = nu[:, 3], nu[:, 4], nu[:, 5]
    x               = nu[:, 9]

    Vmag = np.sqrt(u**2 + v**2 + w**2)
    Wmag = np.sqrt(p**2 + q**2 + r**2)
    print(f"      Peak altitude={x.max():.1f}m, max speed={Vmag.max():.1f}m/s, "
          f"max |p|={np.degrees(np.abs(p).max()):.2f}deg/s")

    print("[1/5] Reconstructing control history ...")
    p_ref, e_p, e_q, e_r, m_cx, m_cy, m_cz = reconstruct_control_history(t, X, control, rocket, phi_ref_main)
    delta_cx_hist = control.Kp_p * e_p + control.Ki_p * xi  #implied canard angle history [rad]
    print(f"      Max roll torque={np.abs(m_cx).max():.4f} N·m, "
          f"max canard deflection={np.degrees(np.abs(delta_cx_hist).max()):.4f} deg")

    print("[1/5] Computing aero quantities ...")
    alpha_array, beta_array, q_dyn, Vbody_array = compute_body_air_quantities(t, nu, rocket.rho)
    print(f"      Max AoA={np.degrees(np.abs(alpha_array).max()):.2f}deg, "
          f"max q_dyn={q_dyn.max():.1f}Pa")

    print("[1/5] Computing force/moment history ...")
    Fx, Fy, Fz, Mx, My, Mz, alpha_log = compute_force_moment_history(t, nu, rocket)
    print(f"      Max |Fx|={np.abs(Fx).max():.1f}N, max |My|={np.abs(My).max():.3f}N·m")

    #Open-loop run
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

    t_ol  = sol_ol.t
    nu_ol = sol_ol.y.T

    #Energy
    E_kin = 0.5 * rocket.mass * Vmag**2
    E_pot = rocket.mass * rocket.g * x
    E_tot = E_kin + E_pot
    print(f"      Peak total energy={E_tot.max():.1f}J  (KE={E_kin.max():.1f}J, PE={E_pot.max():.1f}J)")

    # --- roll angle target sweep ---
    print("\n[3/5] Running roll angle target sweep (phi_ref = 15, 45, 90 deg) ...")
    phi_refs_deg = [15, 45, 90]
    sweep_results = []
    for phi_deg in phi_refs_deg:
        phi_ref_j = phi_deg * np.pi / 180
        print(f"  Sweep: phi_ref={phi_deg} deg ...")
        sol_j = run_closed_loop_case(X0, t_eval, rocket, control, phi_ref_j, ivp_method)
        phi_j = sol_j.y.T[:, 6]
        print(f"         done, t_final={sol_j.t[-1]:.2f}s, final phi={np.degrees(phi_j[-1]):.2f}deg")
        sweep_results.append((sol_j.t, phi_j, phi_deg))
    print("[3/5] Sweep done")

    # --- disturbance rejection ---
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

    t_dist_cl  = sol_dist_cl.t
    X_dist_cl  = sol_dist_cl.y.T

    t_dist_ol  = sol_dist_ol.t
    nu_dist_ol = sol_dist_ol.y.T

    # --- save all results to data/ ---
    print("\n[5/5] Saving simulation data ...")
    os.makedirs("data", exist_ok=True)

    np.savez("data/cl.npz",
        t=t, X=X, nu=nu, xi=xi,
        phi_ref=np.array(phi_ref_main),
        p_ref=p_ref, e_p=e_p, e_q=e_q, e_r=e_r,
        m_cx=m_cx, m_cy=m_cy, m_cz=m_cz,
        delta_cx=delta_cx_hist,
        alpha=alpha_array, beta=beta_array, q_dyn=q_dyn, Vbody=Vbody_array,
        Fx=Fx, Fy=Fy, Fz=Fz, Mx=Mx, My=My, Mz=Mz,
        E_kin=E_kin, E_pot=E_pot, E_tot=E_tot,
        Vmag=Vmag, Wmag=Wmag,
    )
    np.savez("data/ol.npz", t=t_ol, nu=nu_ol)

    # sweep: save each case as sweep_<phi_deg>.npz
    for t_j, phi_j, phi_deg in sweep_results:
        np.savez(f"data/sweep_{phi_deg}.npz", t=t_j, phi=phi_j, phi_deg=np.array(phi_deg))

    # evaluate disturbance profile over each solution's time grid for plotting
    dist_hist_cl = np.array([disturbance(ti) for ti in t_dist_cl])  # (N, 3)
    dist_hist_ol = np.array([disturbance(ti) for ti in t_dist_ol])  # (N, 3)

    np.savez("data/dist_cl.npz", t=t_dist_cl, X=X_dist_cl, dist=dist_hist_cl)
    np.savez("data/dist_ol.npz", t=t_dist_ol, nu=nu_dist_ol, dist=dist_hist_ol)

    print("[5/5] Data saved to data/")
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
