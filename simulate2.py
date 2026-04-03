import time
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from dataclasses import dataclass
from scipy.integrate import solve_ivp
from numba import njit

# Flat-array indices for RocketParams (used by all @njit functions)
_P_RHO, _P_MASS, _P_G, _P_R, _P_L          =  0,  1,  2,  3,  4
_P_L_BODY, _P_K_BODY                         =  5,  6
_P_S_FIN, _P_GAMMA_C, _P_T_FIN              =  7,  8,  9
_P_C_R, _P_C_T, _P_C_BARRE, _P_CANT         = 10, 11, 12, 13
_P_JX, _P_JY, _P_JZ, _P_R_AG_X             = 14, 15, 16, 17
_P_A_REF, _P_D, _P_A_FIN, _P_Y_MAC, _P_K_TB = 18, 19, 20, 21, 22
_P_LEN = 23

# Flat-array indices for ControlParams
_C_KP_P, _C_KI_P, _C_K_THETA, _C_K_PSI, _C_KP_Q, _C_KP_R = 0, 1, 2, 3, 4, 5

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
        # Derived scalars for flat array
        _A_ref = np.pi * self.R**2
        _d     = 2.0 * self.R
        _A_fin = (0.0889 + (0.0889 - 0.0381)) * self.s_fin / 2
        _y_mac = (self.s_fin / 3) * ((self.C_r + 2 * self.C_t) / (self.C_r + self.C_t))
        _K_TB  = 1.0 + self.R / (self.R + self.s_fin)
        # Flat parameter array for @njit functions (indices defined at top of file)
        self._p_arr = np.array([
            self.rho, self.mass, self.g, self.R, self.l,
            self.l_body, self.K_body,
            self.s_fin, self.gamma_c, self.t_fin,
            self.C_r, self.C_t, self.c_barre, self.cant,
            self.Jx, self.Jy, self.Jz, self.r_ag_x,
            _A_ref, _d, _A_fin, _y_mac, _K_TB,
        ], dtype=np.float64)
        # Default thrust arrays (sampled from parametric profile) for @njit callers
        _t = np.linspace(0.0, self.tburn + self.fade_time + 1.0, 200)
        _v = np.array([thrust_profile(ti, self) for ti in _t])
        self._thrust_t = _t.astype(np.float64)
        self._thrust_v = _v.astype(np.float64)

    @property
    def J(self) -> np.ndarray: #inertia matrix
        return self._J

    @property
    def Mmat(self) -> np.ndarray: #combined mass and inertia matrix 6x6
        return self._Mmat
    
    
    def generate_thrust_curve(self, file: str) -> callable:
        thrust = np.loadtxt(file, delimiter=',', skiprows=5)
        self.thrust = lambda t: np.interp(t, thrust[:, 0], thrust[:, 1])
        self._thrust_t = thrust[:, 0].astype(np.float64)
        self._thrust_v = thrust[:, 1].astype(np.float64)

    def plot_thrust_curve(self):
        if not hasattr(self, 'thrust'):
            raise ValueError("Thrust curve not generated. Call generate_thrust_curve() first.")

        t = np.linspace(0, 20, 1000)
        thrust_values = np.array([self.thrust(ti) for ti in t])

        plt.figure()
        plt.plot(t, thrust_values, linewidth=1.5)
        plt.xlabel("Time [s]")
        plt.ylabel("Thrust [N]")
        plt.title("Rocket Thrust Curve")
        plt.grid(True)
        plt.show()


@dataclass
class ControlParams:
    Kp_p: float = 0.2357 #proportional gain for roll rate
    Ki_p: float = 0.5656 #integral gain

    K_theta: float = 2.0 #gain converts pitch angle error to pitch rate command
    K_psi: float = 2.0 #same for yaw

    Kp_q: float = 3.0 #proportional gain for pitch rate
    Kp_r: float = 3.0 #proportional gain for yaw rate

    def __post_init__(self):
        self._c_arr = np.array(
            [self.Kp_p, self.Ki_p, self.K_theta, self.K_psi, self.Kp_q, self.Kp_r],
            dtype=np.float64
        )


#Rotation helpers
@njit(cache=True)
def hat(vec: np.ndarray) -> np.ndarray: #cross product
    return np.array([
        [0.0, -vec[2], vec[1]],
        [vec[2], 0.0, -vec[0]],
        [-vec[1], vec[0], 0.0]
    ])


@njit(cache=True)
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


@njit(cache=True)
def Sba(phi: float, theta: float) -> np.ndarray: #L matrix: L @ [phi_dot, theta_dot, psi_dot] = [p, q, r]
    return np.array([
        [1.0,  0.0,              -np.sin(theta)             ],
        [0.0,  np.cos(phi),       np.sin(phi) * np.cos(theta)],
        [0.0, -np.sin(phi),       np.cos(phi) * np.cos(theta)]
    ])


@njit(cache=True)
def C1(delta: float) -> np.ndarray: #rotation matrix about x axis
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(delta), -np.sin(delta)], 
        [0.0, np.sin(delta), np.cos(delta)]
    ])


# ── JIT hot-path functions ────────────────────────────────────────────────────

@njit(cache=True)
def _canard_torque_nb(delta_c, U, rho, pa):
    s_fin   = pa[_P_S_FIN]
    A_ref   = pa[_P_A_REF]
    A_fin   = pa[_P_A_FIN]
    gamma_c = pa[_P_GAMMA_C]
    y_mac   = pa[_P_Y_MAC]
    R       = pa[_P_R]
    d       = pa[_P_D]
    Mach    = U / 340.0
    beta_m  = np.sqrt(max(1.0 - Mach**2, 1e-6))
    lift_slope = (
        2.0 * np.pi * (s_fin**2 / A_ref) /
        (1.0 + np.sqrt(1.0 + (beta_m * s_fin / (A_fin * np.cos(gamma_c)))**2))
    )
    C_l_delta = 3.0 * (y_mac + R) * lift_slope / d
    return 0.5 * rho * A_ref * d * C_l_delta * delta_c * U**2


@njit(cache=True)
def _forces_and_moments_nb(t, psi, theta, phi, nu, m_c, pa, thrust_t, thrust_v):
    rho     = pa[_P_RHO]
    mass    = pa[_P_MASS]
    g       = pa[_P_G]
    R       = pa[_P_R]
    l       = pa[_P_L]
    l_body  = pa[_P_L_BODY]
    K_body  = pa[_P_K_BODY]
    s_fin   = pa[_P_S_FIN]
    gamma_c = pa[_P_GAMMA_C]
    t_fin   = pa[_P_T_FIN]
    C_r     = pa[_P_C_R]
    C_t     = pa[_P_C_T]
    c_barre = pa[_P_C_BARRE]
    cant    = pa[_P_CANT]
    r_ag_x  = pa[_P_R_AG_X]
    A_ref   = pa[_P_A_REF]
    d       = pa[_P_D]
    A_fin   = pa[_P_A_FIN]
    y_mac   = pa[_P_Y_MAC]
    K_TB    = pa[_P_K_TB]

    DCM     = Cba(psi, theta, phi)
    v_air_b = DCM @ nu[0:3]
    delta   = np.arctan2(v_air_b[2], v_air_b[1])

    Ccb = C1(-delta)
    Cca = Ccb @ DCM

    Ux = v_air_b[0]; Uy = v_air_b[1]; Uz = v_air_b[2]  # noqa: F841
    U     = np.sqrt(Ux**2 + Uy**2 + Uz**2) + 1e-6
    alpha = np.arctan2(Uz, Ux)
    Mach  = U / 340.0
    beta  = np.sqrt(max(1.0 - Mach**2, 1e-6))

    p_roll  = nu[3]
    q_pitch = nu[4]

    # Normal-force coefficients
    C_N_nose = 2.0 * np.sin(-alpha)
    C_N_body = K_body * ((d * l_body) / A_ref) * np.sin(-alpha)**2
    ls_denom = 1.0 + np.sqrt(1.0 + (beta * s_fin / (A_fin * np.cos(gamma_c)))**2)
    lift_slope_fin = 2.0 * np.pi * (s_fin**2 / A_ref) / ls_denom
    C_N_fins = -alpha * K_TB * 1.5 * lift_slope_fin

    # Drag
    C_f   = 0.007
    C_f_c = C_f * (1.0 - 0.1 * Mach**2)
    C_d_friction = C_f_c * (
        ((1.0 + 1.0 / (2.0 * (l / d))) * np.pi * d * l +
         (1.0 + 2.0 * t_fin / c_barre) * 6.0 * A_fin) / A_ref
    )
    if Mach < 1.0:
        C_d_fins = (1.0 - Mach**2)**(-0.417) - 0.88 + 0.13 * Mach**2
    else:
        C_d_fins = 0.0
    C_d_0 = (
        3.0 * t_fin * s_fin * C_d_fins / A_ref
        + (np.pi * 0.1 * R**2) / A_ref
        + C_d_friction
    )

    # Roll moment coefficients (fin cant + damping)
    C_l_f = 3.0 * (y_mac + R) * lift_slope_fin * cant / d
    geom_term = (
        0.5 * (C_r + C_t) * R**2 * s_fin
        + (C_r + 2.0 * C_t) * (1.0 / 3.0) * R * s_fin**2
        + (C_r + 3.0 * C_t) * (1.0 / 12.0) * s_fin**3
    )
    U_damp = max(U, 5.0)
    C_l_d  = 3.0 * p_roll * (2.0 * np.pi / beta) * geom_term / (A_ref * d * U_damp)

    # Forces
    N = 0.5 * rho * A_ref * (C_N_nose + C_N_body + C_N_fins) * U**2
    D = 0.5 * rho * A_ref * C_d_0 * U**2

    m_l = np.array([0.5 * rho * A_ref * d * (C_l_f - C_l_d) * U**2, 0.0, 0.0])

    f_g = np.array([-mass * g, 0.0, 0.0])
    f_d = Cca.T @ np.array([-D, 0.0, 0.0])

    thrust_val = np.interp(t, thrust_t, thrust_v)
    f_t        = DCM.T @ np.array([thrust_val, 0.0, 0.0])

    # Pitch moment from normal force offset: hat([r_ag_x,0,0]) @ (DCM @ f_N)
    # hat([a,0,0]) @ v = [0, -a*v[2], a*v[1]]
    f_N  = Cca.T @ np.array([0.0, N, 0.0])
    tmp  = DCM @ f_N
    pitch_moment = np.array([0.0, -r_ag_x * tmp[2], r_ag_x * tmp[1]])

    # Pitch/yaw damping
    C_damp_body = 0.55 * ((l**4 * R) / (A_ref * d)) * (np.abs(q_pitch) * q_pitch / U_damp**2)
    d_fin_cg    = 0.319  # 0.825 - 0.506 m
    C_damp_fin  = 0.6 * 3.0 * A_fin * d_fin_cg**3 * np.abs(q_pitch) * q_pitch / (A_ref * d * U_damp**2)
    m_damp_y    = 0.5 * rho * A_ref * (-C_damp_fin - C_damp_body) * d * U**2
    m_damping   = Ccb.T @ np.array([0.0, m_damp_y, 0.0])

    if not (np.isfinite(m_damping[0]) and np.isfinite(m_damping[1]) and np.isfinite(m_damping[2])):
        m_damping = np.zeros(3)

    f_total = f_g + f_t + f_d
    m_total = m_l + pitch_moment + m_damping + m_c

    FM = np.empty(6)
    FM[0:3] = f_total
    FM[3:6] = m_total
    return FM


@njit(cache=True)
def _ode_rocket_nb(t, nu, m_c, pa, thrust_t, thrust_v):
    omega = nu[3:6]
    phi   = nu[6]
    theta = nu[7]
    psi   = nu[8]

    Lmat    = Sba(phi, theta)
    dangles = np.linalg.solve(Lmat, omega)

    Jx = pa[_P_JX]; Jy = pa[_P_JY]; Jz = pa[_P_JZ]
    p_r = omega[0]; q_r = omega[1]; r_r = omega[2]

    # Gyroscopic term: omega × (J @ omega)  [translational part is zero]
    gyro6    = np.zeros(6)
    gyro6[3] = (Jz - Jy) * q_r * r_r
    gyro6[4] = (Jx - Jz) * r_r * p_r
    gyro6[5] = (Jy - Jx) * p_r * q_r

    FM  = _forces_and_moments_nb(t, psi, theta, phi, nu, m_c, pa, thrust_t, thrust_v)
    rhs = FM - gyro6

    mass = pa[_P_MASS]
    out = np.empty(12)
    out[0] = rhs[0] / mass
    out[1] = rhs[1] / mass
    out[2] = rhs[2] / mass
    out[3] = rhs[3] / Jx
    out[4] = rhs[4] / Jy
    out[5] = rhs[5] / Jz
    out[6]  = dangles[0]
    out[7]  = dangles[1]
    out[8]  = dangles[2]
    out[9]  = nu[0]   # dpos = v (inertial velocity = d/dt of inertial position)
    out[10] = nu[1]
    out[11] = nu[2]
    return out


@njit(cache=True)
def _ode_cl_nb(t, X, p_step, pa, thrust_t, thrust_v, ca):  # noqa: ARG001 (p_step reserved)
    nu  = X[0:12]
    xi  = X[12]

    p_ref = 0.0  # roll-rate reference (currently held at zero; step logic in disturb variant)

    roll_rate  = nu[3]
    pitch_rate = nu[4]
    yaw_rate   = nu[5]
    theta = nu[7]
    psi   = nu[8]
    phi   = nu[6]

    DCM = Cba(psi, theta, phi)
    v_b = DCM @ nu[0:3]
    U   = np.sqrt(v_b[0]**2 + v_b[1]**2 + v_b[2]**2) + 1e-6

    Kp_p    = ca[_C_KP_P]
    Ki_p    = ca[_C_KI_P]
    K_theta = ca[_C_K_THETA]
    K_psi   = ca[_C_K_PSI]
    Kp_q    = ca[_C_KP_Q]
    Kp_r    = ca[_C_KP_R]

    q_ref = K_theta * (-theta)   # theta_ref = 0
    r_ref = K_psi   * (-psi)     # psi_ref   = 0

    e_p = p_ref - roll_rate
    e_q = q_ref - pitch_rate
    e_r = r_ref - yaw_rate

    delta_cx = Kp_p * e_p + Ki_p * xi
    m_cx     = _canard_torque_nb(delta_cx, U, pa[_P_RHO], pa)
    m_c      = np.array([m_cx, Kp_q * e_q, Kp_r * e_r])

    dnu = _ode_rocket_nb(t, nu, m_c, pa, thrust_t, thrust_v)

    out = np.empty(13)
    out[0:12] = dnu
    out[12]   = e_p
    return out


@njit(cache=True)
def _ode_open_nb(t, nu, pa, thrust_t, thrust_v):
    return _ode_rocket_nb(t, nu, np.zeros(3), pa, thrust_t, thrust_v)


@njit(cache=True)
def _ode_cl_disturb_nb(t, X, p_step, M_dist, pa, thrust_t, thrust_v, ca):
    nu  = X[0:12]
    xi  = X[12]

    if t < 5.0:
        p_ref = 0.0
    else:
        p_ref = p_step

    roll_rate = nu[3]
    phi   = nu[6]
    theta = nu[7]
    psi   = nu[8]

    DCM = Cba(psi, theta, phi)
    v_b = DCM @ nu[0:3]
    U   = np.sqrt(v_b[0]**2 + v_b[1]**2 + v_b[2]**2) + 1e-6

    Kp_p = ca[_C_KP_P]
    Ki_p = ca[_C_KI_P]

    e_p      = p_ref - roll_rate
    delta_cx = Kp_p * e_p + Ki_p * xi
    m_cx     = _canard_torque_nb(delta_cx, U, pa[_P_RHO], pa)
    m_c      = np.array([m_cx, 0.0, 0.0])

    if t > 8.0 and t < 8.2:
        m_c[0] = m_c[0] + M_dist

    dnu = _ode_rocket_nb(t, nu, m_c, pa, thrust_t, thrust_v)

    out = np.empty(13)
    out[0:12] = dnu
    out[12]   = e_p
    return out


@njit(cache=True)
def _ode_open_disturb_nb(t, nu, M_dist, pa, thrust_t, thrust_v):
    m_c = np.zeros(3)
    if t > 8.0 and t < 8.2:
        m_c[0] = M_dist
    return _ode_rocket_nb(t, nu, m_c, pa, thrust_t, thrust_v)


# ── end JIT functions ─────────────────────────────────────────────────────────

#Propulsion
def thrust_profile(t: float, p: RocketParams) -> float:
    if t < p.tburn:
        return p.Tmax
    if t < p.tburn + p.fade_time:
        x = (t - p.tburn) / p.fade_time
        return p.Tmax * 0.5 * (1 + np.cos(np.pi * x)) #cosine decay
    return 0.0


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

    U_damp = max(U, 5.0) #floor for damping denominators — prevents 1/U singularity near apogee
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

    thrust = p.thrust(t) if hasattr(p, 'thrust') else thrust_profile(t, p)
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


def ode_cl(t: float, X: np.ndarray, p_step: float, p: RocketParams, c: ControlParams) -> np.ndarray:
    return _ode_cl_nb(t, X, p_step, p._p_arr, p._thrust_t, p._thrust_v, c._c_arr)


def ode_open(t: float, nu: np.ndarray, p: RocketParams) -> np.ndarray:
    return _ode_open_nb(t, nu, p._p_arr, p._thrust_t, p._thrust_v)


def ode_cl_disturb(t: float, X: np.ndarray, p_step: float, M_dist: float,
                   p: RocketParams, c: ControlParams) -> np.ndarray:
    return _ode_cl_disturb_nb(t, X, p_step, M_dist, p._p_arr, p._thrust_t, p._thrust_v, c._c_arr)


def ode_open_disturb(t: float, nu: np.ndarray, M_dist: float, p: RocketParams) -> np.ndarray:
    return _ode_open_disturb_nb(t, nu, M_dist, p._p_arr, p._thrust_t, p._thrust_v)


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


def reconstruct_control_history(t: np.ndarray, X: np.ndarray, c: ControlParams, rocket: RocketParams):
    nu = X[:, 0:12]
    xi = X[:, 12]

    p_rate = nu[:, 3]
    q_rate = nu[:, 4]
    r_rate = nu[:, 5]
    theta  = nu[:, 7]
    psi    = nu[:, 8]

    p_ref     = np.zeros_like(t)
    theta_ref = np.zeros_like(t)
    psi_ref   = np.zeros_like(t)

    q_ref = c.K_theta * (theta_ref - theta)
    r_ref = c.K_psi   * (psi_ref   - psi)

    e_p = p_ref - p_rate
    e_q = q_ref - q_rate
    e_r = r_ref - r_rate

    # Reconstruct canard deflection angle history, then map to torque
    delta_cx = c.Kp_p * e_p + c.Ki_p * xi
    U_hist = np.array([
        np.linalg.norm(Cba(psi[k], theta[k], nu[k, 6]) @ nu[k, 0:3]) + 1e-6
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
    p_step: float
):
    return solve_ivp(
        fun=ode_cl,
        args=(p_step, rocket, control),
        t_span=(t_eval[0], t_eval[-1]),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )



#Main simulation
def simulate_rocket_trajectory():
    print("=== Polaris Roll-Control Simulation ===")

    rocket = RocketParams()
    control = ControlParams()
    print(f"[init] RocketParams: mass={rocket.mass} kg, R={rocket.R:.4f} m, "
          f"d={rocket.d:.4f} m, l={rocket.l:.4f} m")
    print(f"[init] Inertia: Jx={rocket.Jx} kg·m², Jy={rocket.Jy} kg·m², Jz={rocket.Jz} kg·m²")
    print(f"[init] ControlParams: Kp_p={control.Kp_p}, Ki_p={control.Ki_p}, "
          f"Kp_q={control.Kp_q}, Kp_r={control.Kp_r}")

    rocket.generate_thrust_curve("AeroTech_HP-K535W.csv") #generate thrust curve function from file
    print("[init] Thrust curve loaded")
    rocket.plot_thrust_curve()

    # Trigger JIT compilation before any timed runs (cache=True persists across sessions)
    print("[init] Warming up JIT-compiled ODE kernels ...")
    _t0 = time.perf_counter()
    _dummy_X  = np.zeros(13); _dummy_X[0] = 1.0
    _dummy_nu = np.zeros(12); _dummy_nu[0] = 1.0
    _pa = rocket._p_arr; _tt = rocket._thrust_t; _tv = rocket._thrust_v; _ca = control._c_arr
    _ode_cl_nb(0.0, _dummy_X,  0.0,       _pa, _tt, _tv, _ca)
    _ode_open_nb(0.0, _dummy_nu,           _pa, _tt, _tv)
    _ode_cl_disturb_nb(0.0, _dummy_X, 0.0, 0.0, _pa, _tt, _tv, _ca)
    _ode_open_disturb_nb(0.0, _dummy_nu,   0.0, _pa, _tt, _tv)
    print(f"[init] JIT warm-up done in {time.perf_counter()-_t0:.1f}s")

    nu0 = np.zeros(12)
    nu0[7] = 2 * np.pi / 180   #set intiial pitch to 2
    nu0[8] = 2 * np.pi / 180   #initial yaw to 2
    print(f"[init] Initial pitch={np.degrees(nu0[7]):.1f} deg, yaw={np.degrees(nu0[8]):.1f} deg")

    xi0 = 0.0 #initial inegral state
    X0 = np.concatenate([nu0, [xi0]])

    t_eval = np.linspace(0.0, 20.0, 1000)#1000 evenly spaced points
    p_step_main = 30 * np.pi / 180 #roll command magnitute
    print(f"[init] Roll command: {np.degrees(p_step_main):.1f} deg/s, t_span=0–20 s, {len(t_eval)} eval points")

    #Main closed-loop run
    print("\n[1/5] Running main closed-loop simulation ...")
    _t0 = time.perf_counter()
    sol_cl = solve_ivp(
        fun=ode_cl,
        args=(p_step_main, rocket, control),
        t_span=(0.0, 20.0),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
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
    #unpack state histories
    u = nu[:, 0]
    v = nu[:, 1]
    w = nu[:, 2]
    p = nu[:, 3]
    q = nu[:, 4]
    r = nu[:, 5]
    phi = nu[:, 6]
    theta = nu[:, 7]
    psi = nu[:, 8]
    x = nu[:, 9]
    y = nu[:, 10]
    z = nu[:, 11]

    Vmag = np.sqrt(u**2 + v**2 + w**2)
    print(f"      Peak altitude={x.max():.1f}m, max speed={Vmag.max():.1f}m/s, "
          f"max |p|={np.degrees(np.abs(p).max()):.2f}deg/s")

    print("[1/5] Reconstructing control history ...")
    p_ref, e_p, e_q, e_r, m_cx, m_cy, m_cz = reconstruct_control_history(t, X, control, rocket)
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
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    print(f"[2/5] Open-loop done in {time.perf_counter()-_t0:.1f}s: t_final={sol_ol.t[-1]:.2f}s")

    t_ol = sol_ol.t #open loop time history
    nu_ol = sol_ol.y.T #open loop state history
    p_ol = nu_ol[:, 3] #roll rate
    phi_ol = nu_ol[:, 6] #roll angle

    #Energy
    h = x  #height
    E_kin = 0.5 * rocket.mass * Vmag**2 #kinetic energy
    E_pot = rocket.mass * rocket.g * h #potential energy
    E_tot = E_kin + E_pot
    print(f"      Peak total energy={E_tot.max():.1f}J  (KE={E_kin.max():.1f}J, PE={E_pot.max():.1f}J)")


    Wmag = np.sqrt(p**2 + q**2 + r**2)

    # --- command amplitude sweep (all runs before any figure is created) ---
    print("\n[3/5] Running command amplitude sweep (p_ref = 10, 30, 60 deg/s) ...")
    p_steps_deg = [10, 30, 60]
    sweep_results = []
    for p_deg in p_steps_deg:
        p_step = p_deg * np.pi / 180
        print(f"  Sweep: p_ref={p_deg} deg/s ...")
        sol_j = run_closed_loop_case(X0, t_eval, rocket, control, p_step)
        p_j = sol_j.y.T[:, 3]
        print(f"         done, t_final={sol_j.t[-1]:.2f}s, peak p={np.degrees(np.abs(p_j).max()):.2f}deg/s")
        sweep_results.append((sol_j.t, p_j, p_deg))
    print("[3/5] Sweep done")

    # --- disturbance rejection ---
    M_dist = 0.5 #disturbance moment magnitude
    print(f"\n[4/5] Running disturbance rejection (M_dist={M_dist} N·m at t=8–8.2s) ...")

    _t0 = time.perf_counter()
    sol_dist_cl = solve_ivp(
        fun=ode_cl_disturb,
        args=(p_step_main, M_dist, rocket, control),
        t_span=(0.0, 20.0),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    print(f"      Disturbed CL done in {time.perf_counter()-_t0:.1f}s: t_final={sol_dist_cl.t[-1]:.2f}s")

    _t0 = time.perf_counter()
    sol_dist_ol = solve_ivp(
        fun=ode_open_disturb,
        args=(M_dist, rocket),
        t_span=(0.0, 20.0),
        y0=nu0,
        t_eval=t_eval,
        events=event_apogee,
        method='RK45',
        rtol=1e-6,
        atol=1e-9
    )
    print(f"      Disturbed OL done in {time.perf_counter()-_t0:.1f}s: t_final={sol_dist_ol.t[-1]:.2f}s")
    print("[4/5] Disturbance done")

    t_dist_cl = sol_dist_cl.t
    X_dist_cl  = sol_dist_cl.y.T
    p_dist_cl  = X_dist_cl[:, 3]

    t_dist_ol   = sol_dist_ol.t
    nu_dist_ol  = sol_dist_ol.y.T
    p_dist_ol   = nu_dist_ol[:, 3]

    # --- all plotting, no more solve_ivp calls below this line ---
    print("\n[5/5] Generating plots ...")
    # Fig 1: 3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=1.5, label="Trajectory")

    step = 50
    scale = 10.0
    for i in range(0, len(t), step):
        C = Cba(psi[i], theta[i], phi[i])
        nose_inertial = C.T @ np.array([5.0, 0.0, 0.0])
        pos_inertial = np.array([x[i], y[i], z[i]])
        ax.quiver(
            pos_inertial[0], pos_inertial[1], pos_inertial[2],
            scale * nose_inertial[0], scale * nose_inertial[1], scale * nose_inertial[2],
            length=1.0, normalize=False
        )
    ax.set_xlabel("x"); ax.set_ylabel("y"); ax.set_zlabel("z")
    ax.set_title("3D Trajectory")
    ax.legend()
    ax.view_init(elev=-35, azim=-63)

    # Fig 2: Kinematics
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

    # Fig 3: Attitude & Aerodynamics
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

    # Fig 4: Forces & Moments
    fig, axes = plt.subplots(2, 1, figsize=(9, 7))
    fig.suptitle("Forces & Moments")

    axes[0].plot(t, Fx, label="Fx"); axes[0].plot(t, Fy, label="Fy"); axes[0].plot(t, Fz, label="Fz")
    axes[0].set(xlabel="Time [s]", ylabel="Forces [N]", title="Aero + thrust + weight"); axes[0].legend(); axes[0].grid(True)

    axes[1].plot(t, Mx, label="Mx"); axes[1].plot(t, My, label="My"); axes[1].plot(t, Mz, label="Mz")
    axes[1].set(xlabel="Time [s]", ylabel="Moments [N·m]", title="Roll, pitch, yaw moments"); axes[1].legend(); axes[1].grid(True)

    # Fig 5: Roll control
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

    # Fig 6: Comparisons (uses pre-computed sweep_results and disturbance solutions)
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

#Disturbnace detail
    plt.figure()
    plt.plot(t_dist_ol, p_dist_ol * 180 / np.pi, "--", linewidth=1.5, label="Open-loop")
    plt.plot(t_dist_cl, p_dist_cl * 180 / np.pi, linewidth=1.5, label="Closed-loop PI")
    plt.xlabel("Time [s]")
    plt.ylabel("Roll rate p [deg/s]")
    plt.title("Rejection of a rolling impulse disturbance")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    print("[5/5] Plots ready — displaying windows")
    plt.show()

    print("\n=== Simulation complete ===")
    return {
        "t": t,
        "X": X,
        "nu": nu,
        "xi": xi,
        "p_ref": p_ref,
        "e_p": e_p,
        "m_cx": m_cx,
        "m_cy": m_cy,
        "m_cz": m_cz,
        "alpha": alpha_array,
        "beta": beta_array,
        "q_dyn": q_dyn,
        "Vbody": Vbody_array,
        "Fx": Fx,
        "Fy": Fy,
        "Fz": Fz,
        "Mx": Mx,
        "My": My,
        "Mz": Mz,
        "E_kin": E_kin,
        "E_pot": E_pot,
        "E_tot": E_tot,
        "open_loop_t": t_ol,
        "open_loop_nu": nu_ol,
        "dist_cl_t": t_dist_cl,
        "dist_cl_X": X_dist_cl,
        "dist_ol_t": t_dist_ol,
        "dist_ol_nu": nu_dist_ol,
    }


if __name__ == "__main__":
    simulate_rocket_trajectory()
