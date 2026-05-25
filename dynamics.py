import numpy as np
from scipy.integrate import solve_ivp

from params import RocketParams, ControlParams
from kinematics import hat, Cba, Sba, C1
import aerodata as aero

def canard_torque_total(deflection_angle: float, 
                        pres: float, 
                        temp: float, 
                        U: float, 
                        p: RocketParams) -> float:
    rho = aero.calculate_density(pres, temp)
    re = aero.calculate_reynolds(pres, temp, U, p.c_barre_canard)
    ma = aero.calculate_mach(temp, U)
    
    cl = aero.cl_interp([[re, ma, deflection_angle]])[0] # simplify that AoA is the same as deflection angle
    
    lift_force_per_canard = aero.calculate_lift_force_per_canard(rho, U, p.A_ref, cl)
    total_canard_torque = lift_force_per_canard * (p.y_mac_canard + p.R) * p.num_canards
    return total_canard_torque
    
def forces_and_moments(
    t: float,
    psi: float,
    theta: float, #Euler angles
    phi: float,
    nu: np.ndarray,
    m_c: np.ndarray, #control moment vector
    p: RocketParams
) -> tuple[np.ndarray, float]:
    
    pres = aero.pres_lapse(aero.p_abs_ground, nu[9]) # pressure
    temp = aero.temp_lapse(aero.t_abs_ground, nu[9]) # temperature
    rho = aero.calculate_density(pres, temp)         # density
        
    DCM = Cba(psi, theta, phi) #build orientation matrix from euler angles

    v_air_b = np.real(DCM @ nu[0:3]) #take translational velocity and rotate to give air relative velocity
    delta = np.arctan2(v_air_b[2], v_air_b[1])

    Ccb = C1(-delta) #create  x rotation matrix
    Cca = Ccb @ DCM

    Ux, Uy, Uz = v_air_b #unpack body frame velocity components
    U = np.sqrt(Ux**2 + Uy**2 + Uz**2) + 1e-6
    alpha = np.arctan2(Uz, Ux) #angle of attack
    # note: the pitch axis is dynamically defined such that the pitch angle is as always 
    # co-planar with the AoA. Thus yaw should always be zero. 
    Mach = U / 340.0
    beta_m = np.sqrt(max(1.0 - Mach**2, 1e-6)) # compressibility factor

    p_roll = nu[3] # these are angular velocities in body frame. p -> omega_roll
    q_pitch = nu[4] # q -> omega_pitch
    r_ag = np.array([p.r_ag_x, 0.0, 0.0])

    C_N_nose = (2.0 / p.A_ref) * (p.A_ref * np.sin(-alpha)) #Normal force coefficient nose
    C_N_body = p.K_body * ((p.d * p.l_body) / p.A_ref) * np.sin(-alpha) ** 2 #body
    C_N_fins = -alpha * p.K_TB * (3.0 / 2.0) * ( #fin # this 3/2 factor is for 3 fins. need to correct
        2.0 * np.pi * (p.s_fin**2 / p.A_ref) /
        (1.0 + np.sqrt(1.0 + (beta_m * p.s_fin**2 / (p.A_fin * np.cos(p.gamma_c_fin)))**2))
    )

    C_f = 0.007 #skin friction
    C_f_c = C_f * (1.0 - 0.1 * Mach**2)
    C_d_friction = C_f_c * (
        ((1.0 + 1.0 / (2.0 * (p.l / p.d))) * np.pi * p.d * p.l +
         (1.0 + 2.0 * p.t_fin / p.c_barrre_fin) * 6.0 * p.A_fin) / p.A_ref
    )

    if Mach < 1.0:
        C_d_fins = (1.0 - Mach**2) ** (-0.417) - 0.88 + 0.13 * Mach**2
    else:
        C_d_fins = 0.0

    C_d_0 = (   #baseline drag coef
        p.num_fins * p.t_fin * p.s_fin * C_d_fins / p.A_ref
        + (np.pi * 0.1 * p.R**2) / p.A_ref    # nose cone contribution
        + C_d_friction
    )

    lift_slope_term = (
        2.0 * np.pi * (p.s_fin**2 / p.A_ref) /
        (1.0 + np.sqrt(1.0 + (beta_m * p.s_fin**2 / (p.A_fin * np.cos(p.gamma_c_fin)))**2))
    )

    C_l_f = p.num_fins * (p.y_mac_fin + p.R) * lift_slope_term * p.cant_fin / p.d #rolling moment coef contribution due to fin cant

    geom_term = (
        0.5 * (p.C_r_fin + p.C_t_fin) * p.R**2 * p.s_fin
        + (p.C_r_fin + 2.0 * p.C_t_fin) * (1.0 / 3.0) * p.R * p.s_fin**2
        + (p.C_r_fin + 3.0 * p.C_t_fin) * (1.0 / 12.0) * p.s_fin**3
    )

    # Floor for damping denominators — prevents 1/U and 1/U² singularities near apogee
    # where aerodynamic damping is negligible anyway (dynamic pressure → 0)
    U_damp = max(U, 5.0)

    # this C_l_d eqn looks different from the report
    C_l_d = p.num_fins * p_roll * (2.0 * np.pi / beta_m) * geom_term / (p.A_ref * p.d * U_damp) #roll damping coefficient

    N = 0.5 * rho * p.A_ref * (C_N_nose + C_N_body + C_N_fins) * U**2 #normal force
    D = 0.5 * rho * p.A_ref * C_d_0 * U**2 #drag

    m_l = np.array([  # aero roll moment vector
        0.5 * rho * p.A_ref * p.d * (C_l_f - C_l_d) * U**2,
        0.0,
        0.0
    ])

    f_g = np.array([-p.mass * p.g, 0.0, 0.0]) #weight
    f_d = Cca.T @ np.array([-D, 0.0, 0.0]) #drag rotated into dynamics frame

    if not hasattr(p, 'thrust'):
        raise ValueError("No thrust curve provided in RocketParams. Set p.thrust to a callable.")
    thrust = p.thrust(t)
    f_t = DCM.T @ np.array([thrust, 0.0, 0.0]) #thrust rotated into dynamics frame

    f_N = Cca.T @ np.array([0.0, N, 0.0])
    pitch_moment = hat(r_ag) @ DCM @ f_N

    C_damp_body = 0.55 * ((p.l**4 * p.R) / (p.A_ref * p.d)) * (np.abs(q_pitch) * q_pitch / U_damp**2)
    d_fin_cg = 0.825 - 0.506                                                         #pitch damping
    C_damp_fin = 0.6 * p.num_fins * p.A_fin * d_fin_cg**3 * np.abs(q_pitch) * q_pitch / (p.A_ref * p.d * U_damp**2)

    m_damping = Ccb.T @ np.array([
        0.0,
        0.5 * rho * p.A_ref * (-C_damp_fin - C_damp_body) * p.d * U**2,
        0.0
    ])

    if not np.all(np.isfinite(m_damping)):
        m_damping = np.zeros(3)

    f_total = f_g + f_t + f_d #gravity, thrust, drag
    m_total = m_l + pitch_moment + m_damping + m_c

    FM = np.concatenate([f_total, m_total]) #6 component vector
    return FM, alpha


def ode_rocket(t: float, nu: np.ndarray, m_c: np.ndarray, p: RocketParams) -> np.ndarray:
    v = nu[0:3] #vx, vy, vz inertial-frame translational velocity
    omega = nu[3:6] #p,q,r angular velocity vector in body coordinates
    phi = nu[6]
    theta = nu[7] #pulling out euler angles
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
    # pitch_rate = nu[4] No pitch/yaw control so pitch/yaw rate not needed
    # yaw_rate   = nu[5]
    theta = nu[7]
    psi   = nu[8]

    v_body = Cba(psi, theta, phi) @ nu[0:3]
    U = np.linalg.norm(v_body) + 1e-6

    # Outer loop: roll angle error → roll rate command
    p_ref = c.K_phi * (phi_ref - phi)

    # Pitch/yaw outer loops (unchanged) (Not needed)
    # q_ref = c.K_theta * (0.0 - theta)
    # r_ref = c.K_psi   * (0.0 - psi)

    e_p = p_ref - roll_rate   #inner-loop roll rate error
    # pitch and yaw error are not needed
    # e_q = q_ref - pitch_rate
    # e_r = r_ref - yaw_rate

    dxi = e_p #integrator driven by roll rate error
    
    pres = aero.pres_lapse(aero.p_abs_ground, nu[9])
    temp = aero.temp_lapse(aero.t_abs_ground, nu[9])
    delta_cx = c.Kp_p * e_p + c.Ki_p * xi  #canard deflection [rad]
    m_cx = canard_torque_total(delta_cx*np.pi/180, pres, temp, U, p)

    m_c = np.array([m_cx, 0, 0])
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

    pres = aero.pres_lapse(aero.p_abs_ground, nu[9])
    temp = aero.temp_lapse(aero.t_abs_ground, nu[9])
    delta_cx = c.Kp_p * e_p + c.Ki_p * xi #canard deflection [rad]
    m_cx = canard_torque_total(delta_cx*np.pi/180, pres, temp, U, p)
    m_c = np.array([m_cx, 0.0, 0.0]) + disturbance(t)

    dnu = ode_rocket(t, nu, m_c, p)
    return np.concatenate([dnu, [dxi]])


def ode_open_disturb(t: float, nu: np.ndarray, disturbance, p: RocketParams) -> np.ndarray:
    m_c = disturbance(t)
    return ode_rocket(t, nu, m_c, p)


_APOGEE_MIN_ALT = 50.0  # [m] minimum altitude before apogee detection activates

def event_apogee(t: float, X: np.ndarray, *_) -> float:
    nu = X[0:12]
    h  = nu[9]   # inertial x = altitude
    vx = nu[0]   # velocity whose zero crossing indicates apogee
    # Return positive (no event) until the rocket has left the pad;
    # once above the threshold hand off to vx so direction=-1 triggers correctly.
    return vx if h > _APOGEE_MIN_ALT else 1.0

event_apogee.terminal = True  #stop integrating
event_apogee.direction = -1   #trigger event when function crosses zero in negative direction


def run_closed_loop_case(
    X0: np.ndarray,
    t_eval: np.ndarray,
    rocket: RocketParams,
    control: ControlParams,
    phi_ref: float, # desried roll rate
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
