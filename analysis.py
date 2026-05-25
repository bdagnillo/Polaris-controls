import numpy as np

from params import RocketParams, ControlParams
import aerodata as aero
from kinematics import Cba
from dynamics import canard_torque_total, forces_and_moments


def compute_body_air_quantities(
    t: np.ndarray,
    nu_hist: np.ndarray, #history of state vectors over time
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
        
        pres = aero.pres_lapse(aero.p_abs_ground, nu_hist[k, 9]) # pressure at time step k
        temp = aero.temp_lapse(aero.t_abs_ground, nu_hist[k, 9]) # temperature
        rho = aero.calculate_density(pres, temp)         # density
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

    delta_cx = c.Kp_p * e_p + c.Ki_p * xi # deflection angle in radians
    U_hist = np.array([
        np.linalg.norm(Cba(psi[k], theta[k], phi[k]) @ nu[k, 0:3]) + 1e-6
        for k in range(len(t))
    ])
    
    
    m_cx = np.zeros(len(t))
    for k in range(len(t)):
        pres = aero.pres_lapse(aero.p_abs_ground, nu[k, 9]) # pressure
        temp = aero.temp_lapse(aero.t_abs_ground, nu[k, 9]) # temperature        
        m_cx[k] = canard_torque_total(delta_cx[k]*np.pi/180, pres, temp, U_hist[k], rocket)
    
    m_cy = 0
    m_cz = 0

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
        alpha_log[k] = alpha_k #at each time, store AoA

    return Fx, Fy, Fz, Mx, My, Mz, alpha_log
