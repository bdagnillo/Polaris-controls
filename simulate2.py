import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
from dataclasses import dataclass
from scipy.integrate import solve_ivp

#Parameters
@dataclass
class RocketParams:
    rho: float = 1.225 #TBD
    mass: float = 5 # kg TBD
    g: float = 9.81 # m/s^2

    # all lengths in meters
    d: float = 0.1016 # diameter
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

    # Tmax: float = 10.0 #max thrust
    # tburn: float = 2.5
    # fade_time: float = 15.0

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

    @property
    def J(self) -> np.ndarray: #inertia matrix
        return np.diag([self.Jx, self.Jy, self.Jz])

    @property
    def Mmat(self) -> np.ndarray: #combined mass and inertia matrix 6x6
        return np.block([
            [self.mass * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)), self.J]
        ])
    
    
    def generate_thrust_curve(self, file: str) -> callable:
        thrust = np.loadtxt(file, delimiter=',', skiprows=5)
        self.thrust = lambda t: np.interp(t, thrust[:, 0], thrust[:, 1])

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

    def compute_moment_from_Kp(self, Kp: float, Ki: float, xi: float, e_p: float) -> float:

        # TODO : figure out physical mapping from canard angle to roll moment

        return 0


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


def Sba(psi: float, theta: float) -> np.ndarray: #transform betwen body rates and euler angle rates
    return np.array([
        [np.cos(psi) * np.cos(theta), np.sin(psi), 0.0],
        [-np.sin(psi) * np.cos(theta), np.cos(psi), 0.0],
        [np.sin(theta), 0.0, 1.0]
    ])


def C1(delta: float) -> np.ndarray: #rotation matrix about x axis
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, np.cos(delta), -np.sin(delta)], 
        [0.0, np.sin(delta), np.cos(delta)]
    ])


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

    C_l_d = 3.0 * p_roll * (2.0 * np.pi / beta) * geom_term / (p.A_ref * p.d * U) #roll damping coefficient
    if not np.isfinite(C_l_d):
        C_l_d = 0.0

    N = 0.5 * p.rho * p.A_ref * (C_N_nose + C_N_body + C_N_fins) * U**2 #normal force
    D = 0.5 * p.rho * p.A_ref * C_d_0 * U**2 #drag

    m_l = np.array([  # aero roll moment vector
        0.5 * p.rho * p.A_ref * p.d * (C_l_f - C_l_d) * U**2,
        0.0,
        0.0
    ])

    f_g = np.array([-p.mass * p.g, 0.0, 0.0]) #weight
    f_d = Cca.T @ np.array([-D, 0.0, 0.0]) #drag rotated into dynamics frane

    # thrust = thrust_profile(t, p)
    
    thrust = p.thrust(t) if hasattr(p, 'thrust') else 0.0 #use thrust curve if generated, otherwise zero
    f_t = DCM.T @ np.array([thrust, 0.0, 0.0]) #thrust rotated into dynamics frame

    f_N = Cca.T @ np.array([0.0, N, 0.0])
    pitch_moment = hat(r_ag) @ DCM @ f_N

    C_damp_body = 0.55 * ((p.l**4 * p.R) / (p.A_ref * p.d)) * (q_pitch**2 / U**2)
    d_fin_cg = 0.825 - 0.506                                                         #pitch damping
    C_damp_fin = 0.6 * 3.0 * p.A_fin * d_fin_cg**3 * q_pitch**2 / (p.A_ref * p.d * U**2)

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


    v = nu[0:3] #u,v,w translational velocity vector
    omega = nu[3:6] #p,q,r angular velocity vector in body coordinates
    phi = nu[6]
    theta = nu[7]#pulling out euler angles
    psi = nu[8]

    dangles = np.linalg.solve(Sba(psi, theta), omega) #dangles=vector of euler angle rates
    dpos = v #position derivative

    gyro = np.concatenate([np.zeros(3), hat(omega) @ (p.J @ omega)]) #last 3 elements=rotational gyroscopic term
    rhs = forces_and_moments(t, psi, theta, phi, nu, m_c, p)[0] - gyro #returns vector FM then subtract gyro
    dnu_dyn = np.linalg.solve(p.Mmat, rhs) #gives 3 translational accelerations and 3 angular accelerations

    return np.concatenate([dnu_dyn, dangles, dpos]) #complete derivative of 12 state vector


def ode_cl(t: float, X: np.ndarray, p_step: float, p: RocketParams, c: ControlParams) -> np.ndarray: #p_step=commanded roll-rate step input c=controller gains
    nu = X[0:12] #12 rocket states
    xi = X[12] #integral state for roll PI controller, stores accumulated roll rate error 

    
    if t < 5.0:
        p_ref = 0.0
    else:
        p_ref = 0.0 #??? roll rate ekpt at zero and not tracking 30deg/s roll command

    roll_rate = nu[3]
    pitch_rate = nu[4]
    yaw_rate = nu[5]
    theta = nu[7]
    psi = nu[8] #yaw angle

    theta_ref = 0.0
    psi_ref = 0.0

    q_ref = c.K_theta * (theta_ref - theta) #get desired pitch rate from picth angle error
    r_ref = c.K_psi * (psi_ref - psi) #yaw      angle error to desireed rate to control moment

    e_p = p_ref - roll_rate
    e_q = q_ref - pitch_rate #tracking roll pitch and yaw rate errors
    e_r = r_ref - yaw_rate

    dxi = e_p #derivative of integrator state is roll rate error

    # m_cx = c.Kp_p * e_p + c.Ki_p * xi #roll control moment=proportiotinal+integral 
    m_cx = c.compute_moment_from_angle(c.Kp_p, c.Ki_p, xi, e_p) #use helper function to compute roll control moment based on physical canards
    m_cy = c.Kp_q * e_q #pitch
    m_cz = c.Kp_r * e_r #yaw

    m_c = np.array([m_cx, m_cy, m_cz])
    dnu = ode_rocket(t, nu, m_c, p) #calling plant dynamics

    return np.concatenate([dnu, [dxi]]) #derivatibe of 13 state closed lopp system


def ode_open(t: float, nu: np.ndarray, p: RocketParams) -> np.ndarray: #no control
    return ode_rocket(t, nu, np.zeros(3), p)


def ode_cl_disturb(
    t: float, #testign disturbance
    X: np.ndarray, #full state including integrator
    p_step: float,
    M_dist: float, #disturbance moment
    p: RocketParams,
    c: ControlParams
) -> np.ndarray:
    nu = X[0:12]
    xi = X[12]

    if t < 5.0:
        p_ref = 0.0
    else:
        p_ref = p_step #reference allowed here

    roll_rate = nu[3]

    e_p = p_ref - roll_rate
    dxi = e_p


    m_cx = c.Kp_p * e_p + c.Ki_p * xi
    m_c = np.array([m_cx, 0.0, 0.0])

    if 8.0 < t < 8.2: #add disturbance in this time 
        m_c = m_c + np.array([M_dist, 0.0, 0.0])

    dnu = ode_rocket(t, nu, m_c, p)
    return np.concatenate([dnu, [dxi]]) 


def ode_open_disturb(t: float, nu: np.ndarray, M_dist: float, p: RocketParams) -> np.ndarray: #disturbnace without control
    m_c = np.zeros(3)
    if 8.0 < t < 8.2:
        m_c = m_c + np.array([M_dist, 0.0, 0.0])
    return ode_rocket(t, nu, m_c, p)


def event_apogee(t: float, X: np.ndarray) -> float: #used as an event function
    nu = X[0:12]
    vx = nu[0] #velocity whose zero crossing indicates apogee
    return vx


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


def reconstruct_control_history(t: np.ndarray, X: np.ndarray, c: ControlParams):
    nu = X[:, 0:12]
    xi = X[:, 12] 

    p = nu[:, 3] 
    q = nu[:, 4]
    r = nu[:, 5]
    theta = nu[:, 7]
    psi = nu[:, 8]

 
    p_ref = np.zeros_like(t)
    theta_ref = np.zeros_like(t)
    psi_ref = np.zeros_like(t)

    q_ref = c.K_theta * (theta_ref - theta)
    r_ref = c.K_psi * (psi_ref - psi)

    e_p = p_ref - p
    e_q = q_ref - q
    e_r = r_ref - r

    m_cx = c.Kp_p * e_p + c.Ki_p * xi
    m_cy = c.Kp_q * e_q #control moments
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
    t_eval: np.ndarray, #time points where solution output is to be saved
    rocket: RocketParams,
    control: ControlParams,
    p_step: float
):
    return solve_ivp(
        fun=lambda t, X: ode_cl(t, X, p_step, rocket, control),
        t_span=(t_eval[0], t_eval[-1]), #when to start and stop integrating
        y0=X0, 
        t_eval=t_eval,#return solution at requested time point
        events=event_apogee,#stop early is apogee reached
        rtol=1e-6,#solver tolerance
        atol=1e-9
    )



#Main simulation
def simulate_rocket_trajectory():
    rocket = RocketParams()
    control = ControlParams()

    rocket.generate_thrust_curve("AeroTech_HP-K535W.csv") #generate thrust curve function from file
    rocket.plot_thrust_curve()

    nu0 = np.zeros(12)
    nu0[7] = 2 * np.pi / 180   #set intiial pitch to 2
    nu0[8] = 2 * np.pi / 180   #initial yaw to 2

    xi0 = 0.0 #initial inegral state 
    X0 = np.concatenate([nu0, [xi0]])

    t_eval = np.linspace(0.0, 20.0, 1000)#1000 evenly spaced points
    p_step_main = 30 * np.pi / 180 #roll command magnitute

   
    #Main closed-loop run
  
    sol_cl = solve_ivp(
        fun=lambda t, X: ode_cl(t, X, p_step_main, rocket, control),
        t_span=(0.0, 20.0),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,
        rtol=1e-6,
        atol=1e-9
    )
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

    p_ref, e_p, e_q, e_r, m_cx, m_cy, m_cz = reconstruct_control_history(t, X, control)
    alpha_array, beta_array, q_dyn, Vbody_array = compute_body_air_quantities(t, nu, rocket.rho)
    Fx, Fy, Fz, Mx, My, Mz, alpha_log = compute_force_moment_history(t, nu, rocket)

    
    #Open-loop run
   
    sol_ol = solve_ivp(
        fun=lambda t, nu_: ode_open(t, nu_, rocket),
        t_span=(0.0, 20.0),
        y0=nu0, #no controller integrator state
        t_eval=t_eval,
        rtol=1e-6,
        atol=1e-9
    )

    t_ol = sol_ol.t #open loop time history
    nu_ol = sol_ol.y.T #open loop state history
    p_ol = nu_ol[:, 3] #roll rate
    phi_ol = nu_ol[:, 6] #roll angle

    #Energy
    Vmag = np.sqrt(u**2 + v**2 + w**2) #total transaltional speed
    h = x  #height
    E_kin = 0.5 * rocket.mass * Vmag**2 #kinetic energy
    E_pot = rocket.mass * rocket.g * h #potential energy
    E_tot = E_kin + E_pot


    #3D trajectory
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(x, y, z, linewidth=1.5, label="Trajectory")

    step = 50 #plot orientation arrow every 50 time points
    scale = 10.0 #scale arrow length

    for i in range(0, len(t), step):
        C = Cba(psi[i], theta[i], phi[i]) #build rotation matrix at time i
        nose_inertial = C.T @ np.array([5.0, 0.0, 0.0]) #tak e body axis vector pointing along rocket nose and rotate into pltting frame
        pos_inertial = np.array([x[i], y[i], z[i]]) #store rockets current position

        ax.quiver( #draw 3D arrow
            pos_inertial[0], pos_inertial[1], pos_inertial[2],
            scale * nose_inertial[0],
            scale * nose_inertial[1], #srart arrow at rockets position, point in nose direc, scale for visibility
            scale * nose_inertial[2],
            length=1.0,
            normalize=False
        )

    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D Trajectory of the rocket")
    ax.legend()
    ax.view_init(elev=-35, azim=-63) #viewing angle

   
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(t, u, linewidth=1.5, label="u")
    plt.plot(t, v, linewidth=1.5, label="v")
    plt.plot(t, w, linewidth=1.5, label="w")
    plt.xlabel("Time [s]")
    plt.ylabel("Linear velocity [m/s]")
    plt.title("Linear velocities")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, x, linewidth=1.5, label="x")
    plt.plot(t, y, linewidth=1.5, label="y")
    plt.plot(t, z, linewidth=1.5, label="z")
    plt.xlabel("Time [s]")
    plt.ylabel("Position [m]")
    plt.title("Positions")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    plt.plot(t, Vmag, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("|V| [m/s]")
    plt.title("Velocity magnitude")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    Rmag = np.sqrt(x**2 + y**2 + z**2) #position magnitude
    plt.plot(t, Rmag, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("|r| [m]")
    plt.title("Position magnitude")
    plt.grid(True)
#..............................
    plt.figure()

    plt.subplot(2, 2, 1)
    plt.plot(t, p, linewidth=1.5, label="p")
    plt.plot(t, q, linewidth=1.5, label="q")
    plt.plot(t, r, linewidth=1.5, label="r")
    plt.xlabel("Time [s]")
    plt.ylabel("Angular velocity [rad/s]")
    plt.title("Angular velocities")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 2)
    plt.plot(t, phi * 180 / np.pi, linewidth=1.5, label="phi")
    plt.plot(t, theta * 180 / np.pi, linewidth=1.5, label="theta")
    plt.plot(t, psi * 180 / np.pi, linewidth=1.5, label="psi")
    plt.xlabel("Time [s]")
    plt.ylabel("Euler angles [deg]")
    plt.title("Euler angles")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 2, 3)
    Wmag = np.sqrt(p**2 + q**2 + r**2) #angular velocity magnitutde
    plt.plot(t, Wmag, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("|ω| [rad/s]")
    plt.title("Angular velocity magnitude")
    plt.grid(True)

    plt.subplot(2, 2, 4)
    plt.plot(t, phi, linewidth=1.5, label="phi")
    plt.plot(t, theta, linewidth=1.5, label="theta")
    plt.plot(t, psi, linewidth=1.5, label="psi")
    plt.xlabel("Time [s]")
    plt.ylabel("Euler angles [rad]")
    plt.title("Euler angles (rad)")
    plt.grid(True)
    plt.legend()
#...............................................
    plt.figure()

    plt.subplot(3, 1, 1)
    plt.plot(t, alpha_array * 180 / np.pi, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("alpha [deg]")
    plt.title("Angle of attack alpha(t)")
    plt.grid(True)

    plt.subplot(3, 1, 2)
    plt.plot(t, beta_array * 180 / np.pi, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("beta [deg]")
    plt.title("Sideslip angle beta(t)")
    plt.grid(True)

    plt.subplot(3, 1, 3)
    plt.plot(t, q_dyn, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("q [Pa]")
    plt.title("Dynamic pressure q(t)")
    plt.grid(True)

 #..................................

    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(t, Fx, linewidth=1.5, label="Fx")
    plt.plot(t, Fy, linewidth=1.5, label="Fy")
    plt.plot(t, Fz, linewidth=1.5, label="Fz")
    plt.xlabel("Time [s]")
    plt.ylabel("Forces [N]")
    plt.title("Aerodynamic + thrust + weight forces")
    plt.grid(True)
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(t, Mx, linewidth=1.5, label="Mx")
    plt.plot(t, My, linewidth=1.5, label="My")
    plt.plot(t, Mz, linewidth=1.5, label="Mz")
    plt.xlabel("Time [s]")
    plt.ylabel("Moments [N·m]")
    plt.title("Moments (roll, pitch, yaw)")
    plt.grid(True)
    plt.legend()

#................................
    plt.figure()
    plt.plot(t, E_kin, linewidth=1.5, label="Kinetic")
    plt.plot(t, E_pot, linewidth=1.5, label="Potential")
    plt.plot(t, E_tot, linewidth=1.5, label="Total")
    plt.xlabel("Time [s]")
    plt.ylabel("Energy [J]")
    plt.title("Energy evolution")
    plt.grid(True)
    plt.legend()
  
    plt.figure()
    plt.plot(t, p * 180 / np.pi, linewidth=1.5, label="p")
    plt.plot(t, p_ref * 180 / np.pi, "--", linewidth=1.5, label="p_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("Roll rate p [deg/s]")
    plt.title("Roll rate tracking")
    plt.grid(True)
    plt.legend()

    plt.figure()
    idx_zoom = (t >= 4.0) & (t <= 8.0) #zoom in at the control response between 4 and 8 econds
    plt.plot(t[idx_zoom], p[idx_zoom] * 180 / np.pi, linewidth=1.5, label="p")
    plt.plot(t[idx_zoom], p_ref[idx_zoom] * 180 / np.pi, "--", linewidth=1.5, label="p_ref")
    plt.xlabel("Time [s]")
    plt.ylabel("Roll rate p [deg/s]")
    plt.title("Roll rate tracking (zoom around t = 5 s)")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, e_p * 180 / np.pi, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Error e_p [deg/s]")
    plt.title("Roll rate error (p_ref - p)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, m_cx, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Control moment m_cx [N·m]")
    plt.title("Roll control moment")
    plt.grid(True)

    plt.figure()
    plt.plot(t, xi, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("xi (integral state)")
    plt.title("Integral state evolution (roll)")
    plt.grid(True)

    plt.figure()
    plt.plot(t, phi * 180 / np.pi, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("Roll angle phi [deg]")
    plt.title("Roll angle evolution")
    plt.grid(True)

    plt.figure()
    plt.plot(phi * 180 / np.pi, p * 180 / np.pi, linewidth=1.5)
    plt.xlabel("phi [deg]") #roll rate agaisnt roll angle
    plt.ylabel("p [deg/s]")
    plt.title("Roll phase portrait (phi vs p)")
    plt.grid(True)

    plt.figure()
    roll_fraction = np.abs(p) / (Wmag + 1e-6) #if close to 1, most of angular motion is roll
    plt.plot(t, roll_fraction, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("|p| / ||omega||")
    plt.title("Relative contribution of roll to total angular rate")
    plt.grid(True)

#.......................
    plt.figure()

    plt.subplot(2, 1, 1)
    plt.plot(t, alpha_array * 180 / np.pi, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("alpha [deg]")
    plt.title("Angle of attack alpha(t)")
    plt.grid(True)

    plt.subplot(2, 1, 2)
    plt.plot(t, Vbody_array, linewidth=1.5)
    plt.xlabel("Time [s]")
    plt.ylabel("||v_body|| [m/s]")
    plt.title("Body-frame speed")
    plt.grid(True)

#.....................  
    plt.figure()
    plt.plot(t, p * 180 / np.pi, linewidth=1.5, label="Closed-loop (PI)")
    plt.plot(t_ol, p_ol * 180 / np.pi, "--", linewidth=1.5, label="Open-loop (no control)")
    plt.xlabel("Time [s]")
    plt.ylabel("Roll rate p [deg/s]")
    plt.title("Roll rate: closed-loop vs open-loop")
    plt.grid(True)
    plt.legend()

    plt.figure()
    plt.plot(t, phi * 180 / np.pi, linewidth=1.5, label="Closed-loop (PI)")
    plt.plot(t_ol, phi_ol * 180 / np.pi, "--", linewidth=1.5, label="Open-loop (no control)")
    plt.xlabel("Time [s]")
    plt.ylabel("Roll angle phi [deg]")
    plt.title("Roll angle: closed-loop vs open-loop")
    plt.grid(True)
    plt.legend()

   
    #Command amplitude sweep
    #runs multiple closed loop cases with differen roll rate magnitudes
    p_steps_deg = [10, 30, 60]
    plt.figure()
    for p_deg in p_steps_deg:
        p_step = p_deg * np.pi / 180
        sol_j = run_closed_loop_case(X0, t_eval, rocket, control, p_step)
        t_j = sol_j.t
        X_j = sol_j.y.T
        p_j = X_j[:, 3]
        plt.plot(t_j, p_j * 180 / np.pi, linewidth=1.5, label=f"p_ref = {p_deg} deg/s")

    plt.xlabel("Time [s]")
    plt.ylabel("Roll rate p [deg/s]")
    plt.title("Influence of the command amplitude on the roll response")
    plt.grid(True)
    plt.legend()

    
    #Disturbance rejection
    
    M_dist = 0.5 #distrubnave moment magnitude

    sol_dist_cl = solve_ivp(
        fun=lambda t, X: ode_cl_disturb(t, X, p_step_main, M_dist, rocket, control),
        t_span=(0.0, 20.0),
        y0=X0,
        t_eval=t_eval,
        events=event_apogee,     #run disturbed closed loop
        rtol=1e-6,
        atol=1e-9
    )

    sol_dist_ol = solve_ivp(
        fun=lambda t, nu_: ode_open_disturb(t, nu_, M_dist, rocket),
        t_span=(0.0, 20.0),
        y0=nu0,
        t_eval=t_eval,   #run disturbed open loop
        rtol=1e-6,
        atol=1e-9
    )

    t_dist_cl = sol_dist_cl.t
    X_dist_cl = sol_dist_cl.y.T
    p_dist_cl = X_dist_cl[:, 3] #roll rate history for closed

    t_dist_ol = sol_dist_ol.t
    nu_dist_ol = sol_dist_ol.y.T
    p_dist_ol = nu_dist_ol[:, 3] #roll rate history for open

#Disturbnace plot for open and closed
    plt.figure()
    plt.plot(t_dist_ol, p_dist_ol * 180 / np.pi, "--", linewidth=1.5, label="Open-loop")
    plt.plot(t_dist_cl, p_dist_cl * 180 / np.pi, linewidth=1.5, label="Closed-loop PI")
    plt.xlabel("Time [s]")
    plt.ylabel("Roll rate p [deg/s]")
    plt.title("Rejection of a rolling impulse disturbance")
    plt.grid(True)
    plt.legend()

    plt.tight_layout()
    plt.show()

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
