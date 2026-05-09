import numpy as np
from dataclasses import dataclass


@dataclass
class RocketParams:
    rho: float = 1.225 #TBD
    mass: float = 5 # kg TBD
    g: float = 9.81 # m/s^2

    # all lengths in meters
    R: float = 0.0762 / 2 # rocket radius
    l: float = 1.9558 # total length maybe??? where does 1.9558 comes from
    l_body: float = 0.711 # body length
    K_body: float = 1.1 # correction factor for body tube normal force

    # fin geometry
    num_fins: int = 3 # number of fins
    s_fin: float = 0.0635 #fin span
    gamma_c: float = 0.2915
    # in Matteo's report gamma_c is called fineness parameter, but
    # actually this is actually mid-chord sweep angle [rad]
    t_fin: float = 0.00508 #thickness
    C_r: float = 0.0889 #root chord
    C_t: float = 0.0889 - 0.0381 #tip chord
    c_barre: float = 0.0716 # mean aerodynamic chord
    cant: float = 0.1 * np.pi / 180 # cant angle [rad]

    Jx: float = 0.02
    Jy: float = 0.2 #inertia
    Jz: float = 0.2

    Tmax: float = 10.0 #max thrust
    tburn: float = 2.5 # burn time
    fade_time: float = 15.0

    r_ag_x: float = 0.134 # not sure what it means, but it is used 
    # in pitch moment equation

    @property
    def A_ref(self) -> float:
        return np.pi * self.R**2

    @property
    def d(self) -> float: #rocket diameter
        return 2 * self.R

    @property
    def A_fin(self) -> float: #fin area (one side)
        return (self.C_r + self.C_t) * self.s_fin / 2

    @property
    def y_mac(self) -> float: # spanwise distance from the fin root 
                              # to the mean aerodynamic chord
        return (self.s_fin / 3) * ((self.C_r + 2 * self.C_t) / (self.C_r + self.C_t))

    @property
    def K_TB(self) -> float: #correction factor for normal force
                             # accounting for body-fin interference
        return 1 + self.R / (self.R + self.s_fin)

    def __post_init__(self):
        self._J    = np.diag([self.Jx, self.Jy, self.Jz])
        self._Mmat = np.block([
            [self.mass * np.eye(3), np.zeros((3, 3))],
            [np.zeros((3, 3)),      self._J          ]
        ])# 6x6 combined mass-inertia matrix

    @property
    def J(self) -> np.ndarray: # rotational inertia matrix
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
