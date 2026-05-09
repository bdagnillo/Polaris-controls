import numpy as np


def hat(vec: np.ndarray) -> np.ndarray: #cross product
    return np.array([
        [0.0, -vec[2], vec[1]],
        [vec[2], 0.0, -vec[0]],
        [-vec[1], vec[0], 0.0]
    ])

# phi: roll, theta: pitch, psi: yaw
def Cba(psi: float, theta: float, phi: float) -> np.ndarray:
    #transform vector #transform vector from inertial frame to body frame
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


def Sba(phi: float, theta: float) -> np.ndarray:
    # transforms Euler angle rates to body angular velocities   
    # p: roll, q: pitch, r: yaw
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
