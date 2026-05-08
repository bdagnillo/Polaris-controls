import numpy as np


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
