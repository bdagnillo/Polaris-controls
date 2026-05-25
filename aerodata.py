import numpy as np
from scipy.interpolate import RegularGridInterpolator

# constants
R = 287 # gas constant for air
GAMMA = 1.4 # ratio of specific heats

# on flight, pressure and temp from sensors are directly used
# here, we set a pres and temp on the ground and use linear relation with
# altitude to simulate pres and temp change

p_abs_gounrd = 100500 # Pa, absolute pressure
t_abs_ground = 293 # Kelvin

# %% air quantity calculations
def temp_lapse(t_abs_ground: float, 
               agl: float # altitude above ground level in [meters]
               ) -> float: # [K]
    """
    Returns temperature at altitude using Linear temperature profile 
    with altitude, based on the last measured temperature before launch.
    """
    if t_abs_ground <= 273:
        raise ValueError("Temperature at ground level is too low. Please verify.")
    if agl < 0:
        raise ValueError("Altitude above ground cannot be negative.")
    
    return t_abs_ground - 0.0065 * agl

def pres_lapse(p_abs_ground: float,
               agl: float # altitude above ground level in [meters]
               ) -> float: # [K]
    if p_abs_ground <= 95000:
        raise ValueError("Pressure at ground level is too low. Please verify.")
    if agl < 0:
        raise ValueError("Altitude above ground cannot be negative.")
    
    return p_abs_ground - 10 * agl

def calculate_density(pres: float, temp: float) -> float: # [kg/m^3]
    if pres <= 0 or temp <= 0:
        raise ValueError("Pressure and temperature must be positive.")
    
    return pres / (temp * R)

def calculate_speed_of_sound(temp: float) -> float: # [m/s]
    if temp <= 0: 
        raise ValueError("Temperature must be positive.")
    
    return np.sqrt(GAMMA * R * temp)

def calculate_mach(temp: float, 
                   U: float # computed airspeed, in [m/s]
                   ) -> float: # [/]
    if U < 0:
        raise ValueError("Airspeed cannot be negative.")
    
    mach = U / calculate_speed_of_sound(temp)
    
    if mach > 0.9:
        raise ValueError("It's impossible for this rocket to reach Mach "
                         "number greater than 0.9.")
    
    return mach

def calculate_viscosity(temp: float) -> float: # [N*s/m^2]
    """
    Southerland law of viscosity
    """
    if temp <= 0:
        raise ValueError("Temperature must be positive.")
    
    mu0 = 1.716e-5 # [N*s/m^2]
    t0 = 273.15 # [K]
    s = 110.4 # [K]
    
    return mu0 * (t0 + s) * (temp / t0)**1.5 / (temp + s) 


def calculate_reynolds(pres: float, temp: float, U: float, l: float
                       ) -> float: # [/]
    # do the positive value checks
    #
    
    rho = calculate_density(pres, temp)
    mu = calculate_viscosity(temp)
    
    # l is the length of mean aerodynamic chord of the canard.
    # need to implement canard geometry data in params module
    # canards != fins
    return rho * U * l / mu

# %% canard lift coefficient look up using interpolation table

# independent variable grids
Re_values = np.array([0, 6.0e6, 6.0e7])
Ma_values = np.array([0, 0.4, 0.6, 0.8])
AoA_values = np.array([0, 1.1, 2.1, 3.1, 4.2, 5.2, 6.2, 7.2, 8.3, 9.3, 10.3], 
                      ) # degrees. treat as deflection angle

# lift coefficient table. dimension: [Re index, Ma index, AoA index]
cl_table = np.array(
    [
         [ # Re = 0
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Ma = 0
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Ma = 0.4
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Ma = 0.6
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Ma = 0.8
         ],
         
         [ # Re = 6.0e6
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Ma = 0
              [0, 0.004, 0.039, 0.070, 0.099, 0.143, 0.188, 0.237, 0.292, 0.344, 0.404], # Ma = 0.4
              [0, 0.044, 0.078, 0.119, 0.150, 0.183, 0.229, 0.286, 0.334, 0.389, 0.443], # Ma = 0.6
              [0, 0.029, 0.068, 0.113, 0.147, 0.186, 0.237, 0.296, 0.351, 0.405, 0.459], # Ma = 0.8
         ],
         
         [ # Re = 6.0e7
              [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0], # Ma = 0
              [0, 0.028, 0.062, 0.098, 0.129, 0.163, 0.201, 0.234, 0.280, 0.329, 0.393], # Ma = 0.4
              [0, 0.024, 0.060, 0.093, 0.131, 0.166, 0.205, 0.246, 0.297, 0.354, 0.423], # Ma = 0.6
              [0, 0.026, 0.066, 0.103, 0.142, 0.185, 0.273, 0.285, 0.338, 0.403, 0.468]  # Ma = 0.8
         ]
    ]    
    )

cl_interp = RegularGridInterpolator( # the interpolator
    (Re_values, Ma_values, AoA_values),
    cl_table,
    bounds_error=False,
    fill_value=None
    )
# note: in simulation, we will simplify that AoA is the same as deflection angle

# test point
# re_test = 1.5e5
# ma_test = 0.25
# aoa_test = 2.1
#cl_test = cl_interp([[re_test, ma_test, aoa_test]])[0]


# %% canard lift force (should this be in the dynamics section?)
def calculate_lift_force_per_canard(rho: float, U: float, A_ref: float, cl: float
                         ) -> float:
    return 0.5 * rho * U**2 * A_ref * cl
