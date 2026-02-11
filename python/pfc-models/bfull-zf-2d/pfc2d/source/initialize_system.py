import cupy as cp
import numpy as np

def initialize_parms(pfc_parms, parm_dict):
    # Set parameters from dictionary
    for key, value in parm_dict.items():
        if hasattr(pfc_parms, key):
            setattr(pfc_parms, key, value)

    # Set derived parameters
    # document the following
    if pfc_parms.theta == 0.0:
        pfc_parms.mx = np.round(pfc_parms.lx / pfc_parms.nu * np.sqrt(3)).astype(int) # assuming dx~dy
        pfc_parms.my = np.round(pfc_parms.ly / pfc_parms.nu).astype(int)
        pfc_parms.dx = np.sqrt(3) * pfc_parms.mx / pfc_parms.my * pfc_parms.dy * pfc_parms.ly / pfc_parms.lx
        pfc_parms.qx0 = 4.0 * np.pi / (pfc_parms.lx / pfc_parms.mx * pfc_parms.dx)
        pfc_parms.qy0 = 2.0 * np.pi / (pfc_parms.ly / pfc_parms.my * pfc_parms.dy)
    if pfc_parms.theta == 30.0:
        pfc_parms.my = np.round(pfc_parms.ly / pfc_parms.nu * np.sqrt(3)).astype(int) # assuming dx~dy
        pfc_parms.mx = np.round(pfc_parms.lx / pfc_parms.nu).astype(int)
        pfc_parms.dy = np.sqrt(3) * pfc_parms.my / pfc_parms.mx * pfc_parms.dx * pfc_parms.lx / pfc_parms.ly
        pfc_parms.qx0 = 2.0 * np.pi / (pfc_parms.lx / pfc_parms.mx * pfc_parms.dx)
        pfc_parms.qy0 = 4.0 * np.pi / (pfc_parms.ly / pfc_parms.my * pfc_parms.dy)

def initialize_fields(pfc_data, pfc_parms, dict_settings=None):
    settings = {
        'phase': 's',
    }
    if dict_settings is not None:
        settings.update(dict_settings)
    
    # Set initial order parameters to disordered (liquid) state
    pfc_data.phiA = cp.ones((pfc_parms.lx, pfc_parms.ly), dtype=cp.float64)*pfc_parms.disordered
    pfc_data.phiB = cp.ones((pfc_parms.lx, pfc_parms.ly), dtype=cp.float64)*pfc_parms.disordered

    # If ordered state is desired, set order parameters to ordered (solid) state
    if settings['phase'] == 's':
        # lattice vectors
        sqrt3 = np.sqrt(3)
        q1 = np.array([-sqrt3/2, -1/2])
        q2 = np.array([0, 1])
        q3 = np.array([sqrt3/2, -1/2])
        
        x = np.arange(pfc_parms.lx) * pfc_parms.dx
        y = np.arange(pfc_parms.ly) * pfc_parms.dy
        X, Y = np.meshgrid(x, y)
        pfc_data.phiA = (
            cp.ones((pfc_parms.lx, pfc_parms.ly), dtype=cp.float64)*pfc_parms.ordered +
            pfc_parms.ordered_amplitude * (cp.cos(q1[0]*X + q1[1]*Y) + cp.cos(q2[0]*X + q2[1]*Y) + cp.cos(q3[0]*X + q3[1]*Y)))
        pfc_data.phiB = (
            cp.ones((pfc_parms.lx, pfc_parms.ly), dtype=cp.float64)*pfc_parms.ordered +
            pfc_parms.ordered_amplitude * (cp.cos(q1[0]*X + q1[1]*Y) + cp.cos(q2[0]*X + q2[1]*Y) + cp.cos(q3[0]*X + q3[1]*Y)))

    # Set initial fields
    pass