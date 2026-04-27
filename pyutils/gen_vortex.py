import numpy as np

def gen_vortex(x, y, c, order):
    """
    Generate vortex phase.
    
    Parameters:
    -----------
    x, y : ndarray
        Coordinate grids
    c : list
        Center coordinates [cx, cy]
    order : int
        Topological charge
    
    Returns:
    --------
    phase : ndarray
        Vortex phase
    """
    phase = np.arctan2(y - c[1], x - c[0])
    phase = phase * order
    
    return phase
