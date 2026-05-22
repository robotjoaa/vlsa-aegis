import numpy as np
from scipy.ndimage import distance_transform_edt, binary_erosion
from scipy.sparse import diags
from scipy.sparse.linalg import spsolve
from scipy.ndimage import map_coordinates

def poisson_cbf_from_occupancy(occ_grid):
    """
    Simple version to construct the poisson field w/o guidance field
    """
    ny, nx = occ_grid.shape
    dx = dy = 1.0

    dist_out = distance_transform_edt(occ_grid == 0)
    dist_in = distance_transform_edt(occ_grid == 1)
    # positive in free space, negative inside obstacles 
    signed_dist = dist_out - dist_in

    obs_interior = binary_erosion(occ_grid)
    obs_boundary = (occ_grid == 1) & (~obs_interior)

    # forcing function 
    f = -signed_dist / (np.max(np.abs(signed_dist)) + 1e-10)

    N = nx * ny
    main_diag = -4.0 * np.ones(N)
    side_diag = np.ones(N - 1)
    side_diag[np.arange(1, N) % nx == 0] = 0
    up_down_diag = np.ones(N - nx)
    A = diags([main_diag, side_diag, side_diag, up_down_diag, up_down_diag],
              [0, -1, 1, -nx, nx], format='csc') / dx**2

    f_flat = f.ravel() # flatten with reference 
    mask_boundary = obs_boundary.ravel()
    A = A.tolil()
    for i in np.where(mask_boundary)[0]:
        A[i, :] = 0
        A[i, i] = 1.0
        f_flat[i] = 0.0
    # compressed sparse column 
    A = A.tocsc()

    h_flat = spsolve(A, f_flat)
    h = h_flat.reshape((ny, nx))
    return h


def cbfqp_poisson(state, nominal_control, h_field, alpha=1.0, grid_res=0.05):
    """
    CF solution using the poisson field as h
    Args:
        state: np.array([x, y]) in meters
        nominal_control: np.array([vx, vy])
        h_field: dict with keys 'h', 'dx', 'dy'
        alpha: relaxation gain
        grid_res: meter per grid cell
    """
    # Map state to grid coordinates
    x, y = state
    ny, nx = h_field['h'].shape
    gx = x / grid_res
    gy = y / grid_res

    # Interpolate h and gradients
    h_val = map_coordinates(h_field['h'], [[gy], [gx]], order=1)[0]
    dhx = map_coordinates(h_field['dhx'], [[gy], [gx]], order=1)[0]
    dhy = map_coordinates(h_field['dhy'], [[gy], [gx]], order=1)[0]
    # grad_h = np.array([dhx, dhy]).reshape(1, -1)
    grad_h = np.array([dhx, dhy]).flatten()

    # CBF constraint
    cbf_constr = (grad_h @ nominal_control + alpha * h_val).squeeze()
    if cbf_constr >= 0:
        return nominal_control.squeeze()
    else:
        correction = (cbf_constr / (grad_h @ grad_h.T).squeeze()) * grad_h.T
        return (nominal_control - correction).squeeze()


def clfqp(state, target_state, alpha=1.0):
    """
    Args:
        state: numpy array of shape (2,) for XY position
        target_state: numpy array of shape (2, ) for goal XY position
        alpha: tuning parameter of the CLF, the larger the more aggressive
    """
    nominal_control = np.zeros((2,))
    v = np.linalg.norm(state - target_state) ** 2
    Lfv = np.zeros((1,))
    Lgv = 2 * (state - target_state)
    clf_constr = (Lfv + Lgv @ nominal_control + alpha * v).squeeze()
    if clf_constr <= 0:
        return nominal_control.squeeze()
    else:
        return (nominal_control - clf_constr / (Lgv @ Lgv.T).squeeze() * Lgv.T).squeeze()

