from pcbf import poisson_cbf_from_occupancy, cbfqp_poisson, clfqp
from robot_sim import OmniDirRobotSim
import numpy as np
from IPython.display import display, clear_output
import imageio

import matplotlib
matplotlib.use("Qt5Agg", force=True)
import matplotlib.pyplot as plt
# Setup the map
grid_res = 0.05  # 5 cm per cell
nx, ny = int(10 / grid_res), int(10 / grid_res)
occ = np.zeros((ny, nx))
occ[int(2/grid_res):int(5/grid_res), int(3/grid_res):int(6/grid_res)] = 1  # obstacle 1
occ[int(6/grid_res):int(7.5/grid_res), int(6/grid_res):int(7.0/grid_res)] = 1  # obstacle 2
occ = np.flipud(occ)

plt.imshow(occ)
plt.axis('off')
plt.show()

# Compute h from Poisson (non-obstacle h > 0; on obstacle edge h = 0; inside obstacle h < 0)
h_im = poisson_cbf_from_occupancy(occ)
# Flip vertically -> world coordinates (y up)
h_world = np.flipud(h_im)
dhy_world, dhx_world = np.gradient(h_world, grid_res, grid_res)
h_field = {'h': h_world, 'dhx': dhx_world, 'dhy': dhy_world}
plt.imshow(h_im)
plt.colorbar()
plt.axis('off')
plt.show()

# Run simulation
robot = OmniDirRobotSim(10, 10, 0.2)
goal = np.array([8.0, 8.0])

for _ in range(100):
    nominal_control = clfqp(robot.position, goal, alpha=1.0)
    assert nominal_control.shape[0] == 2
    control_input = cbfqp_poisson(robot.position, nominal_control, h_field, alpha=0.3, grid_res=grid_res)
    robot.move(*control_input)
    robot.visualize(control_input, nominal_control, goal, occ, h_field, grid_res)

robot.save_gif("poisson_cbf_robot.gif")