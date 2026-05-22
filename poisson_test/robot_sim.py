import numpy as np
from IPython.display import display, clear_output
import imageio
import matplotlib.pyplot as plt

# Define the simulator
class OmniDirRobotSim:
    def __init__(self, world_width, world_height, robot_size):
        self.world_width = world_width
        self.world_height = world_height
        self.robot_size = robot_size
        self.position = np.array([0.5, 0.5])
        self.trajectory = []
        self.frames = []

    def move(self, vel_x, vel_y, dt=0.1):
        new_position = self.position + np.array([vel_x, vel_y]) * dt
        if 0 <= new_position[0] <= self.world_width and 0 <= new_position[1] <= self.world_height:
            self.position = new_position
        self.trajectory.append(tuple(self.position))

    def visualize(self, control_input, nominal_control, goal_position, occ_map, h_field, grid_res, save_frame=True):
        clear_output(wait=True)
        fig, ax = plt.subplots(figsize=(6, 6))

        # Background: show Poisson-CBF field h(x,y)
        extent = [0, occ_map.shape[1]*grid_res, 0, occ_map.shape[0]*grid_res]
        im = ax.imshow(h_field['h'], cmap='RdBu_r', origin='lower', extent=extent, alpha=0.7)
        ax.contour(
            np.linspace(0, extent[1], occ_map.shape[1]),
            np.linspace(0, extent[3], occ_map.shape[0]),
            h_field['h'],
            levels=[0],
            colors='black',
            linewidths=1.5
        )

        ax.set_xlim(0, self.world_width)
        ax.set_ylim(0, self.world_height)
        ax.set_aspect('equal')
        ax.grid(True, linestyle='--', alpha=0.4)

        # Plot trajectory
        if self.trajectory:
            traj_x, traj_y = zip(*self.trajectory)
            ax.plot(traj_x, traj_y, 'b-', alpha=0.7, linewidth=1.5, label='Trajectory')

        # Draw robot and goal
        rx, ry = self.position
        robot = plt.Circle((rx, ry), self.robot_size, color='blue', alpha=0.8, label='Robot')
        ax.add_patch(robot)

        gx, gy = goal_position
        goal = plt.Circle((gx, gy), 0.25, color='orange', alpha=1.0, label='Goal')
        ax.add_patch(goal)

        # Control arrows
        vx_nom, vy_nom = nominal_control
        ax.arrow(rx, ry, vx_nom, vy_nom, head_width=0.2, fc='r', ec='r', label='Nominal', alpha=0.8)

        vx_safe, vy_safe = control_input
        ax.arrow(rx, ry, vx_safe, vy_safe, head_width=0.2, fc='g', ec='g', label='Safe', alpha=0.8)

        ax.legend(loc='upper right', fontsize=8)
        ax.set_xlabel("X [m]")
        ax.set_ylabel("Y [m]")
        ax.set_title("Poisson-CBF Field and Robot Trajectory")

        if save_frame:
            fig.canvas.draw()
            frame = np.array(fig.canvas.renderer.buffer_rgba())
            self.frames.append(frame)

        display(fig)
        plt.close(fig)


    def save_gif(self, filename="robot_simulation.gif", fps=20):
        imageio.mimsave(filename, self.frames, fps=fps, loop=0)



