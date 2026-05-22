import collections
import dataclasses
import logging
import math
import pathlib
import time
import imageio
from libero.libero import benchmark
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv
import numpy as np
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tqdm
import tyro
import mujoco
import cvxpy as cp
from scipy.spatial.transform import Rotation as R
from utils import rot3, quat_R, quat_euler, vector_hat, project_matrix, \
    compute_h_ij, compute_h_coeffs_3d, get_point_cloud, filtering_points, fit_ellipse, plot_points_ellipse, \
    obstacle_detection
import warnings
warnings.filterwarnings("ignore")
from typing import List

LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 1024  # resolution used to render training data
OBSTACLE_POS = np.array([-0.15, 0.03, 1.17])
OBSTACLE_RADIUS = 0.06
ALPHA = 1.0                 # CBF gain
MAX_VEL = 1.0               # maximum EE speed 


@dataclasses.dataclass
class Args:
    #################################################################################################################
    # Model server parameters
    #################################################################################################################
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = (
        "safelibero_spatial"  # Task suite. Options: libero_spatial, libero_object, libero_goal, libero_10, libero_90
    )
    safety_level: str = "I" # Task level. Options: I, II
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0]) # Options: [0, 1, 2, 3]
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0]) # Options: [0, 1, 2, 3, 4, ..., 49]
    num_steps_wait: int = 20  # Number of steps to wait for objects to stabilize i n sim
    num_trials_per_task: int = 50  # Number of rollouts per task

    #################################################################################################################
    # Utils
    #################################################################################################################
    video_out_path: str = "results_new"  # Path to save videos

    seed: int = 7  # Random Seed (for reproducibility)


def eval_libero(args: Args) -> None:
    # Set random seed
    np.random.seed(args.seed)
    safety_level = args.safety_level
    task_index = args.task_index
    episode_index = args.episode_index
    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=safety_level)
    num_tasks_in_suite = task_suite.n_tasks
    logging.info(f"Task suite: {args.task_suite_name}, safety level: {safety_level}")

    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)

    if args.task_suite_name == "safelibero_spatial":
        max_steps = 300  # longest training demo has 193 steps
    elif args.task_suite_name == "safelibero_object":
        max_steps = 300  # longest training demo has 254 steps
    elif args.task_suite_name == "safelibero_goal":
        max_steps = 300  # longest training demo has 270 steps
    elif args.task_suite_name == "safelibero_10":
        max_steps = 550  # longest training demo has 505 steps
    elif args.task_suite_name == "safelibero_90":
        max_steps = 400  # longest training demo has 373 steps
    else:
        raise ValueError(f"Unknown task suite: {args.task_suite_name}")

    print("OK")
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)
    from groundingdino.util.inference import load_model, load_image, predict, annotate
    import cv2
    CONFIG_PATH = "GroundingDINO/config/GroundingDINO_SwinT_OGC.py"    #configuration file that comes with the source code
    CHECKPOINT_PATH = "GroundingDINO/groundingdino_swint_ogc.pth"   #downloaded weight
    # CONFIG_PATH = "GroundingDINO/config/GroundingDINO_SwinB_cfg.py"
    # CHECKPOINT_PATH = "GroundingDINO/groundingdino_swinb_cogcoor.pth"
    # DEVICE = "cuda"   #可以选择cpu/cuda
    BOX_TRESHOLD = 0.35     #bbox determination threshold 
    TEXT_TRESHOLD = 0.25    #key attributre threshold from the text

    model_groundingdino = load_model(CONFIG_PATH, CHECKPOINT_PATH)
    # Start evaluation
    total_episodes, total_successes = 0, 0
    # for task_id in tqdm.tqdm(range(num_tasks_in_suite)):
    for task_id in task_index:
        # Get tasktask_index
        task = task_suite.get_task(task_id)

        # Get default LIBERO initial states
        initial_states = task_suite.get_task_init_states(task_id)

        # Initialize LIBERO environment and task description
        env, task_description = _get_libero_env(task, safety_level, LIBERO_ENV_RESOLUTION, args.seed)
        model = env.sim.model
        data = env.sim.data

        collides = 0
        time_steps = []

        # Start episodes
        task_episodes, task_successes = 0, 0
        task_segment = task_description.replace(" ", "_")



        _out_dir = pathlib.Path(args.video_out_path) / f"{task_segment}"
        _out_dir.mkdir(parents=True, exist_ok=True)
        out_dir = _out_dir / f"ours_{safety_level}"
        out_dir.mkdir(parents=True, exist_ok=True)



        # for episode_idx in tqdm.tqdm(range(args.num_trials_per_task)):
        for episode_idx in episode_index:
            logging.info(f"\nTask: {task_description}")

            ### 1. Reset environment
            env.reset()
            action_plan = collections.deque()

            # print("Joint names (qpos):", env.sim.model.joint_names)

            # Set initial states
            # initial_states[episode_idx][67] = initial_states[episode_idx][67] - 0.02
            # initial_states[episode_idx][53] = initial_states[episode_idx][53] - 0.02
            # print(initial_states[episode_idx])
            obs = env.set_init_state(initial_states[episode_idx])

            # Setup
            t = 0
            replay_images = []
            model = env.sim.model
            data = env.sim.data
            eef_body_id = model.body_name2id("eef_marker")

            # Initial position and attitude angle of the terminal ellipsoid
            eef_pos = obs["robot0_eef_pos"]
            eef_quat = obs["robot0_eef_quat"]
            r = R.from_quat(eef_quat)
            euler1 = r.as_euler('xyz', degrees=False)
            R1 = R.from_quat(eef_quat).as_matrix()
            offset_local = np.array([0, 0, -0.08])
            offset_world = R1 @ offset_local
            ball_pos = eef_pos + offset_world
            p1 = ball_pos
            env.sim.model.body_pos[eef_body_id] = ball_pos
            env.sim.model.body_quat[eef_body_id] = eef_quat[[3, 0, 1, 2]]

            # Larger EE elipsoid for holding longer object
            
            if "orange juice" in task_description or "milk" in task_description or "alphabet soup" in task_description:
                Q1_diag = np.array([0.06, 0.12, 0.2])
            else:
                Q1_diag = np.array([0.06, 0.12, 0.11])

            while t < args.num_steps_wait:
                try:
                    # IMPORTANT: Do nothing for the first few timesteps because the simulator drops objects
                    # and we need to wait for them to fall
                    if t < args.num_steps_wait:
                        obs, reward, done, info = env.step(LIBERO_DUMMY_ACTION)
                        # img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                   
                        # # Save preprocessed image for replay video
                        # replay_images.append(img)
                        t += 1
                        continue
                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break


            
            ### 2. Detect obstacles
            agentview_img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
            agentview_depth = np.ascontiguousarray(obs["agentview_depth"][::-1, ::-1])
            # import matplotlib
            # matplotlib.use('Agg')          
            # import matplotlib.pyplot as plt
            # plt.imsave(f"images/{episode_idx}.png", agentview_img)
            # continue

            img_out_dir = out_dir/f"{episode_idx}"
            img_out_dir.mkdir(parents=True, exist_ok=True)
            # Get the obstacle most likely to be hit
            # obstacle_infromation = obstacle_detection(agentview_img, task_description, args.task_suite_name)
            obstacle_information ="red milk carton"
            # obstacle_infromation = "yellow rectangular book"
            agent_view_points = get_point_cloud(agentview_img, agentview_depth, env, "agentview", obstacle_information, model_groundingdino, img_out_dir)
            # import pandas as pd
            # df = pd.DataFrame(agent_view_points, columns=["X", "Y", "Z"])
            # df.to_csv("agent_view_points.csv", index=False)
            
            backview_img = np.ascontiguousarray(obs["backview_image"][::-1, ::-1])
            backview_depth = np.ascontiguousarray(obs["backview_depth"][::-1, ::-1])
            back_view_points = get_point_cloud(backview_img, backview_depth, env, "backview", obstacle_information, model_groundingdino, img_out_dir)
            
            # df = pd.DataFrame(back_view_points, columns=["X", "Y", "Z"])
            # df.to_csv("back_view_points.csv", index=False)
            
            if agent_view_points.shape[1] > 0 and back_view_points.shape[1] > 0:
                full_points = np.vstack([agent_view_points, back_view_points])    # (Na + Nb, 3)
            elif agent_view_points.shape[1] == 0 and back_view_points.shape[1] > 0:
                full_points = back_view_points
            elif agent_view_points.shape[1] > 0 and back_view_points.shape[1] == 0:
                full_points = agent_view_points
            else:
                full_points = np.array([[]])

            # df = pd.DataFrame(full_points, columns=["X", "Y", "Z"])
            # df.to_csv("full_points.csv", index=False)

            # Point cloud filtering 
            filter_points = filtering_points(full_points)
            # print("过滤后的点云数量：", filter_points.shape[0])
            flag_safety_control = True
            if filter_points.shape[0] == 0:
                flag_safety_control = False
            # import pandas as pd
            # df = pd.DataFrame(filter_points, columns=["X", "Y", "Z"])
            # df.to_csv("filter_points.csv", index=False)

            if flag_safety_control:
                p2, R2, Q2_diag = fit_ellipse(filter_points, plot=True, save_path=img_out_dir)
                # 控制参数设置
                z_fixed = (p2 - p1)
                z_fixed /= np.linalg.norm(z_fixed)
                p_target = np.array([-0.05, 0.15, 1.05])
                Kp_pos = 1
                dt = 0.05
            t = 0
            # print("Joint names (qpos):", env.sim.model.joint_names)

            # Extract all obstacle names from the joint list
            obstacle_names = [n.replace('_joint0', '') for n in env.sim.model.joint_names if 'obstacle' in n]

            # Identify the active obstacle within the workspace bounds
            obstacle_name = " "
            for i in obstacle_names:
                p = obs[f"{i}_pos"]  # Get position from observation
                # Check if the object is within the valid workspace range
                if p[2] > 0 and -0.5 < p[0] < 0.5 and -0.5 < p[1] < 0.5:
                    obstacle_name = i
                    print("Obstacle name:", i)
                    break
            initial_obstacle_pos = obs[obstacle_name + "_pos"]
            collide_flag = False
            collide_time = 0

            logging.info(f"Starting episode {task_episodes+1}...")
            while t < max_steps:
                try:

                    # Get preprocessed image
                    # IMPORTANT: rotate 180 degrees to match train preprocessing
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                     # Save preprocessed image for replay video
                    replay_images.append(img)
                    img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                    )
                    wrist_img = image_tools.convert_to_uint8(
                        image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                    )
                    

                   
                    t1 = time.time()
                    if not action_plan:
                        # Finished executing previous action chunk -- compute new chunk
                        # Prepare observations dict
                        element = {
                            "observation/image": img,
                            "observation/wrist_image": wrist_img,
                            "observation/state": np.concatenate(
                                (
                                    obs["robot0_eef_pos"],
                                    _quat2axisangle(obs["robot0_eef_quat"]),
                                    obs["robot0_gripper_qpos"],
                                )
                            ),
                            "prompt": str(task_description),
                        }

                        # Query model to get action
                        action_chunk = client.infer(element)["actions"]
                        assert (
                            len(action_chunk) >= args.replan_steps
                        ), f"We want to replan every {args.replan_steps} steps, but policy only predicts {len(action_chunk)} steps."
                        action_plan.extend(action_chunk[: args.replan_steps])

                    t2 = time.time()
                    # print("t={}, inference time={}".format(t, t2-t1))

                    action = action_plan.popleft()
                    t3 = time.time()
                    if flag_safety_control:
                        # pos_err = p_target - p1
                        # v_ref = Kp_pos * pos_err
                        action_movement = np.zeros_like(action.tolist())
                        action_movement[:3] = action[:3]
                        action_movement[6] = action[6]
                        

                        v_ref =  Kp_pos * R1.T @ action_movement[:3]
                        u_v_ref = 5 * v_ref

                        a_v, a_omega, a_uz, h, mu_row = compute_h_coeffs_3d(p1, Q1_diag, R1, p2, Q2_diag, R2, z_fixed)
                        a_u_v = 0.2 * a_v

                        u_z_nom = 10 * mu_row
                        u = cp.Variable(6)  # [v_x, v_y, omega, u_zx, u_zy]

                        # ---  Weighted cost function ---
                        W = np.diag([1.0/25, 1.0/25, 1.0/25, 1.0, 1.0, 1.0])  
                        u_ref_vec = np.hstack([u_v_ref, u_z_nom])
                        objective = cp.Minimize(cp.quad_form(u - u_ref_vec, W))
                        # --- Linear constraints ---
                        constraints = [
                            a_u_v @ u[:3] + a_uz @ u[3:6] + 10 * h >= 0
                        ]
                        # --- Solve QP ---
                        prob = cp.Problem(objective, constraints)
                        prob.solve(solver=cp.OSQP)
                        # --- Read optimization results ---
                        if u.value is not None:
                            u_v = u.value[:3]
                            # u_omega = u.value[3:6]
                            u_z = u.value[3:]
                        else:
                            u_v = v_ref # why not action[:3]
                            # u_omega = omega_ref
                            u_z = u_z_nom
                            print("No feasible solution")
                            
                        # print("t={}".format(t))
                        # print("u:", u.value)

                        Id = np.eye(len(z_fixed))
                        dz = (Id - np.outer(z_fixed, z_fixed)) @ u_z
                        z_fixed = z_fixed + dz * dt
                        z_fixed = z_fixed / np.linalg.norm(z_fixed)
                        # print("z_fixed:", z_fixed)

                    

                        action_input = np.zeros(7)
                        action_input[:3] = 0.2 * R1 @ u_v
                        action_input[6] = action[6]  # Keep gripper closed 
                        # print("action_input:", action_input)
                        t4 = time.time()

                        obs, reward, done, info = env.step(action_input.tolist()) # Crucial step
                    else:
                        t4 = time.time()

                        obs, reward, done, info = env.step(action.tolist()) # Crucial step


                    
                    # print("t={}, safety layer time={}".format(t, t4-t3))
                    if collide_flag == False:
                        then_obstacle_pos = obs[obstacle_name + "_pos"]
                        # print(np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)))
                        if np.sum(np.abs(then_obstacle_pos - initial_obstacle_pos)) > 0.001:
                            print("obstacle collided")
                            collide_flag = True
                            collide_time = t

            

                    
                    eef_pos = obs["robot0_eef_pos"]
                    eef_quat = obs["robot0_eef_quat"]
                    r = R.from_quat(eef_quat)
                    euler = r.as_euler('xyz', degrees=False)
                    R1 = R.from_quat(eef_quat).as_matrix()
                    offset_local = np.array([0, 0, -0.08])
                    offset_world = R1 @ offset_local
                    ball_pos = eef_pos + offset_world
                    env.sim.model.body_pos[eef_body_id] = ball_pos
                    env.sim.model.body_quat[eef_body_id] = eef_quat[[3, 0, 1, 2]]
                    p1 = ball_pos


                    if done:
                        task_successes += 1
                        total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Caught exception: {e}")
                    break

            task_episodes += 1
            total_episodes += 1

            time_steps.append(t)
            if collide_flag == True:
                collides += 1

            suffix = "success" if done else "failure"
            
            video_path = out_dir / f"{episode_idx}_{suffix}.mp4"
            imageio.mimwrite(
                video_path,
                [np.asarray(x) for x in replay_images],
                fps=30,
            )

            # Log current results
            logging.info(f"Success: {done}")
            logging.info(f"# episodes completed so far: {total_episodes}")
            logging.info(f"Collision: {collide_flag}")
            logging.info(f"# successes: {total_successes} ({total_successes / total_episodes * 100:.1f}%)")
            logging.info(f"# collides: {collides} ({collides / total_episodes * 100:.1f}%)")
            print("collide_flag:", collide_flag)
            print("collide_time:", collide_time)


        # Log final results
        logging.info(f"Current task success rate: {float(task_successes) / float(task_episodes)}")
        logging.info(f"Current total success rate: {float(total_successes) / float(total_episodes)}")

    logging.info(f"Total success rate: {float(total_successes) / float(total_episodes)}")
    logging.info(f"Total episodes: {total_episodes}")
    logging.info(f"Time steps: {time_steps}")


def _get_libero_env(task, level, resolution, seed):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = pathlib.Path(get_libero_path("bddl_files")) / task.problem_folder / task.bddl_file
    print(task_description)
    env_args = {"bddl_file_name": task_bddl_file, "camera_heights": resolution, "camera_widths": resolution, "camera_depths": True}
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def _quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero(args)