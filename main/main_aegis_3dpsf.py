"""Evaluate the 3DPSF safety filter in the safelibero environment.

Drop-in replacement for main_aegis.py that swaps the VLSA ellipsoid CBF for
the full 3DPSF pipeline (psfjax steps 1–4).

Pipeline per episode
--------------------
  OFFLINE (once per episode, after env.set_init_state):
    1. Extract obstacle geoms from safelibero env via psfjax.scene
    2. Build 100×100×100 buffered safe region C_ε (SafeRegionBuilder)
    3. Solve 3D Poisson equation on C_ε (PoissonSolver, JAX SOR)

  ONLINE (every control step):
    4. Sync new-mujoco robot model with current arm joints q
    5. Sample robot surface points y_i and compute J_i(q) (MuJoCo Jacobians)
    6. Query h(y_i) and ∂h/∂y_i from PoissonResult (trilinear interpolation)
    7. Filter nominal joint velocity through CBF QP (PSFFilter, qpax via cbfpy)
    8. Convert safe joint velocity back to EEF delta action for env.step()

Joint ↔ EEF action conversion
-------------------------------
  safelibero action: 7D [Δx, Δy, Δz, Δax, Δay, Δaz, gripper] (EEF delta, OSC controller)
  3DPSF filter:      operates on 7D joint velocities
  Bridge:  v_joints = J_eef† @ action[:6]   (pseudo-inverse of 6×7 EEF Jacobian)
           action_safe[:6] = J_eef @ v_safe  (back-projection)

Usage
-----
  conda activate libero
  source main/.venv/bin/activate
  export PYTHONPATH=$(pwd):$PYTHONPATH
  cd main/
  python main_aegis_3dpsf.py --task_suite_name safelibero_spatial \\
      --safety_level II --task_index 0 --episode_index 0
"""

from __future__ import annotations

import collections
import dataclasses
import logging
import math
import pathlib
import time
import warnings
from typing import List, Optional

import imageio
import numpy as np
from scipy.spatial.transform import Rotation as R

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# safelibero / robosuite (mujoco-py API)
# --------------------------------------------------------------------------
from libero.libero import benchmark, get_libero_path
from libero.libero.envs import OffScreenRenderEnv
from openpi_client import image_tools
from openpi_client import websocket_client_policy as _websocket_client_policy
import tyro

# --------------------------------------------------------------------------
# new mujoco (psfjax uses mujoco 3.x API separately from env.sim)
# --------------------------------------------------------------------------
import mujoco as mj

# --------------------------------------------------------------------------
# psfjax pipeline
# --------------------------------------------------------------------------
import sys, os
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from psfjax.occupancy import GridBounds
from psfjax.robot import RobotSurface
from psfjax.safe_region import SafeRegionBuilder, SafeRegionResult
from psfjax.scene import get_obstacles_from_robosuite_sim
from psfjax.poisson import PoissonSolver, PoissonResult
from psfjax.qp import PSFFilter, PSFFilterResult

# --------------------------------------------------------------------------
# Constants
# --------------------------------------------------------------------------
LIBERO_DUMMY_ACTION = [0.0] * 6 + [-1.0]
LIBERO_ENV_RESOLUTION = 256  # reduced for speed; use 1024 for full eval

# FR3/Panda workspace bounds and 100×100×100 grid (per 3DPSF paper)
WORKSPACE_BOUNDS = GridBounds(
    x_min=-1.0, x_max=1.0,
    y_min=-1.0, y_max=1.0,
    z_min=0.0,  z_max=2.0,
)
VOXEL_SIZE = 0.02        # 100×100×100 grid at 2 m range
EPSILON    = 0.10        # Poisson disk / buffer radius [m]
CBF_ALPHA  = 10.0        # CBF class-K gain
POISSON_F  = -1.0        # Forcing function value (must be < 0)
POISSON_ITERS = 1000     # SOR iterations (warm-started after first solve)

# Panda robot XML (robosuite bundled with main/.venv)
_ASSETS = (
    pathlib.Path(__file__).resolve().parent
    / ".venv/lib/python3.11/site-packages/robosuite/models/assets"
)
ROBOT_XML = _ASSETS / "robots/panda/robot.xml"
GRIPPER_XML = _ASSETS / "grippers/panda_gripper.xml"
# EEF body name in robosuite panda/robot.xml (the body attached to joint 7)
EEF_BODY_NAME = "right_hand"   # adjust if different in your robot.xml


# --------------------------------------------------------------------------
# CLI args
# --------------------------------------------------------------------------

@dataclasses.dataclass
class Args:
    host: str = "0.0.0.0"
    port: int = 8000
    resize_size: int = 224
    replan_steps: int = 5

    task_suite_name: str = "safelibero_spatial"
    safety_level: str = "II"
    task_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    episode_index: List[int] = dataclasses.field(default_factory=lambda: [0])
    num_steps_wait: int = 20
    num_trials_per_task: int = 50

    video_out_path: str = "results_3dpsf"
    save_psf_vis: bool = True       # save PSF slice/volume images per episode
    use_jax: bool = True            # use JAX GPU for SOR and PSF operations
    seed: int = 7


# --------------------------------------------------------------------------
# PSF3DLayer — manages the entire psfjax pipeline for one episode
# --------------------------------------------------------------------------

class PSF3DLayer:
    """Manages the 3DPSF pipeline (steps 1–4) for safelibero evaluation.

    Lifecycle:
      layer = PSF3DLayer(robot_xml, n_arm_joints=7)
      layer.rebuild(env, q_arm, seed)      # offline, once per episode
      action_safe = layer.filter_action(action_nom, obs)   # per timestep
    """

    def __init__(
        self,
        robot_xml: pathlib.Path,
        n_arm_joints: int = 7,
        alpha: float = CBF_ALPHA,
        voxel_size: float = VOXEL_SIZE,
        epsilon: float = EPSILON,
        use_jax: bool = True,
        gripper_xml: pathlib.Path | None = None,
    ) -> None:
        if not robot_xml.exists():
            raise FileNotFoundError(f"Robot XML not found: {robot_xml}")

        self.n_arm = n_arm_joints
        self.alpha = alpha
        self.voxel_size = voxel_size
        self.epsilon = epsilon
        self.use_jax = use_jax

        # --- Step 1 setup: robot surface sampler (new mujoco) ----------------
        _gripper = gripper_xml if (gripper_xml is not None and gripper_xml.exists()) else None
        if _gripper is not None:
            self.robot_surface = RobotSurface.from_xml_paths(robot_xml, _gripper)
        else:
            self.robot_surface = RobotSurface.from_xml_path(robot_xml)
        self._mj_model = self.robot_surface._model
        self._mj_data = self.robot_surface._data
        self.n_mj_joints = self._mj_model.nq   # may be > 7 if fingers in XML

        # Find EEF body id in new mujoco model
        try:
            self._eef_body_id = mj.mj_name2id(
                self._mj_model, mj.mjtObj.mjOBJ_BODY, EEF_BODY_NAME
            )
        except Exception:
            # Fall back to the last body in the kinematic chain
            self._eef_body_id = self._mj_model.nbody - 1
            logging.warning(
                f"EEF body '{EEF_BODY_NAME}' not found; using body id "
                f"{self._eef_body_id} ({mj.mj_id2name(self._mj_model, mj.mjtObj.mjOBJ_BODY, self._eef_body_id)})"
            )

        # --- Step 2 setup: SafeRegionBuilder ---------------------------------
        self.safe_builder = SafeRegionBuilder(
            robot_xml_path=robot_xml,
            bounds=WORKSPACE_BOUNDS,
            voxel_size=voxel_size,
            epsilon=epsilon,
            n_surface_pts_per_prim=300,
            use_jax=use_jax,
            gripper_xml_path=_gripper,
        )

        # --- Step 4 setup: PSFFilter (qpax) ----------------------------------
        qdot_lim = np.array([2.62, 2.62, 2.62, 2.62, 3.14, 3.14, 3.14])
        self.psf_filter = PSFFilter(
            n_joints=n_arm_joints,
            alpha=alpha,
            qdot_min=-qdot_lim,
            qdot_max=qdot_lim,
        )

        # --- OSC velocity controller (replaces pseudoinverse DiffIK) ----------
        # PoseTaskVelocityController uses the dynamically-consistent J_bar
        # = M_inv J^T (J M_inv J^T)^{-1} instead of pinv(J), and adds
        # nullspace posture control N = I - J_bar J.
        try:
            from oscbf.core.controllers import PoseTaskVelocityController as _PTVC
            import jax.numpy as _jnp
            kp_pos, kp_rot, kp_joint = 50.0, 20.0, 10.0
            self._osc_controller = _PTVC(
                n_joints=n_arm_joints,
                kp_task=np.array([kp_pos]*3 + [kp_rot]*3),
                kp_joint=np.full(n_arm_joints, kp_joint),
                qdot_min=-qdot_lim,
                qdot_max=qdot_lim,
            )
            self._jnp = _jnp
            logging.info("[3dpsf] OSC velocity controller loaded (dynamically-consistent J_bar)")
        except ImportError:
            self._osc_controller = None
            logging.warning("[3dpsf] oscbf not available; falling back to pseudoinverse DiffIK")

        # State
        self._safe_result: Optional[SafeRegionResult] = None
        self._psf_result: Optional[PoissonResult] = None
        self._poisson_solver: Optional[PoissonSolver] = None
        self._warm_start: Optional[np.ndarray] = None

        # Fixed sample geometry computed once per episode in rebuild().
        # Local-frame coords are constant; world-frame positions are recovered
        # at each control step by applying current FK, eliminating per-step
        # Poisson-disk resampling and its variable-N JAX recompilation risk.
        self._sample_local: Optional[np.ndarray] = None   # (N, 3) local coords
        self._sample_geom_ids: Optional[np.ndarray] = None  # (N,) geom IDs
        self._sample_body_ids: Optional[np.ndarray] = None  # (N,) body IDs

        # Per-episode h timeseries: accumulated in filter_action()
        self._h_history: list = []
        self._body_ids_history: list = []

        # Per-episode action comparison: nominal vs safe EEF deltas
        self._action_nom_history: list = []    # list of (6,) arrays
        self._action_safe_history: list = []   # list of (6,) arrays
        self._converged_history: list = []     # list of bool
        self._filter_time_history: list = []   # list of float [ms]
        self._n_infeasible_history: list = []  # list of int

    # ------------------------------------------------------------------ rebuild

    def rebuild(
        self,
        env,       # safelibero OffScreenRenderEnv (mujoco-py)
        q_arm: np.ndarray,
        seed: int = 0,
        out_dir: Optional[pathlib.Path] = None,
    ) -> None:
        """Rebuild safe region and solve Poisson (offline, once per episode).

        Args:
            env:   safelibero env (mujoco-py). Obstacles are extracted from
                   env.sim.model / env.sim.data.
            q_arm: (7,) current arm joint positions (from obs['robot0_joint_pos']).
            seed:  RNG seed for Poisson disk sampling.
            out_dir: if given, save PSF visualisations there.
        """
        # Reset per-episode history
        self._h_history = []
        self._body_ids_history = []
        self._action_nom_history = []
        self._action_safe_history = []
        self._converged_history = []
        self._filter_time_history = []
        self._n_infeasible_history = []

        t0 = time.perf_counter()

        # Step 2a: extract obstacle geoms from safelibero env
        obstacles = get_obstacles_from_robosuite_sim(
            sim_model=env.sim.model,
            sim_data=env.sim.data,
            obstacle_body_keyword="obstacle",
        )
        print(f"[3dpsf] {len(obstacles)} obstacle geoms extracted from env")

        # Step 2b: build buffered safe region C_ε
        self._safe_result = self.safe_builder.build(
            qpos=q_arm, obstacles=obstacles, seed=seed
        )
        n_safe = int(self._safe_result.buffered_safe.sum())
        n_total = self._safe_result.buffered_safe.size
        print(f"[3dpsf] C_ε: {n_safe}/{n_total} voxels safe "
              f"({100*n_safe/n_total:.1f}%),  "
              f"{len(self._safe_result.robot_sample_pts)} sample pts y_i")

        # Step 3: solve Poisson equation on C_ε
        self._poisson_solver = PoissonSolver(
            safe_mask=self._safe_result.buffered_safe,
            bounds=WORKSPACE_BOUNDS,
            voxel_size=self.voxel_size,
            max_iter=POISSON_ITERS,
            use_jax=self.use_jax,
        )
        self._psf_result = self._poisson_solver.solve(
            f_value=POISSON_F,
            warm_start=self._warm_start,    # None on first call → cold start
        )
        self._warm_start = self._psf_result.h.copy()  # save for next rebuild

        t1 = time.perf_counter()
        print(f"[3dpsf] Poisson solve: iters={self._psf_result.n_iter}, "
              f"residual={self._psf_result.residual:.2e}, "
              f"h_max={self._psf_result.h.max():.4f}, "
              f"elapsed={1000*(t1-t0):.0f} ms")

        # Compute per-sample local-frame coords for fixed-N filter_action.
        # Done once per episode — FK must match q_arm used for sampling.
        sample_pts_w = self._safe_result.robot_sample_pts  # (N, 3) world frame
        qpos_vis = np.zeros(self.n_mj_joints)
        qpos_vis[:self.n_arm] = q_arm
        self._mj_data.qpos[:] = qpos_vis
        mj.mj_fwdPosition(self._mj_model, self._mj_data)

        geom_ids = np.asarray(self.robot_surface.collision_geom_ids)  # (G,)
        geom_pos = self._mj_data.geom_xpos[geom_ids]                  # (G, 3)
        sq_dists = np.sum(
            (sample_pts_w[:, None, :] - geom_pos[None, :, :]) ** 2, axis=-1
        )  # (N, G)
        nearest_idx = np.argmin(sq_dists, axis=1)   # (N,) → index into geom_ids
        sample_geom_ids = geom_ids[nearest_idx]      # (N,) actual geom IDs

        local_coords = np.empty_like(sample_pts_w)
        for i, gid in enumerate(sample_geom_ids):
            xpos = self._mj_data.geom_xpos[gid]
            xmat = self._mj_data.geom_xmat[gid].reshape(3, 3)
            local_coords[i] = xmat.T @ (sample_pts_w[i] - xpos)

        self._sample_local    = local_coords
        self._sample_geom_ids = sample_geom_ids
        self._sample_body_ids = np.array(
            [self._mj_model.geom_bodyid[gid] for gid in sample_geom_ids]
        )
        print(f"[3dpsf] Local-frame tracking: {len(sample_pts_w)} sample points stored")

        # Optional visualisations of the PSF field (no timeseries yet — call
        # plot_h_timeseries() / plot_action_comparison() after the episode loop)
        if out_dir is not None:
            out_dir.mkdir(parents=True, exist_ok=True)
            self._poisson_solver.visualize_slices(
                self._psf_result,
                out_dir / "psf_slices.png",
                robot_pts=self._safe_result.robot_dense_pts,
            )
            self._poisson_solver.visualize_volume(
                self._psf_result,
                out_dir / "psf_volume.png",
                n_slices=12,
            )
            self._safe_result.occupancy_grid.visualize_3d(
                out_path=out_dir / "occupancy_3d.png",
                buffered_safe=self._safe_result.buffered_safe,
                robot_pts=self._safe_result.robot_sample_pts,
                off_screen=True,
            )
            self._poisson_solver.visualize_voxels_3d(
                self._psf_result,
                out_dir / "psf_voxels_3d.png",
                threshold=0.0,
                robot_pts=self._safe_result.robot_sample_pts,
                off_screen=True,
                occupancy_grid=self._safe_result.occupancy_grid,
            )
            self.robot_surface.visualize_samples_3d(
                qpos=q_arm,
                dense_pts=self._safe_result.robot_dense_pts,
                sample_pts=self._safe_result.robot_sample_pts,
                out_path=out_dir / "robot_samples_3d.png",
                off_screen=True,
            )

    # ------------------------------------------------------------------ filter

    def filter_action(
        self,
        action_nom: np.ndarray,     # (7,) nominal EEF delta [dx,dy,dz,dax,day,daz,grip]
        obs: dict,
    ) -> np.ndarray:
        """Apply 3DPSF safety filter and return the safe EEF action.

        Args:
            action_nom: (7,) nominal EEF action from VLA policy.
            obs:        Current environment observation dict.

        Returns:
            (7,) safe action (same gripper command, filtered EEF delta).
        """
        if self._psf_result is None:
            raise RuntimeError("Call rebuild() before filter_action().")

        q_arm = np.asarray(obs["robot0_joint_pos"], dtype=np.float64)  # (7,)

        # --- Sync new mujoco model with current arm joints -------------------
        self._mj_data.qpos[:self.n_arm] = q_arm
        mj.mj_fwdPosition(self._mj_model, self._mj_data)

        # --- Step 4a: recover world-frame sample positions via current FK ----
        # Local-frame coords are constant per episode (set in rebuild()).
        # Applying current geom transforms is O(N) matrix-vector mults — no
        # dense cloud generation or Poisson disk resampling on the hot path.
        if self._sample_local is None or self._sample_geom_ids is None:
            self._h_history.append(np.zeros(0, dtype=np.float32))
            self._body_ids_history.append(np.zeros(0, dtype=int))
            return action_nom.copy()

        xpos_all = self._mj_data.geom_xpos[self._sample_geom_ids]           # (N, 3)
        xmat_all = self._mj_data.geom_xmat[self._sample_geom_ids].reshape(-1, 3, 3)  # (N,3,3)
        sample_pts = np.einsum("nij,nj->ni", xmat_all, self._sample_local) + xpos_all

        # --- Step 4b: body IDs for each sample point -------------------------
        body_ids = self._sample_body_ids

        # --- Step 4c: linear Jacobians J_i(q) for each sample point ----------
        jacobians = PSFFilter.compute_jacobians(
            self._mj_model, self._mj_data, sample_pts, body_ids
        )  # (N, 3, nv); slice to arm joints
        jacobians = jacobians[:, :, :self.n_arm]  # (N, 3, n_arm)

        # --- Step 4d: nominal EEF action → nominal joint velocity ------------
        J_eef = self._compute_eef_jacobian()   # (6, n_arm)
        v_eef_nom = action_nom[:6].astype(np.float64)
        if self._osc_controller is not None:
            v_joints_nom = self._nom_via_osc(q_arm, J_eef, v_eef_nom)
        else:
            v_joints_nom = np.linalg.lstsq(J_eef, v_eef_nom, rcond=None)[0]  # fallback

        # --- Step 4e: CBF QP filter in joint space ---------------------------
        result: PSFFilterResult = self.psf_filter.filter(
            v_nom=v_joints_nom.astype(np.float32),
            psf_result=self._psf_result,
            sample_pts=sample_pts,
            jacobians=jacobians,
            q=q_arm.astype(np.float32),
        )

        # Accumulate h values for per-link timeseries plot
        self._h_history.append(result.h_vals.copy())
        self._body_ids_history.append(body_ids.copy())
        self._converged_history.append(result.converged)
        self._n_infeasible_history.append(result.n_infeasible)
        self._filter_time_history.append(result.qp_time_ms)

        if not result.converged:
            logging.warning("[3dpsf] QP did not converge; falling back to nominal action")
            self._action_nom_history.append(v_eef_nom.copy())
            self._action_safe_history.append(v_eef_nom.copy())  # fallback = nominal
            return action_nom.copy()

        # --- Step 4f: safe joint velocity → safe EEF action ------------------
        v_joints_safe = result.v_safe.astype(np.float64)  # (n_arm,)
        v_eef_safe = J_eef @ v_joints_safe                # (6,)

        self._action_nom_history.append(v_eef_nom.copy())
        self._action_safe_history.append(v_eef_safe.copy())

        action_safe = action_nom.copy()
        action_safe[:6] = v_eef_safe
        return action_safe

    def _compute_eef_jacobian(self) -> np.ndarray:
        """Compute 6×n_arm EEF Jacobian using new mujoco (FK must be current)."""
        nv = self._mj_model.nv
        jacp = np.zeros((3, nv))
        jacr = np.zeros((3, nv))
        eef_pos = self._mj_data.body(self._eef_body_id).xpos.copy()
        mj.mj_jac(self._mj_model, self._mj_data, jacp, jacr, eef_pos, self._eef_body_id)
        J6 = np.vstack([jacp, jacr])[:, :self.n_arm]   # (6, n_arm)
        return J6

    def _get_mass_matrix_inv(self) -> np.ndarray:
        """Return (n_arm × n_arm) inverse mass matrix via MuJoCo crb (FK must be current)."""
        nv = self._mj_model.nv
        M_flat = np.zeros(nv * nv)
        mj.mj_fullM(self._mj_model, M_flat, self._mj_data.qM)
        M = M_flat.reshape(nv, nv)[:self.n_arm, :self.n_arm]
        return np.linalg.inv(M)

    def _nom_via_osc(
        self,
        q_arm: np.ndarray,
        J_eef: np.ndarray,
        v_eef_nom: np.ndarray,
    ) -> np.ndarray:
        """Compute nominal joint velocity via OSC with dynamically-consistent J_bar.

        The VLA action encodes desired EEF displacements [delta_pos, delta_axisangle].
        We treat them as feedforward velocity targets (des_vel, des_omega) with no
        position error term (des_pos == cur_pos, des_rot == cur_rot), so the OSC
        output equals J_bar @ v_eef_nom + N @ nullspace_posture_vel.

        Returns:
            (n_arm,) nominal joint velocity.
        """
        import jax.numpy as jnp

        M_inv = self._get_mass_matrix_inv()  # (n_arm, n_arm)

        # Current EEF pose (no position error → controller acts as J_bar mapping)
        eef_body = self._mj_data.body(self._eef_body_id)
        cur_pos = eef_body.xpos.copy()
        cur_rot = eef_body.xmat.copy().reshape(3, 3)

        # Desired pose = current pose (error = 0); feedforward = VLA delta
        des_pos = cur_pos
        des_rot = cur_rot
        des_vel = v_eef_nom[:3]
        des_omega = v_eef_nom[3:6]

        # Panda home posture for nullspace target
        q_rest = np.array([0.0, -np.pi/6, 0.0, -2*np.pi/3, 0.0, np.pi/2, np.pi/4])

        v_nom = self._osc_controller(
            jnp.asarray(q_arm, dtype=jnp.float32),
            jnp.asarray(cur_pos, dtype=jnp.float32),
            jnp.asarray(cur_rot, dtype=jnp.float32),
            jnp.asarray(des_pos, dtype=jnp.float32),
            jnp.asarray(des_rot, dtype=jnp.float32),
            jnp.asarray(des_vel, dtype=jnp.float32),
            jnp.asarray(des_omega, dtype=jnp.float32),
            jnp.asarray(q_rest, dtype=jnp.float32),
            jnp.asarray(J_eef, dtype=jnp.float32),
            jnp.asarray(M_inv, dtype=jnp.float32),
        )
        return np.asarray(v_nom, dtype=np.float64)

    def plot_h_timeseries(
        self,
        out_path: pathlib.Path,
        dt: float = 1.0,
    ) -> None:
        """Save the per-link PSF timeseries plot for the current episode.

        Call after the episode loop has finished so the full history is available.

        Args:
            out_path: Output file path (e.g. ep_dir / "psf_sample_h.png").
            dt:       Seconds per control step — scales the x-axis label to [s].
        """
        if not self._h_history:
            return
        if self._poisson_solver is None:
            return
        self._poisson_solver.visualize_sample_h(
            self._h_history,
            self._body_ids_history,
            self._mj_model,
            out_path,
            dt=dt,
        )

    def plot_action_comparison(
        self,
        out_path: pathlib.Path,
        dt: float = 1.0,
    ) -> None:
        """Save a 6-panel plot comparing nominal vs safe EEF actions over time.

        Each panel shows one EEF degree of freedom (dx, dy, dz, dax, day, daz).
        Timesteps where the QP did not converge are marked with a red background
        stripe.  A summary panel shows QP convergence rate, mean filter time,
        and infeasible-h statistics.

        Args:
            out_path: Output file path (e.g. ep_dir / "action_comparison.png").
            dt:       Seconds per control step — scales the x-axis to [s].
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            logging.warning("matplotlib not available; skipping plot_action_comparison.")
            return

        T = len(self._action_nom_history)
        if T == 0:
            return

        nom = np.array(self._action_nom_history)    # (T, 6)
        safe = np.array(self._action_safe_history)  # (T, 6)
        converged = np.array(self._converged_history, dtype=bool)   # (T,)
        times = np.arange(T, dtype=np.float64) * dt
        x_label = "Time [s]" if dt != 1.0 else "Timestep"

        dof_labels = ["dx", "dy", "dz", "dax", "day", "daz"]
        fig, axes = plt.subplots(3, 2, figsize=(14, 9), sharex=True)
        axes_flat = axes.ravel()

        for k, ax in enumerate(axes_flat):
            ax.plot(times, nom[:, k],  color="#2196F3", linewidth=1.5,
                    label="nominal", alpha=0.85)
            ax.plot(times, safe[:, k], color="#FF5722", linewidth=1.5,
                    label="safe (filtered)", alpha=0.85)

            # Mark non-converged steps
            fail_mask = ~converged
            if fail_mask.any():
                for t_fail in times[fail_mask]:
                    ax.axvspan(t_fail - 0.5 * dt, t_fail + 0.5 * dt,
                               color="red", alpha=0.15, zorder=0)

            diff = safe[:, k] - nom[:, k]
            ax.fill_between(times, 0, diff, alpha=0.12, color="#9C27B0",
                            label="filter Δ", zorder=1)
            ax.axhline(0, color="gray", linewidth=0.6, linestyle=":")
            ax.set_ylabel(dof_labels[k], fontsize=10)
            ax.grid(alpha=0.3, linestyle=":")
            if k == 0:
                ax.legend(fontsize=8, loc="upper right")

        axes_flat[-2].set_xlabel(x_label, fontsize=10)
        axes_flat[-1].set_xlabel(x_label, fontsize=10)

        n_ok = int(converged.sum())
        n_total = len(converged)
        filter_times = np.array(self._filter_time_history)
        n_infeas = np.array(self._n_infeasible_history)
        eef_delta_norm = np.linalg.norm(safe - nom, axis=1)   # (T,)

        summary = (
            f"QP convergence: {n_ok}/{n_total} ({100*n_ok/max(n_total,1):.1f}%)\n"
            f"Mean QP time: {filter_times.mean():.2f} ms  "
            f"(max {filter_times.max():.2f} ms)\n"
            f"Mean |safe−nom| EEF: {eef_delta_norm.mean():.4f}  "
            f"max {eef_delta_norm.max():.4f}\n"
            f"Steps with h≤0: {int((np.array(self._h_history, dtype=object) != np.array([])).sum())} "
            f"infeasible samples mean: {n_infeas.mean():.1f}"
        )
        fig.text(0.5, 0.01, summary, ha="center", va="bottom", fontsize=9,
                 bbox=dict(boxstyle="round,pad=0.4", fc="#F5F5F5", ec="gray"))

        fig.suptitle(
            "3DPSF Safety Filter: Nominal vs Safe EEF Actions",
            fontsize=12, fontweight="bold",
        )
        fig.tight_layout(rect=[0, 0.06, 1, 1])
        fig.savefig(str(out_path), dpi=120)
        plt.close(fig)
        print(f"[3dpsf] Saved action comparison → {out_path}")

    def plot_psf_3d(
        self,
        out_path: pathlib.Path,
        threshold: float = 0.0,
    ) -> None:
        """Save a 3D PyVista voxel rendering of the current PSF field.

        Wraps PoissonSolver.visualize_voxels_3d().

        Args:
            out_path:  Output PNG path.
            threshold: Only show voxels with h > threshold.
        """
        if self._psf_result is None or self._poisson_solver is None:
            return
        robot_pts = (
            self._safe_result.robot_dense_pts
            if self._safe_result is not None else None
        )
        occ = self._safe_result.occupancy_grid if self._safe_result is not None else None
        self._poisson_solver.visualize_voxels_3d(
            self._psf_result,
            out_path,
            threshold=threshold,
            robot_pts=robot_pts,
            off_screen=True,
            occupancy_grid=occ,
        )

    @property
    def is_ready(self) -> bool:
        return self._psf_result is not None


# --------------------------------------------------------------------------
# Main evaluation loop
# --------------------------------------------------------------------------

def eval_libero_3dpsf(args: Args) -> None:
    np.random.seed(args.seed)

    # --- Task suite ----------------------------------------------------------
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[args.task_suite_name](safety_level=args.safety_level)
    max_steps = {
        "safelibero_spatial": 300,
        "safelibero_object": 300,
        "safelibero_goal": 300,
        "safelibero_long": 550,
    }.get(args.task_suite_name, 300)

    # --- VLA policy client ---------------------------------------------------
    client = _websocket_client_policy.WebsocketClientPolicy(args.host, args.port)

    # --- 3DPSF safety layer --------------------------------------------------
    layer = PSF3DLayer(
        robot_xml=ROBOT_XML,
        n_arm_joints=7,
        alpha=CBF_ALPHA,
        voxel_size=VOXEL_SIZE,
        epsilon=EPSILON,
        use_jax=args.use_jax,
        gripper_xml=GRIPPER_XML,
    )
    print(f"[3dpsf] Robot surface: {len(layer.robot_surface.collision_geom_ids)} collision geoms")
    print(f"[3dpsf] New mujoco model: nq={layer._mj_model.nq}, nv={layer._mj_model.nv}")

    # --- Evaluation ----------------------------------------------------------
    pathlib.Path(args.video_out_path).mkdir(parents=True, exist_ok=True)
    total_episodes, total_successes, total_collisions = 0, 0, 0

    for task_id in args.task_index:
        task = task_suite.get_task(task_id)
        initial_states = task_suite.get_task_init_states(task_id)
        env, task_description = _get_libero_env(task, args.safety_level, LIBERO_ENV_RESOLUTION, args.seed)

        task_seg = task_description.replace(" ", "_")
        out_dir = pathlib.Path(args.video_out_path) / task_seg / "3dpsf"
        out_dir.mkdir(parents=True, exist_ok=True)

        task_ep, task_ok = 0, 0
        obstacle_names = [
            n.replace("_joint0", "")
            for n in env.sim.model.joint_names
            if "obstacle" in n
        ]

        for episode_idx in args.episode_index:
            # ── Episode reset ─────────────────────────────────────────────────
            env.reset()
            action_plan: collections.deque = collections.deque()
            obs = env.set_init_state(initial_states[episode_idx])

            # Wait for objects to settle
            for _ in range(args.num_steps_wait):
                obs, *_ = env.step(LIBERO_DUMMY_ACTION)

            # ── Identify active obstacle ──────────────────────────────────────
            obstacle_name = ""
            for oname in obstacle_names:
                pos = obs.get(f"{oname}_pos", np.zeros(3))
                if pos[2] > 0 and -0.5 < pos[0] < 0.5 and -0.5 < pos[1] < 0.5:
                    obstacle_name = oname
                    print(f"[3dpsf] Active obstacle: {obstacle_name} @ {np.round(pos, 3)}")
                    break
            initial_obs_pos = obs.get(obstacle_name + "_pos", np.zeros(3)).copy()

            # ── OFFLINE: rebuild PSF for this episode ─────────────────────────
            ep_dir = out_dir / str(episode_idx)
            q_arm = obs["robot0_joint_pos"].copy()
            layer.rebuild(
                env=env,
                q_arm=q_arm,
                seed=args.seed + episode_idx,
                out_dir=ep_dir if args.save_psf_vis else None,
            )

            # ── Episode loop ──────────────────────────────────────────────────
            t, done, collide_flag = 0, False, False
            replay_images: list = []

            while t < max_steps:
                try:
                    img = np.ascontiguousarray(obs["agentview_image"][::-1, ::-1])
                    wrist_img = np.ascontiguousarray(obs["robot0_eye_in_hand_image"][::-1, ::-1])
                    replay_images.append(img)

                    if not action_plan:
                        element = {
                            "observation/image": image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(img, args.resize_size, args.resize_size)
                            ),
                            "observation/wrist_image": image_tools.convert_to_uint8(
                                image_tools.resize_with_pad(wrist_img, args.resize_size, args.resize_size)
                            ),
                            "observation/state": np.concatenate([
                                obs["robot0_eef_pos"],
                                _quat2axisangle(obs["robot0_eef_quat"]),
                                obs["robot0_gripper_qpos"],
                            ]),
                            "prompt": str(task_description),
                        }
                        action_chunk = client.infer(element)["actions"]
                        action_plan.extend(action_chunk[: args.replan_steps])

                    action_nom = action_plan.popleft()

                    # ONLINE: apply 3DPSF filter
                    t_filter_start = time.perf_counter()
                    action_safe = layer.filter_action(action_nom, obs)
                    t_filter = (time.perf_counter() - t_filter_start) * 1000

                    obs, reward, done, info = env.step(action_safe.tolist())

                    # Collision detection (obstacle moved)
                    if not collide_flag and obstacle_name:
                        new_pos = obs.get(obstacle_name + "_pos", initial_obs_pos)
                        if np.sum(np.abs(new_pos - initial_obs_pos)) > 0.001:
                            print(f"[3dpsf] COLLISION at t={t}  (filter took {t_filter:.1f} ms)")
                            collide_flag = True

                    if done:
                        task_ok += 1; total_successes += 1
                        break
                    t += 1

                except Exception as e:
                    logging.error(f"Exception at t={t}: {e}")
                    break

            # ── Per-episode plots (generated after loop so full history is available)
            if args.save_psf_vis:
                layer.plot_h_timeseries(ep_dir / "psf_sample_h.png", dt=0.1)
                layer.plot_action_comparison(ep_dir / "action_comparison.png", dt=0.1)

            # ── Episode bookkeeping ───────────────────────────────────────────
            task_ep += 1; total_episodes += 1
            if collide_flag:
                total_collisions += 1

            suffix = "success" if done else "failure"
            safe_tag = "safe" if not collide_flag else "unsafe"
            video_path = ep_dir / f"{episode_idx}_{suffix}_{safe_tag}.mp4"
            imageio.mimwrite(video_path, [np.asarray(x) for x in replay_images], fps=30)

            ss = done and not collide_flag
            print(f"[3dpsf] ep={episode_idx} | success={done} | collision={collide_flag} | SS={ss}")
            print(f"[3dpsf] total: ep={total_episodes} ok={total_successes} coll={total_collisions}")

        env.close()

    print(f"\n[3dpsf] === FINAL ===")
    print(f"  Success rate : {total_successes}/{total_episodes} = {100*total_successes/max(1,total_episodes):.1f}%")
    print(f"  Collision rate: {total_collisions}/{total_episodes} = {100*total_collisions/max(1,total_episodes):.1f}%")
    safe_ok = total_successes - total_collisions
    print(f"  Safe success : {safe_ok}/{total_episodes} = {100*safe_ok/max(1,total_episodes):.1f}%")


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------

def _get_libero_env(task, level, resolution, seed):
    task_bddl = (
        pathlib.Path(get_libero_path("bddl_files"))
        / task.problem_folder
        / task.bddl_file
    )
    env_args = {
        "bddl_file_name": task_bddl,
        "camera_heights": resolution,
        "camera_widths": resolution,
        "camera_depths": True,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(seed)
    return env, task.language


def _quat2axisangle(quat: np.ndarray) -> np.ndarray:
    quat = quat.copy()
    quat[3] = np.clip(quat[3], -1.0, 1.0)
    den = np.sqrt(1.0 - quat[3] ** 2)
    if math.isclose(den, 0.0):
        return np.zeros(3)
    return (quat[:3] * 2.0 * math.acos(quat[3])) / den


# --------------------------------------------------------------------------
if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    args = tyro.cli(Args)
    eval_libero_3dpsf(args)
