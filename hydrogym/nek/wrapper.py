import os
from typing import Dict, List, Optional, Tuple

import gym
import numpy as np
from mpi4py import MPI
from omegaconf import OmegaConf

from .configs import Config
from .nek_marl import parallel_env


def load_nek_config(
    config_path: str,
    overrides: Optional[List[str]] = None,
) -> "OmegaConf":
  """Load Nek MARL config from YAML and apply overrides."""
  if overrides is None:
    overrides = []

  conf = OmegaConf.merge(
      OmegaConf.structured(Config()),
      OmegaConf.load(config_path),
      OmegaConf.from_dotlist(overrides),
  )
  return conf


def mpi_split(comm_world: MPI.Comm) -> MPI.Comm:
  """Split MPI world into master/worker inter-communicator."""
  mpi_rank = comm_world.Get_rank()
  mpi_size = comm_world.Get_size()
  if mpi_size < 2:
    raise RuntimeError(
        "MPI world size must be >= 2 to create the Nek inter-communicator. "
        "Launch with MPMD, e.g. `mpirun -n 1 python ... : -n N ./nek5000`, "
        "so rank 0 can connect to the Nek worker ranks."
    )
  if mpi_rank == 0:
    color = 0
  else:
    color = 1
  local_comm = comm_world.Split(color, mpi_rank)
  sub_comm = local_comm.Create_intercomm(
      local_leader=0, peer_comm=MPI.COMM_WORLD, remote_leader=1, tag=99)
  return sub_comm


class NekMARLGymWrapper(gym.Env):
  """
  Gym wrapper that concatenates PettingZoo agent observations/actions.
  The underlying Nek MARL environment is unchanged and still uses MPI.
  """

  metadata = {"render.modes": ["human"]}

  def __init__(
      self,
      conf,
      run_root: str = "runs",
      run_name: Optional[str] = None,
      reward_agg: str = "mean",
  ):
    self.conf = conf

    # Create run folder for the Nek env.
    if run_name is None:
      run_name = str(int(MPI.COMM_WORLD.Get_rank()))
    self.run_folder = os.path.join(run_root, run_name)
    if not os.path.exists(self.run_folder):
      os.makedirs(self.run_folder, exist_ok=True)

    # MPI communicator required by Nek env.
    comm_world = MPI.COMM_WORLD
    self.sub_comm = mpi_split(comm_world)

    # Initialize PettingZoo ParallelEnv.
    self.pz_env = parallel_env(
        conf=self.conf,
        rank_folder=self.run_folder,
        sub_comm=self.sub_comm,
    )

    self.agent_order = list(self.pz_env.possible_agents)
    self.n_agents = len(self.agent_order)

    # Infer per-agent spaces.
    sample_agent = self.agent_order[0]
    self._per_agent_obs_space = self.pz_env.observation_space(sample_agent)
    self._per_agent_act_space = self.pz_env.action_space(sample_agent)

    self.per_agent_obs_size = int(np.prod(self._per_agent_obs_space.shape))
    self.per_agent_act_size = int(np.prod(self._per_agent_act_space.shape))

    # Concatenated Gym spaces.
    self.observation_space = gym.spaces.Box(
        low=-np.inf,
        high=np.inf,
        shape=(self.n_agents * self.per_agent_obs_size,),
        dtype=np.float32,
    )

    per_act_low = np.asarray(self._per_agent_act_space.low).reshape(-1)
    per_act_high = np.asarray(self._per_agent_act_space.high).reshape(-1)
    act_low = np.tile(per_act_low, self.n_agents)
    act_high = np.tile(per_act_high, self.n_agents)
    self.action_space = gym.spaces.Box(
        low=act_low,
        high=act_high,
        dtype=np.float32,
    )

    self.reward_agg = reward_agg

  def _concat_obs(self, obs_dict: Dict[str, np.ndarray]) -> np.ndarray:
    obs_list = []
    for agent in self.agent_order:
      obs = np.asarray(obs_dict[agent]).reshape(-1)
      obs_list.append(obs)
    return np.concatenate(obs_list, axis=0).astype(np.float32)

  def _split_actions(self, action: np.ndarray) -> Dict[str, np.ndarray]:
    action = np.asarray(action).reshape(-1)
    expected = self.n_agents * self.per_agent_act_size
    if action.size != expected:
      raise ValueError(
          f"Action size {action.size} does not match expected {expected}")
    actions = {}
    for i, agent in enumerate(self.agent_order):
      start = i * self.per_agent_act_size
      end = (i + 1) * self.per_agent_act_size
      per_act = action[start:end].reshape(self._per_agent_act_space.shape)
      actions[agent] = per_act
    return actions

  def _aggregate_reward(self, rewards: Dict[str, float]) -> float:
    values = [float(rewards[a]) for a in self.agent_order]
    if self.reward_agg == "sum":
      return float(np.sum(values))
    return float(np.mean(values))

  def reset(self) -> np.ndarray:
    result = self.pz_env.reset()
    if isinstance(result, tuple):
      obs_dict = result[0]
    else:
      obs_dict = result
    return self._concat_obs(obs_dict)

  def step(
      self, action: np.ndarray
  ) -> Tuple[np.ndarray, float, bool, dict]:
    action_dict = self._split_actions(action)
    obs_dict, rewards, dones, infos = self.pz_env.step(action_dict)

    obs = self._concat_obs(obs_dict)
    reward = self._aggregate_reward(rewards)
    done = all(bool(dones[a]) for a in self.agent_order)

    info = {
        "reward_per_agent": rewards,
        "done_per_agent": dones,
        "info_per_agent": infos,
    }
    return obs, reward, done, info

  def render(self, mode="human"):
    if hasattr(self.pz_env, "render"):
      return self.pz_env.render(mode=mode)
    return None

  def close(self):
    if hasattr(self.pz_env, "close"):
      self.pz_env.close()
