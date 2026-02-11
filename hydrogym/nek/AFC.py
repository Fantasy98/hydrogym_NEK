"""
The classical approaches for wing control
Matching the structure from drl_repo_TEST/src/lib/AFC.py
"""

import numpy as np


class AFC:
  """A general object for classical AFC approaches"""
  def __init__(self, agent_list):
    self.agent_list = agent_list

  def policy(self, observation):
    """The control policy depends on the observation"""
    action = None
    if action == None:
      raise NotImplementedError('[ERROR] Please Add the policy function!')
    return action

  def predict(self, observations, state, episode_start, deterministic):
    """Return the action to ENV based on the policy"""
    if isinstance(observations, dict):
      # Multi-agent case: observations is a dict
      actions = {agent: self.policy(observations[agent]) for agent in observations.keys()}
    else:
      # Single agent case: observations is a numpy array
      actions = self.policy(observations)
    return actions, None


class OppoCtrl(AFC):
  """Opposition control implementation"""
  def __init__(self, agent_list, ctrl_max_amp):
    super().__init__(agent_list)
    self.alpha = ctrl_max_amp

  def policy(self, observation):
    """v = -alpha * (v - <v>)"""
    # We define the V-Vel <==> 1
    action = np.array([-1.0 * self.alpha * observation[1, 0, 0]], dtype=np.float32)
    return action


class BLCtrl(AFC):
  """Steady uniform blowing/suction"""
  def __init__(self, agent_list, ctrl_max_amp):
    super().__init__(agent_list)
    self.alpha = ctrl_max_amp

  def policy(self, observation):
    """v = Psi"""
    action = np.array([1 * self.alpha], dtype=np.float32)
    return action


class SinWave:
  """Imposing Sinusoidal wave to check the mean"""
  def __init__(self, agent_list, ctrl_max_amp, Kx, Kz, Lx, Lz):
    self.alpha = ctrl_max_amp
    self.agent_list = agent_list
    self.Kx, self.Kz = Kx, Kz
    self.Lx, self.Lz = Lx, Lz

  def load_node_info(self, Node_Info):
    """Get the node info to impose the Sinusoidal Wave"""
    self.Node_Info = Node_Info
    print(f"Node Info {self.Node_Info}", flush=True)
    return

  def policy(self, observation):
    """We define the V-Vel <==> 1"""
    x, z = observation
    kx = self.Kx * 2 * np.pi * x / self.Lx
    kz = self.Kz * 2 * np.pi * z / self.Lz
    action = np.array([self.alpha * np.sin(kx)], dtype=np.float32)
    return action

  def predict(self, observations, state, episode_start, deterministic):
    obs = {}
    for il, agent_name_ in enumerate(self.agent_list):
      agent_name = self.nameAgent(
        nid=self.Node_Info['NID'][il],
        gllid=self.Node_Info['GLLID'][il],
        iface=self.Node_Info['FACEID'][il],
        ix=self.Node_Info['ix'][il],
        iy=self.Node_Info['iy'][il],
        iz=self.Node_Info['iz'][il],
      )
      x, z = self.Node_Info['x'][il], self.Node_Info['z'][il]
      obs[agent_name] = (x, z)
    actions = {agent: self.policy(obs[agent]) for agent in obs.keys()}
    return actions, None

  @staticmethod
  def nameAgent(nid, gllid, iface, ix, iy, iz):
    """Name the agent based on the GRID information"""
    agent_name = (f"jet_np{nid:05d}_"
                  f"gid{gllid:05d}_"
                  f"iface{iface}_"
                  f"ix{ix:05d}_"
                  f"iy{iy:05d}_"
                  f"iz{iz:05d}")
    return agent_name


class ZeroCtrl(AFC):
  """Zero action controller (no control)"""
  def policy(self, observation):
    return np.array([0.0], dtype=np.float32)


def make_afc_controller(env, ctrl_type="AFC"):
  """
  Factory function to create an AFC controller compatible with integrate().
  
  The controller adapts between NekMARLGymWrapper (concatenated obs/actions) 
  and parallel_env (dict obs/actions) formats.
  
  To use an SB3 model, you can pass it directly to integrate():
    from stable_baselines3 import PPO
    loaded_model = PPO.load("path/to/model")
    hgym.integrate(env, ..., controller=loaded_model)
  
  Args:
    env: Environment instance (NekMARLGymWrapper or parallel_env)
    ctrl_type: Algorithm name ("AFC", "OC", "BL", "SIN", "ZERO", or SB3 algorithm name)
  
  Returns:
    controller: Controller object with .predict() method, or None if not an AFC algorithm
  """
  # Get agent list and config
  if hasattr(env, 'pz_env'):
    # NekMARLGymWrapper: get underlying parallel_env
    pz_env = env.pz_env
    agent_list = list(pz_env.possible_agents)
    conf = env.conf
    is_wrapped = True
  elif hasattr(env, 'possible_agents'):
    # Direct parallel_env
    pz_env = env
    agent_list = list(env.possible_agents)
    conf = env.conf
    is_wrapped = False
  else:
    raise ValueError("Environment must be NekMARLGymWrapper or parallel_env")

  # Create AFC controller based on algorithm
  if ctrl_type == "OC":
    afc_controller = OppoCtrl(agent_list, conf.runner.ctrl_max_amp)
  elif ctrl_type == "BL":
    afc_controller = BLCtrl(agent_list, conf.runner.ctrl_max_amp)
  elif ctrl_type == "SIN":
    afc_controller = SinWave(
      agent_list, conf.runner.ctrl_max_amp,
      Kx=1, Kz=1, Lx=conf.simulation.Lx, Lz=conf.simulation.Lz
    )
    if hasattr(pz_env, 'agent_info'):
      afc_controller.load_node_info(pz_env.agent_info)
  elif ctrl_type == "ZERO" or ctrl_type is None:
    # Zero action (no control) - create a simple controller
    afc_controller = ZeroCtrl(agent_list)
  else:
    # Not an AFC controller - return None to use legacy function or SB3 model
    print(f"[WARNING] Controller type {ctrl_type} not supported, please use SB3")
    return None

  # Create adapter wrapper if needed for NekMARLGymWrapper
  if is_wrapped:
    class ControllerAdapter:
      """Adapter to convert between concatenated and dict formats"""
      def __init__(self, afc_ctrl, wrapper_env):
        self.afc_controller = afc_ctrl
        self.env = wrapper_env

      def predict(self, obs, state=None, episode_start=None, deterministic=True):
        # Convert concatenated obs to dict
        obs_dict = {}
        for i, agent in enumerate(self.env.agent_order):
          start = i * self.env.per_agent_obs_size
          end = (i + 1) * self.env.per_agent_obs_size
          agent_obs = obs[start:end].reshape(self.env._per_agent_obs_space.shape)
          obs_dict[agent] = agent_obs

        # Call AFC controller
        actions_dict, new_state = self.afc_controller.predict(
          obs_dict, state, episode_start, deterministic
        )

        # Convert dict actions back to concatenated
        action_list = []
        for agent in self.env.agent_order:
          action = np.asarray(actions_dict[agent])
          # Reshape to match per-agent action space shape, then flatten
          action = action.reshape(self.env._per_agent_act_space.shape).reshape(-1)
          action_list.append(action)
        action = np.concatenate(action_list, axis=0).astype(np.float32)

        return action, new_state

    return ControllerAdapter(afc_controller, env)
  else:
    # Direct parallel_env - return AFC controller as-is
    return afc_controller

