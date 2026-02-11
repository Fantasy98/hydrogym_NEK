from typing import Callable, Iterable, Optional, Tuple

import numpy as np

from hydrogym.core import CallbackBase

from hydrogym.nek.nek_lib.nek_utils import show_title, show_end

def integrate(
    env,
    t_span: Tuple[float, float],
    dt: Optional[float] = None,
    callbacks: Iterable[CallbackBase] = [],
    controller: Optional[Callable] = None,
    max_steps: Optional[int] = None,
):
  """
  Integrate a nek environment through time.

  Args:
    env: Nek environment (NekMARLGymWrapper or parallel_env)
    t_span: Tuple of (start_time, end_time)
    dt: Time step (optional, uses env's default if not provided)
    callbacks: List of callbacks to evaluate throughout the solve
    controller: Controller object or function. Supports multiple formats:
      - SB3-style object: Object with `predict()` method (e.g., `model.predict(obs, state=..., episode_start=..., deterministic=True)`)
      - Legacy function: `action = controller(t, obs, env)` or `action = controller(t, obs)`
    max_steps: Maximum number of steps (optional)

  Returns:
    The environment after integration
  """
  show_title()
  t_start, t_end = t_span
  iter = 0
  t = t_start

  # Reset environment
  if hasattr(env, 'reset'):
    obs = env.reset()
  else:
    obs = None

  # Get time step from environment config if not provided
  if dt is None:
    if hasattr(env, 'conf') and hasattr(env.conf, 'simulation'):
      # Try to get dt from config
      dt = getattr(env.conf.simulation, 'dt', 0.01)
    else:
      dt = 0.01  # Default fallback

  # Calculate max steps if not provided
  if max_steps is None:
    max_steps = int((t_end - t_start) / dt) + 1

  # Track last reward for callbacks
  last_reward = 0.0
  
  # Initialize controller state for SB3-style controllers
  controller_states = None
  episode_start = np.ones((1,), dtype=bool)  # True on first step
  
  # Main integration loop
  while t < t_end and iter < max_steps:
    # Get controller action if provided
    if controller is not None:
      # Try SB3-style controller: controller.predict(obs, state=states, episode_start=episode_starts)
      # This pattern matches stable_baselines3 model.predict() signature
      if hasattr(controller, 'predict'):
        # Controller is an object with predict() method (e.g., SB3 model)
        action, controller_states = controller.predict(
            obs, state=controller_states, episode_start=episode_start, deterministic=True
        )
      else:
        # Simple fallback: controller is a callable function with (t, obs, env) or (t, obs) signature
        try:
          action = controller(t, obs, env)
        except TypeError:
          action = controller(t, obs)
    else:
      # Default: zero action
      if hasattr(env, 'action_space'):
        action = np.zeros(env.action_space.shape, dtype=np.float32)
      else:
        # For parallel_env, need dict of actions
        action = {agent: np.zeros(env.action_space(agent).shape, dtype=np.float32)
                  for agent in env.possible_agents}

    # Step the environment
    if hasattr(env, 'step'):
      result = env.step(action)
      if isinstance(result, tuple) and len(result) >= 2:
        obs, reward, done, info = result[:4]
        last_reward = reward
        # Store reward in env for callbacks
        env.last_reward = reward
        if done:
          break
      else:
        obs = result

    # Update time
    t = t_start + (iter + 1) * dt
    iter += 1
    
    # Update episode_start flag (False after first step)
    episode_start = np.zeros((1,), dtype=bool)

    # Call callbacks
    for cb in callbacks:
      cb(iter, t, env)

  # Close callbacks
  for cb in callbacks:
    if hasattr(cb, 'close'):
      cb.close()

  show_end()
  return env

