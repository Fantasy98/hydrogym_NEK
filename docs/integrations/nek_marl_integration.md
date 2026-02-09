# Nek MARL Integration Summary

This note summarizes the main integration challenges between HydroGym and
`/home/yuninw/codes/drl/drl_repo_TEST`, plus practical paths forward. It includes
code snippets that highlight the key mismatches.

## Challenges

### Gym single-agent vs PettingZoo MARL
HydroGym exposes a `gym.Env` with a single observation/action vector, while the
Nek setup uses a PettingZoo `ParallelEnv` returning per-agent dicts.

```323:376:hydrogym/core.py
class FlowEnv(gym.Env):
  def __init__(self, env_config: dict):
    self.flow: PDEBase = env_config.get("flow")(
        **env_config.get("flow_config", {}))
    self.solver: TransientSolver = env_config.get("solver")(
        self.flow, **env_config.get("solver_config", {}))
...
  def step(
      self,
      action: Iterable[ArrayLike] = None
  ) -> Tuple[ArrayLike, float, bool, dict]:
    self.solver.step(self.iter, control=action)
    self.iter += 1
...
    obs = self.flow.get_observations()
    reward = self.get_reward()
    done = self.check_complete()
```

```314:375:home/yuninw/codes/drl/drl_repo_TEST/src/nek_marl.py
    def step(self, actions):
        ...
        # Sending the new action values to the environment
        self.action(ctrl_value)
        # Let the solution evolve with the new control values
        rewards = self.evolve()
        # Obtain new observation
        flow_time, observation = self.state()
        # Distribute observations to the agents
        observations = self._distribute_field(observation,reward=False)
        ...
        if flow_time > self.conf.simulation.tmax:
            dones = {agent : True for agent in self.agents}
...
        return observations,rewards,dones,infos
```

### Solver backend architecture (Firedrake vs Nek5000)
HydroGym expects a solver object (`TransientSolver`) that steps in-process,
while Nek uses MPI message passing and file-based staging inside the env.

```430:497:home/yuninw/codes/drl/drl_repo_TEST/src/nek_marl.py
    def state(self):
        request = b'STATE'
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])
        ...
        self.sub_comm.Recv([current_time,MPI.DOUBLE],0,tag=1998)
        ...
    def action(self,ctrl_value:dict):
        request=b"CNTRL"
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],dest=0,tag=tag_dict['COMMAND']['tag'])
        ...
        self.sub_comm.Send([act_buffer,tag_dict['ACTION']['mpi_dtype']],nid,
                            tag=nid+tag_dict['ACTION']['tag'])
```

### Reward semantics (scalar vs per-agent)
HydroGym expects a scalar reward (`evaluate_objective`), while Nek produces
per-agent rewards derived from wall shear stress.

```501:578:home/yuninw/codes/drl/drl_repo_TEST/src/nek_marl.py
    def evolve(self):
        request=b"EVOLV"
        self.sub_comm.Send([request,tag_dict['COMMAND']['mpi_dtype']],
                            dest=0,tag=tag_dict['COMMAND']['tag'])
        ...
        for il, nid in enumerate(self.uniqID): 
            ...
            r_reward   = ws_stress_buffer[agent_name]
            i_reward   = self._normalize_reward(r_reward)
            rewards[agent_name] = i_reward
        ...
        return rewards
```

### Action/observation scaling and ZNMF constraints
The Nek env rescales actions and can enforce ZNMF conditions, which do not map
directly to HydroGym’s `MAX_CONTROL` and actuator model.

### MPI lifecycle and process control
Nek assumes an MPI master/worker split and inter-communicators at runtime, while
HydroGym does not manage MPI spawning.

### File-based IO and case management
Nek relies on case file staging and rewriting (e.g., `.re2`, `.par`), whereas
HydroGym’s Firedrake flows use mesh/checkpoint methods.

```44:120:home/yuninw/codes/drl/drl_repo_TEST/src/lib/nek_utils.py
class NEK_INIT():
    def get_Case_Files(self):
        ...
        for fname in checklist["must"]:
            from_file = os.path.join(self.nek.compile_path, fname)
            to_file = os.path.join(self.rank_folder, fname)
            ...
            shutil.copy(from_file, to_file)
```

## Possible paths (with minimal modifications)

### Path A: Adapter wrapper (lowest effort)
Keep the existing Nek PettingZoo MARL env and wrap it with a HydroGym-compatible
`gym.Env`:
- Concatenate per-agent observations into a single vector.
- Split/reshape a single action vector into per-agent actions.
- Aggregate per-agent rewards into a scalar (mean or weighted sum).
- Preserve per-agent logging in the existing Nek reward logger.

This is the lowest-risk option and keeps the current MPI workflow intact.

### Path B: Native HydroGym backend (moderate effort)
Create a Nek backend inside HydroGym:
- Implement `NekFlowConfig(PDEBase)` and `NekTransientSolver(TransientSolver)`
  using the MPI commands (`STATE`, `CNTRL`, `EVOLV`).
- Map Nek file staging to `save_checkpoint`/`load_checkpoint`.
- Add optional multi-agent support or an aggregation layer.

This yields a cleaner architecture but requires more refactor.

### Path C: Hybrid (keep MARL stack, add Gym layer)
Keep the PettingZoo MARL implementation and provide a thin Gym wrapper only for
HydroGym workflows, leaving your MARL training stack unchanged.

## Recommended starting point
Start with Path A or C. It preserves the Nek/MARL system with minimal changes,
while still letting HydroGym run at the API level for integration with the rest
of the ecosystem.
