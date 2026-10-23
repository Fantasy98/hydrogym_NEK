# Large-channel training and evaluation

The copied scientific drivers retain the original TD3 training, checkpoint,
normalization, restart, and evaluation logic. The local
`config/pettingzoo_tcflarge_re207.yml` owns the scientific parameters.
The six initial-condition sets must pass `../tools/channel_restarts.py` first.
All 36 original restart hashes match the pinned HF dataset, so no old local
dataset is needed. From the bundle root, after downloading the container:

```bash
./1_tcf_train/01_prepare_tcf.sh --download
```

This downloads and checks the entire 42-file (~531 MB) HF large-channel package,
stores it in `data/hf/TCFlarge_3D_Re207`, installs independent exact restart copies
in `data/channel_restarts`, then prepares the channel workspace. Mini-channel
preparation is not a prerequisite. The downloader uses host Python 3.9+ with no
third-party packages. `./tools/host_python.sh tools/download_channel.py` prints its plan only.
No solver, training process, or batch job is started by preparation.

`01_prepare_tcf.sh` creates the local environment package and copies the channel
case files/executable from the container's LargeChannel module. It does not run
Nek. `02_stage_solver.py` generates an allocation's disposable solver directory.
Without `--download`, preparation expects verified restarts to exist already.
The raw HF `.par` and sensor file are upstream templates; runtime staging uses the
reproduction-owned YAML to generate the actual parameters and sensor locations.
Training from scratch creates a new policy when no complete checkpoint exists;
do not import the old model merely to initialize a fresh training experiment.

Future runs use one controller and 512 Nek ranks. Site profiles choose 11 JUWELS
nodes or 5 JURECA nodes. From the bundle root, preview commands with:

```bash
./tools/host_python.sh tools/submit.py train
./tools/host_python.sh tools/submit.py train --chain 5 --begin=now+2hours
./tools/host_python.sh tools/submit.py evaluate
```

These commands do not submit. Explicit `--submit` is reserved for an actual
experiment after preparation validation; it is forbidden in the current agent
prompt. `04_submit_chain.sh 5 --begin=now+2hours` is another preview entry point.

Training continuation lives under `run/checkpoints` as complete
model/replay-buffer/VecNormalize triplets; solver histories are archived between
allocations and `run/current_solver` is rebuilt. `run/COMPLETE` prevents another
training segment after the target is reached. Training holds its existing lock.

Deterministic evaluation snapshots the selected model/VecNormalize pair and
uses its own `evaluate/e-<jobid>/s` directory. The default checkpoint is
576000000 and the default sweep is `init_1` … `init_6`. Its results include
per-actuator observations, policy actions, actuations, rewards and timeseries.
`06_plot_reward.py` plots training history; `10_postprocess_deterministic.py`
analyses evaluations. Both read the shared container/path settings.

Preparation and container integration have not been run for this new copy.
See the root prompt for the later checks and exact reference input locations.
