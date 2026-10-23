# NACA0012 zero-shot deployment

`meta_policy_naca_wing.py` remains the source of deployment parameters. It
deploys the fixed episode-40 policy (step 576000000) on the compiled suction-side
actuators. The policy path is derived from this bundle or overridden with
`WING_TCF_CHECKPOINT_DIR`. The preparer merges these values with the pinned HF
mesh/layout configuration and generates `config/runtime.yml`.

Preparation commands, to be run later in the sandbox:

```bash
# Plan only, even when local reuse settings have been configured:
./01_prepare_workspace.sh

# Download approximately 174.53 GB into this case's data directory:
./01_prepare_workspace.sh --download --accept-large-download

# Or explicitly reuse existing immutable HF data with recorded provenance:
# Set WING_LOCAL_SOURCE to the original environment directory and
# WING_SOURCE_PREPARED to its original w/PREPARED.json in sites/local.sh.
./01_prepare_workspace.sh --reuse-existing
```

Reuse creates input symlinks inside the new case, records their origin, and
checks the reference repository, environment, and HF revision. Keep those two
settings available for all subsequent container commands; the source dataset
and provenance directory are bound read-only. Reuse does not prove a fresh
174 GB transfer; `tools/download_probe.py --download` separately tests retrieval
of one pinned configuration into the sandbox. The downloaded/reused input
files must remain immutable. Persistent solver restarts are independent writable
copies produced by the existing staging logic.

From the bundle root, `./tools/host_python.sh tools/submit.py deploy --mv-data=false` previews
the first deployment. Use the default `--mv-data=true` for later continuations.
The wrapper requires explicit `--submit` to launch, which the preparation audit
must never use. Default resources are 4096 Nek ranks plus one controller,
86 JUWELS nodes or 33 JURECA nodes, and a 24-hour allocation. The controller
deadline remains 86040 seconds (23.9 hours); no dated begin time or personal
email address is embedded in the launcher.

The existing lifecycle is preserved: `r40/s` is writable persistent flow state,
`r40/j-<jobid>` records each allocation, and `results40/res_X` archives the previous
run. `random_init=-2` preserves controlled-flow restarts. Solver locks prevent
concurrent mutation; the Nek `.restart` selector determines the committed set.
The model stays pinned even if TCF training advances. `u_tau=1.0` in the policy
input divisor is intentional because the v17 executable performs the scaling.

MPI/UCX diagnostics are under `diagnostics/` and are excluded from the no-run
audit. Existing restart lifecycle tests use synthetic files and mocked launchers:

```bash
# Run from the bundle root, later with the container available:
tools/container_python.sh deploy -m unittest discover -s 2_naca0012_deploy/tests -v
```

No production state is imported automatically. The model/normalizer pair can be
copied with `tools/import_policy.py`; the initial HF restart dataset and a
previous controlled-flow checkpoint are distinct inputs.
