# Portable Nek zero-shot reproduction

This directory contains the reproduction workflow: container setup, the optional
mini-channel test, resumable large-channel TD3 training, deterministic evaluation,
NACA0012 deployment, archiving, diagnostics, and validation tools. It can be copied
outside the HydroGym checkout. HydroGym and Nek5000 are supplied by the pinned
container; this is not a source build of those dependencies.

The next agent's complete assignment is in [prompt/AGENT_PROMPT.md](prompt/AGENT_PROMPT.md).
Its current target is `/p/scratch/deepwing/yuningw/sandBox_HydroGym_fresh`: download
the pinned image and full large-channel/wing packages from remote sources,
verify the original channel restart hashes, and check complete file preparation
and real restart staging without running cases. Mini-channel preparation is
optional; policy availability is reported separately. The earlier `sandBox_HydroGym`
audit reused the image and wing data; it does not satisfy this fresh-download
target. This prompt update does not execute the new assignment.

## Contents

- `0_setup/`: pinned image acquisition and optional mini-channel preparation/test.
- `1_tcf_train/`: training, evaluation, plotting, and the reproduction-owned YAML.
- `2_naca0012_deploy/`: deployment, policy configuration, archive lifecycle, tests,
  and optional MPI diagnostics under `diagnostics/`.
- `common/`, `sites/`: shared paths and JUWELS/JURECA allocation settings.
- `tools/`: source export, checksum imports, download probe, validation, submission.
- `manifests/`: exact channel restart hashes, original source provenance, bundle inventory.
- `dependencies.json`: upstream commit, container identity, dataset revision, policy pair.

## Safe source checks

From this directory, these commands create no case state and do not access the network:

```bash
./tools/host_python.sh tools/check_layout.py
./tools/host_python.sh -m unittest discover -s tests -v
./tools/host_python.sh tools/download_probe.py
./tools/host_python.sh tools/submit.py train
./tools/host_python.sh tools/submit.py evaluate
./tools/host_python.sh tools/submit.py deploy --mv-data=false
```

The unit tests use tiny temporary fixtures. The preserved deployment lifecycle
tests additionally require the container environment; run them during the later
sandbox validation, as specified in the prompt.

## Configuration and destinations

`common/settings.sh` discovers `ZS_ROOT` from its own location. `ZS_SITE` selects
`juwels` (default) or `jureca`. Copy `sites/local.sh.example` to `sites/local.sh`
in the intended working copy to configure local assets. Settings retain explicit
environment overrides; start a fresh shell for a different workspace to avoid
inheriting an old `ZS_ROOT`, `TCF_*`, `WING_*`, or `HG_*` value.

On JUWELS no export is necessary: `juwels` is the default in every new terminal.
For JURECA, use `export ZS_SITE=jureca` in each session (or prefix commands with
`ZS_SITE=jureca`). A shell export lasts only for that shell and its children.
The current loader selects the site before reading `sites/local.sh`; do not put
the site selector in that file. It is for path/account/resource overrides.

No Conda environment is needed. `tools/host_python.sh` selects `/usr/bin/python3`
explicitly, requires Python 3.9+, and ignores PYTHONHOME, PYTHONPATH, and user-site
packages. Host preparation/submission helpers use this launcher automatically;
use it instead of plain `python3` for direct host utility commands. On this JUWELS
installation the system interpreter is 3.9.25. If another machine needs a different
system interpreter, set `ZS_HOST_PYTHON` to its absolute executable path (also
supported in `sites/local.sh`). Only the Python standard library and Bash are
needed on the host; there is no Conda activation or pip installation step.

Container entry points need `apptainer` on PATH and container access. They load
the Nek module and activate `/home/easybuild/venvs/hydrogym_cpu` inside the pinned
image. No host HydroGym, PyTorch, mpi4py, or Nek module is needed for preparation.
Scientific Python remains supplied by that image. Avoid propagating an unrelated
host PYTHONPATH into the container.

| Item | Default beneath this directory |
| --- | --- |
| Container image | `containers/images/hydrogym-nvhpc-26.1.sif` |
| Container download cache / temporary files | `containers/cache/`, `containers/tmp/` |
| HF and Python caches | `.cache/` |
| Original channel initial conditions | `data/channel_restarts/init_1` … `init_6` |
| Complete pinned HF large-channel package | `data/hf/TCFlarge_3D_Re207/` |
| Channel prepared input package | `1_tcf_train/data/TCFlarge_3D_Re207/` |
| Channel executable / case templates | `1_tcf_train/workspace/` |
| Channel training state / checkpoints | `1_tcf_train/run/` |
| Channel deterministic evaluations | `1_tcf_train/evaluate/` |
| Wing input dataset | `2_naca0012_deploy/data/NACA0012_3D_Re200000_AOA0/` |
| Wing generated configuration / executable | `2_naca0012_deploy/config/`, `2_naca0012_deploy/w/` |
| Wing persistent flow / job records | `2_naca0012_deploy/r40/s/`, `r40/j-<jobid>/` |
| Wing archived results | `2_naca0012_deploy/results40/` |
| Validation evidence | `validation/` |

Keep paths short. Nek constructs filenames in a 132-character buffer; evaluation
and wing staging require the complete `<solver-directory>/<casename>.rea` path
to be at most 131 characters. The `s` directory names are deliberate.

## Exact inputs

The channel workflow uses the six `rs6_lc_noresize_reth365` initial conditions
from the original experiment. `manifests/channel_restarts.json` pins all 36 files
and their sizes: 530,950,176 bytes. The HF package at the pinned revision has
exactly the same sizes and SHA-256 values for all 36 restart files, verified
against remote metadata on 22 October 2026. No original local data is needed.
The additional `manifests/channel_hf.json` pins all 42 package files, totaling
531,351,133 bytes. Start directly with the large channel:

```bash
# From this directory; these commands do not launch a simulation.
./0_setup/01_pull_hydrogym_image.sh
./1_tcf_train/01_prepare_tcf.sh --download
```

Skip `0_setup/02_prepare_nek_workspace.sh` unless you specifically want the
independent mini-channel example. `tools/download_channel.py` prints a plan by
default; `--download` downloads all 42 files using host standard-library Python,
checks exact sizes and HF content identities, then installs independent restart
copies after checking the original manifest again. No HF Python package is needed
on the host. Evidence is written to `validation/channel-download.json`.
Re-running verifies existing files and downloads only missing ones; mismatched
files fail without being overwritten. Source templates remain in the raw HF
package. The reproduction YAML and module templates still govern runtime staging;
do not replace generated parameters/sensors with the raw HF template files.

The optional offline import remains available:

```bash
./tools/host_python.sh tools/channel_restarts.py --source /path/to/original/restart/root --install
./tools/host_python.sh tools/channel_restarts.py
```

The importer verifies the source, copies regular independent files, checks the
copies, and then publishes the destination. It refuses mismatches and does not
replace an existing different dataset. Channel preparation verifies these hashes
again. No similarly named downloaded channel package is an acceptable substitute.

`dependencies.json` also pins the episode-40 model/VecNormalize pair. These are
training outputs, not present in the pinned HF environment dataset. Starting
large-channel training needs no existing model; deployment later needs its trained
model and normalizer, or a separately obtained verified pair. Optional
`tools/import_policy.py --source /path/to/checkpoints --install` copies only this
pair after checking both hashes. It does not load pickle/model objects or create
`latest.json`. Resuming training requires the matching replay buffer as well.

The wing HF revision is fixed to the revision recorded by the original
`PREPARED.json`. Container OCI digest and reference SIF checksum are different
identities: a fresh SIF conversion may have different container-file bytes.
The reference SIF checksum is historical evidence, not a new hash computed here.

## Preparation and submission

Preparation commands are described in each case README and the agent prompt.
They write files or download inputs and are deferred to the sandbox task.
`tools/submit.py` and the chain/deployment wrappers only print plans by default.
Only `--submit` actually calls `sbatch`. Allocation arguments and log paths are
resolved before submission, and the wrapper exports the case path for Slurm's
spooled script. Use the wrapper instead of submitting a `.slurm` file directly.

The initialization sandbox task must never use `--submit`, `srun`, `mpirun`, a Nek
executable, or any controller's simulation entry point. The optional smoke and
MPI diagnostic launchers are included for later experiments, not that task.

## Export

Use `tools/export_sources.py --destination /path/to/empty/destination` to copy the
files enumerated by `manifests/bundle_files.json`. It verifies file hashes and
excludes downloaded data, checkpoints, caches, logs, local settings, and results.
After intentionally editing distributed source files, regenerate that inventory
before exporting. Original source hashes remain in `manifests/source_files.json`.

Read [SOURCE_VALIDATION.md](SOURCE_VALIDATION.md) for checks completed during
bundle creation and the checks explicitly deferred to the sandbox.
