# Source bundle validation — 22 October 2026

This records source checks performed while preparing `zeroShot_Reproduce`.
Cloning and downstream preparation were explicitly deferred to the agent prompt.

## Follow-up: system Python without Conda

Host commands now use `tools/host_python.sh`, selecting `/usr/bin/python3` by
default instead of looking up Python on PATH. It checks Python 3.9+, ignores
PYTHONHOME/PYTHONPATH and user-site packages, and accepts an explicit absolute
`ZS_HOST_PYTHON` override. Scientific Python remains inside the pinned container.

All 16 tests, source syntax/layout checks (27 Python and 26 shell files), and the
download plan passed with `env -i PATH=/usr/bin:/bin`. Added regression checks
place a failing fake `python3` first on PATH and set invalid inherited Python
environment paths; the host launcher and submission previews still use system
Python successfully. Invalid interpreter overrides fail with an actionable error.

A real HTTPS metadata request through this clean system interpreter also passed:
all 42 pinned HF file identities still match. No dataset payload, container image,
or simulation was downloaded/launched as part of this follow-up.

## Follow-up: direct HF large-channel setup

On 22 October 2026, queried the pinned HF revision's file metadata and compared
all 36 large-channel restart sizes and SHA-256 values to the unchanged original
manifest: every value matched. The new `manifests/channel_hf.json` pins the complete
42-file, 531,351,133-byte package. Only metadata and two small text files were
retrieved for inspection; no flow fields or container image were downloaded.

`1_tcf_train/01_prepare_tcf.sh --download` now downloads that package, verifies
content identities, and installs exact independent restart copies before the
existing module-based preparation. Mini-channel setup is optional and no old
local data is required. The pinned HF dataset does not contain the trained
policy/normalizer; those remain future training outputs for a fresh experiment.

All 14 standard-library tests passed with the host `/usr/bin/python3` (3.9.25).
The six added tests use tiny mocked responses to check pinned URLs, ordinary Git
blob and LFS hashes, corrupt/truncated/oversized payload rejection, safe retries,
and refusal to write through symlink/traversal destinations. The download plan
also ran using system Python. Actual full transfer and container preparation via
this new route remain for the user's manual test or later sandbox task.

## Completed during initial bundle creation

- Parsed all 25 Python files without executing simulation entry points.
- Checked all 25 shell/Slurm files with `bash -n`.
- Parsed the dependency and provenance JSON files.
- Passed all 8 standard-library tests in `tests/test_portability.py`:
  JUWELS/JURECA allocation plans, settings in a renamed temporary directory,
  all six submission previews without invoking `sbatch`, exact independent
  restart copying with tiny fixtures, corruption rejection, rejection of
  hardlinked/symlinked destination fields, and source-only export isolation.
- Confirmed all 39 original source files recorded in `manifests/source_files.json`
  retain their original SHA-256 hashes. Changes are confined to this new bundle.
- Confirmed the core training driver, TCF staging helper, deterministic evaluator,
  wing staging helper, deployment driver, and archive script remain byte-identical
  to their reference versions (six files).
- Compared the training YAML values and the wing runner/simulation/policy values
  with the originals; only source paths, comments and portability plumbing changed.
- Preserved the earlier read-only audit's exact 36-file channel restart manifest:
  530,950,176 bytes and manifest digest
  `0b894e9725d2ef90839a1285845c0b8bc2c428d521dc56ad700d6c4c0960dc4a`.
- Hashed the reference episode-40 model/VecNormalize files without deserializing
  either; recorded their sizes and hashes in `dependencies.json`.
- Confirmed `/p/scratch/deepwing/yuningw/sandBox_HydroGym` was not created.

## Deferred

- Fresh Git clone and export of the actual bundle into the sandbox.
- Container inspection, image acquisition/copying, and container import checks.
- Download probe and any full dataset downloads.
- Import of the real channel restart files or model pair into a new workspace.
- TCF/wing preparation and generated configuration validation in the container.
- Integration of the new wing immutable-input reuse path with the real container.
- The existing 10 deployment restart lifecycle tests, which need OmegaConf and
  the container module/venv used by their mocked launcher shells. The host Python
  used for source checks does not provide OmegaConf; no dependency was installed.
- Optional mini-channel availability at the pinned HF revision.
- Actual multi-node MPI, simulations, training, deployment and numerical results.

No container command, network download, case preparation, Slurm submission,
training, evaluation, inference, or CFD solver was run during bundle creation.
Temporary unit-test copies contained source fragments or tiny synthetic data only.
The later agent must distinguish preparation validation from numerical reproduction.
