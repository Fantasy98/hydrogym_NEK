# Agent assignment: fresh downloads and complete file initialization

Execute this assignment when the user invokes this prompt in a later task.
Updating this prompt does not authorize its execution in the editing session.

## Target and acceptance criteria

Create a new workspace at:

`/p/scratch/deepwing/yuningw/sandBox_HydroGym_fresh`

Download the pinned container and the complete required Hugging Face packages
again, starting with empty destinations and private empty caches. Prepare and
stage the large-channel and wing inputs, and verify every required file and
resolved path. Skip mini-channel preparation. Assume there are no local datasets
or checkpoints. Do not run the actual cases.

This replaces the earlier economical preparation audit. Its optional historical
report, if the user has retained the old sandbox, is at:

`/p/scratch/deepwing/yuningw/sandBox_HydroGym/examples/nek/zeroShot_Reproduce/validation/report.md`

That audit reused the external container and all 18 wing inputs, downloaded only
an 802-byte configuration probe, skipped mini-channel preparation, and did not
stage the real wing restart fields. Its passing checks do not establish fresh
download or complete initialization readiness. Do not reuse its assets or caches
to satisfy this assignment. Leave any retained sandbox/report intact. Their
availability is not a prerequisite for this fresh initialization task.

The authoritative portable source bundle is:

`/p/scratch/deepwing/yuningw/hydroGym_reproduce/examples/nek/zeroShot_Reproduce`

The scientific reference is:

`/p/scratch/deepwing/yuningw/hydroGym_reproduce/examples/nek/zero_shot_reproduce`

Treat both source locations, the previous sandbox, and all original inputs as
read-only. Make any implementation fixes and validation outputs only inside the
new workspace. Do not commit, push, publish, delete, or reset existing work.

The required acquisition modes are:

| Asset | Required action |
| --- | --- |
| HydroGym checkout | Fresh clone at the recorded commit |
| Portable reproduction code | Verified source-only export from the authoritative bundle |
| Container | Fresh pull of the pinned OCI digest into the new workspace |
| Large-channel HF package | Download all 42 pinned files, including all six restart sets |
| Mini-channel HF package | Skip; it is an independent optional example |
| Wing HF package | Download the entire package at the pinned revision; no local reuse |
| Large-channel executable/templates | Extract/copy from the newly downloaded container module |
| Exact channel restarts | Install independent copies from the new HF download; verify original hashes |
| Deployment model and VecNormalize | Future training outputs; report unavailable, do not import local files |

All 36 original channel restart sizes and SHA-256 values were checked against
HF metadata at the pinned revision on 22 October 2026 and match exactly.
`manifests/channel_hf.json` records all 42 package files (531,351,133 bytes).
Keep `manifests/channel_restarts.json` unchanged; use it to verify downloaded
initial conditions without accessing the original local dataset.

The pinned HF environment dataset has no trained policy/normalization pair.
Starting channel training from scratch does not require one. This no-simulation
task can validate wing input/staging readiness but cannot produce or claim a
ready deployment policy. Do not copy the old local pair or manufacture a model.
Retain its reference hashes and step 576000000 in dependencies.json for provenance;
report policy availability separately from file-initialization acceptance.

Success requires completed transfers, full file-integrity checks, and successful
real file staging. A download plan, small-file probe, size-only check, synthetic
test, existing PREPARED.json, or successful import alone is insufficient.

## Boundaries

No CFD, training, evaluation, inference, or real MPI execution. Never call real
`sbatch`, `srun`, `mpirun`, `mpiexec`, or `nek5000`; never use `--submit`. Do not run
controllers, `test_nek_direct.py`, benchmarks, or `NekEnv.reset/step`. Container
imports with MPI initialization disabled, downloads, file-only preparation and
staging, and tests with synthetic fixtures/mocked launchers are permitted.

Inspect the installed initialization code before calling it. If a path launches
a solver or initializes MPI, isolate its file-only preparation or report the
unvalidated step. Do not relax the execution boundary to complete a check.

Preserve scientific parameters, pinned versions, TD3 and normalization settings,
all six channel initial conditions, the wing v17 restart lifecycle, and policy
step 576000000. The checkout and container have different HydroGym versions;
do not install the checkout over the container library or select it through a
host PYTHONPATH. Fix routing or preparation defects without changing the science.

## 1. Establish a genuinely fresh workspace

Read applicable AGENTS.md files, README.md, dependencies.json,
SOURCE_VALIDATION.md, and the case READMEs. Read the previous report only if
available; do not require the old sandbox or scientific reference data.

The new destination must initially be absent. If it already contains an attempt
at this assignment, inspect its provenance before resuming only that attempt.
Do not silently treat old inputs as newly downloaded, and do not erase an
unrelated directory. Use a clearly recorded unused sibling if necessary, keeping
Nek filename lengths within the limit below.

```bash
source_bundle=/p/scratch/deepwing/yuningw/hydroGym_reproduce/examples/nek/zeroShot_Reproduce
sandbox_checkout=/p/scratch/deepwing/yuningw/sandBox_HydroGym_fresh
git clone --no-checkout https://github.com/dynamicslab/hydrogym.git "$sandbox_checkout"
git -C "$sandbox_checkout" checkout --detach 4ab9854dea3d84e38a59c25e0f5835a00cf8225f
"$source_bundle/tools/host_python.sh" "$source_bundle/tools/export_sources.py" \
    --destination "$sandbox_checkout/examples/nek/zeroShot_Reproduce"
bundle_root="$sandbox_checkout/examples/nek/zeroShot_Reproduce"
cd "$bundle_root"
```

Do not initialize the Firedrake submodule or install unrelated dependencies.
Verify the source inventory before and after export. Record commit, origin,
inventory digest, host, site, timestamps, and initially empty asset/cache paths.
Do not copy sites/local.sh from the previous sandbox: it selects reused assets.

Assume no Conda environment or user-installed Python packages are available.
Use tools/host_python.sh for host utilities: it selects /usr/bin/python3 (3.9+)
and ignores inherited Python settings/user-site packages. Host shell helpers use
it automatically. If necessary, configure ZS_HOST_PYTHON to another absolute
system Python path; do not depend on PATH resolving to the user's Conda Python.
Verify source checks, download plans, and launch previews with a clean environment
such as `env -i PATH=/usr/bin:/bin ./tools/host_python.sh tools/check_layout.py`.
Preserve legitimate site/network credentials for real transfers without printing
them. Container Python and Nek modules are supplied by the downloaded image.

Use a clean environment without inherited ZS_*, TCF_*, WING_*, HG_* path overrides
or host PYTHONPATH. Set ZS_SITE explicitly for JUWELS or JURECA. Inspect stale
Apptainer bind/environment overrides and HF cache overrides too. Preserve required
authentication without printing tokens. Explicitly route HF_HOME, HF_HUB_CACHE,
HUGGINGFACE_HUB_CACHE, XDG_CACHE_HOME, APPTAINER_CACHEDIR, APPTAINER_TMPDIR, and
corresponding container overrides into this bundle. Put any host temporary files
used for downloads/conversion here as well. Do not repurpose HOME or CODEX_HOME.

Record effective paths in both host and container contexts. All dataset, cache,
image, work, checkpoint, log, temporary download, and future result paths must
resolve under the new bundle. The only permitted external reads are source code,
optional read-only reference comparisons, and site system software. No original
local datasets or checkpoints may supply the new prepared inputs.
WING_LOCAL_SOURCE, WING_SOURCE_PREPARED, and WING_RESTART_FROM must be unset or
empty; ZS_IMAGE must select the new local image. No runtime link may point to an
original dataset or either older workspace.

Keep the complete `<solver-directory>/<casename>.rea` path at most 131 characters.
Under the proposed root, `validation/wing/s/naca_wing.rea` is 114 characters and
the default `2_naca0012_deploy/r40/s/naca_wing.rea` is 120. Recheck actual paths,
including any chosen alternative checkout name.

```bash
./tools/host_python.sh tools/check_layout.py
./tools/host_python.sh -m unittest discover -s tests -v
```

Before large transfers check filesystem free space AND applicable user/project
quota. Budget for the image, OCI conversion/cache, both HF package caches/local
copies, independent channel copies, and eight writable wing restart copies.
The wing package is 174,525,626,413 bytes; staging its eight restart fields adds
170,826,282,560 bytes without reflink savings. The historical SIF is about 34.5 GB.
These alone approach 380 GB before duplicate caches, temporary conversion space,
and the large-channel package/copies. A provisional 600–800 GB allowance is
reasonable; calculate the actual requirement from metadata and implementation.
Do not assume reflinks or the filesystem-wide free-space number imply sufficient
quota. Report a specific space limitation rather than falling back to reuse.

## 2. Download and verify the container

Require the new image path and private Apptainer caches to be empty initially.
Run the existing acquisition entry point:

```bash
0_setup/01_pull_hydrogym_image.sh
```

It must perform a fresh pull of the OCI digest in dependencies.json:

`docker://clagemann/hydrogym-nvhpc-26.1_cuda-12.9_turing_ampere@sha256:60a752cef00fd8dc6b505ca6faae6c1bf02abf6e19551e5e6e12250305ae9b53`

Do not copy, reflink, hardlink, or point ZS_IMAGE to the existing external SIF.
Capture download/conversion logs, final path, size, SHA-256, inspection metadata,
and evidence of the requested OCI identity. The new SIF must be a local regular
file independent of the reference. A newly converted SIF can differ bytewise
because of conversion metadata; its hash is not the OCI digest and need not equal
the historical SIF hash. Verify image identity and contents rather than skipping
verification because the SIF bytes differ.

Using tools/container_python.sh train|deploy, record interpreter,
hydrogym.__file__, versions, NEK_CASE_DIR, and MPI.Is_initialized() == False.
Compare with dependencies.json and any retained previous environment records.
Check the LargeChannel and NACA0012 module executables/templates. Do not mask a
discrepancy by installing or upgrading scientific dependencies. Host utilities
use Python 3.9+ standard library; container wrappers load their own Nek modules
and activate /home/easybuild/venvs/hydrogym_cpu. No host scientific stack is needed.

## 3. Download complete HF packages and verify their bytes

Use dataset `dynamicslab/HydroGym-environments` at exactly:

`cf5ef6748443ae72e9647f398487e98d421a7243`

Resolve and record this commit, never main. Retrieve the pinned repository's
complete file metadata for `TCFlarge_3D_Re207/` and
`NACA0012_3D_Re200000_AOA0/` before downloading. Save relative paths, exact sizes,
and available content identities in validation/hf-remote-manifest.json. Check the
large-channel identities against manifests/channel_hf.json and the original
restart manifest. Do not download unrelated environments or change revisions.

Start with empty package destinations and empty private HF caches. Download the
full wing package using the production preparation entry point:

```bash
2_naca0012_deploy/01_prepare_workspace.sh
2_naca0012_deploy/01_prepare_workspace.sh --download --accept-large-download
```

The first command is only the plan. The second must transfer and prepare all
18 required files, including all eight restart fields and the mask. Do not use
--reuse-existing or --reuse-local. Keep data beneath
`2_naca0012_deploy/data/NACA0012_3D_Re200000_AOA0/`, and caches beneath `.cache/`.
Require regular local payload files; no external links or hardlinks to old data.
Account for any additional files listed in the pinned package metadata.

Download the complete large-channel package and prepare its workspace:

```bash
./tools/host_python.sh tools/download_channel.py
1_tcf_train/01_prepare_tcf.sh --download
./tools/host_python.sh tools/channel_restarts.py
```

The first command is a side-effect-free plan. The second uses standard-library
host Python to download every pinned file into data/hf/TCFlarge_3D_Re207, verifies
LFS SHA-256 / ordinary Git blob identities and exact sizes, installs independent
restart copies in data/channel_restarts, and prepares the container-module case.
Inspect validation/channel-download.json. Require all 42 acquisition records to
show fresh downloads, or documented retries within this same fresh attempt.
No original local source path is involved. Do not run the mini-channel preparer.

For both packages, compare each downloaded file's exact size and content identity
with remote metadata. For LFS files, stream SHA-256 and compare with the advertised
LFS SHA-256. For ordinary Git blobs, compare the proper Git blob identity, not a
plain-file SHA-1 mislabeled as a blob ID; also record SHA-256 for local evidence.
Do not assume an HTTP ETag is a SHA-256. If remote metadata lacks a usable hash,
retrieve authoritative pinned object metadata or report the integrity gap.
Self-computed hashes alone are not proof of agreement with the remote content.
Use streaming reads for large fields and record before/after file stats to catch
concurrent changes. Validate restart headers and required inventory too.

The existing wing preparer primarily checks existence and minimum sizes. Its
success/PREPARED.json does not replace this exact-size and checksum audit. Save
per-file expected/actual sizes, identities, real paths, link types, and results.
Reject truncated payloads, LFS pointer text, unexpected missing files, and any
external cache resolution. Resolve and explicitly account for extra files and
local HF bookkeeping instead of silently ignoring them.

Do not prepopulate caches from previous work. Resuming a failed network transfer
within this same fresh attempt is permitted; record the interruption and resume.
Do not switch to local reuse if the network fails. The 802-byte download probe
can be used for diagnosis, but cannot satisfy the full-download requirement.

## 4. Verify exact downloaded restarts and policy boundaries

Require exactly 36 installed restart files, 530,950,176 bytes: init_1 through
init_6, each with rs6tcf0.f00001 through rs6tcf0.f00006. Check every size and
SHA-256 in the downloaded package and independent installed copies against the
unchanged manifests/channel_restarts.json. The canonical sha256sum-list digest is:

`0b894e9725d2ef90839a1285845c0b8bc2c428d521dc56ad700d6c4c0960dc4a`

Destination data/channel_restarts must contain independent regular files, no
source symlinks/hardlinks. Do not access original local restart directories or
mutable run/current_solver fields. Any mismatch fails this requirement.

Leave training checkpoints empty. Inspect the training driver's fresh-start path
without running it. Record that a model/VecNormalize pair is a future training
output and is not supplied by this HF environment dataset. Verify future policy
selection points inside the new bundle at the intended step; do not deserialize
models, perform inference, create latest.json, or import the previous experiment's
policy. Complete deployment readiness requires a real trained pair later.

## 5. Validate large-channel file initialization

The --download command above has prepared the workspace. Verify the copied
executable, mesh, templates, SIZE, tcf.usr, and int_pos against the newly downloaded
container's LargeChannel module. Check that package YAML resolves to the
bundle-owned 1_tcf_train/config/pettingzoo_tcflarge_re207.yml and restart links to
the independent data/channel_restarts. Preserve all bundled scientific values.
Raw HF tcf.par/int_pos are upstream templates and differ from the reproduction's
working settings/sensors; do not overwrite the scientific YAML or staged runtime
files with them. Runtime staging generates the intended parameters and sensors.
The HF tcf.re2 and tcf.ma2 identities match the historical channel mesh.

Read the installed NEK_INIT and invoked helpers before staging. Once confirmed
to be file-only, stage into a fresh short audit directory:

```bash
tools/container_python.sh train 1_tcf_train/02_stage_solver.py \
    --config "$bundle_root/1_tcf_train/config/pettingzoo_tcflarge_re207.yml" \
    --workspace "$bundle_root/1_tcf_train/workspace" \
    --restart-root "$bundle_root/data/channel_restarts" \
    --solver-dir "$bundle_root/validation/tcf/s"
```

Check generated tcf.par values against the YAML, sensor dimensions/bounds,
required mesh links/copies, restart selection, all six selected restart hashes,
and both workspace and solver SESSION.NAME values. Confirm writable restart
files are independent from the installed dataset and no input data changed.
The helper can intentionally regenerate workspace int_pos; distinguish that
prepared working file from the immutable module/package input.

Exercise the existing non-empty-directory refusal and verify it leaves the staged
case unchanged. Do not delete valid staged files to manufacture a second pass.

## 6. Stage and verify the real wing restart fields

Inspect generated config/runtime.yml, w/PREPARED.json, executable, SIZE, and
policy template. Require pinned HF provenance, full input inventory, and download
mode, with solver v17, random_init=-2, runner.rank=1, unchanged scientific
parameters, and the policy's intentional input divisor of 1.0.

Unlike the previous audit, real wing file staging is REQUIRED. It copies about
171 GB of restart fields but does not run the solver. Check remaining quota first
and invoke the existing file-only staging helper in a new directory:

```bash
tools/container_python.sh deploy 2_naca0012_deploy/02_stage_solver.py \
    --config "$bundle_root/2_naca0012_deploy/config/runtime.yml" \
    --environment-dir "$bundle_root/2_naca0012_deploy/data/NACA0012_3D_Re200000_AOA0" \
    --workspace "$bundle_root/2_naca0012_deploy/w" \
    --solver-dir "$bundle_root/validation/wing/s"
```

Verify all eight staged fields against the downloaded originals by exact size
and SHA-256. Require independent regular writable restart files and selector:
no symlinks/hardlinks to immutable inputs. A supported reflink is acceptable
because writes are independent; do not assume support or mutate real fields to
prove it. Check selector 0/1 and the corresponding committed four-file set,
header precision, increasing flow times and steps, and restart-input.json.
The selector, not modification times, determines the committed set.

Verify generated .rea parameters, executable/SIZE, runtime-config.yml,
SESSION.NAME, and every mesh/mask/sensor link. Immutable links may point only to
the newly downloaded bundle data. Check Nek filename length and configuration
paths in the actual container view.

Record hashes and selector content, then repeat the SAME staging command against
this initialized audit directory. Require reuse mode and unchanged restart field
hashes/selector. Refreshing small launch files is allowed. Do not alter selector,
truncate fields, or fabricate failures in the real dataset; use synthetic tests
for invalid-selector, incomplete-copy, and restart-lifecycle failure cases.

Do not create production flow state under r40/s or simulate a completed episode.
The real audit staging is under validation/wing/s; production startup remains
fresh. Do not call the controller to test reset, even with zero timesteps.

## 7. Check the complete layout and launch construction

```bash
./tools/host_python.sh -m unittest discover -s tests -v
tools/container_python.sh deploy -m unittest discover -s 2_naca0012_deploy/tests -v
./tools/host_python.sh tools/check_layout.py --prepared
./tools/host_python.sh tools/submit.py train --chain 2
./tools/host_python.sh tools/submit.py evaluate
./tools/host_python.sh tools/submit.py deploy --mv-data=false
```

Inspect lifecycle-test isolation first; all launchers must be mocked and fixtures
small. Submission commands above must remain previews. Verify 513 channel ranks,
4097 wing ranks, the selected site's node layout, exported case directories,
working directories, checkpoint selection, logs, and future output paths.
Validate paths as seen by host, container, and spooled-script settings.

check_layout.py --prepared covers only a subset of artifacts. Supplement it with
the full inventory, checksums, and real staging checks above;
do not equate its success with complete initialization verification.

Audit resolved links and configuration values across the complete new bundle.
Runtime dependencies must stay inside it (apart from system software); report
any remaining old-root path as a defect unless it is provenance text only.
Verify the source inventory and original restart manifest remain unchanged.
If reference locations exist, compare their metadata before/after the task and
explain concurrent changes. Their absence must not block fresh initialization.

If portability or initialization bugs appear, fix the fresh sandbox copy, add
focused tests only for substantive behavior, and rerun affected checks. Preserve
the failed evidence and describe the cause. Never silently fix a problem by
reusing an old asset, weakening an integrity check, or changing scientific input.
Regenerate the fresh bundle's source inventory after reviewed source fixes.

## 8. Deliver a complete evidence record

Write validation/report.md and validation/acceptance.json in the new bundle.
Use per-requirement statuses PASS, FAIL, or NOT_RUN with evidence filenames.
Required rows are: fresh checkout/export, fresh container pull/identity, pinned
HF metadata, full large-channel download/integrity, full wing download/integrity,
exact channel restart installation, channel preparation and staging, real wing
staging and reuse, path/cache containment, unit/lifecycle tests, launch previews,
and reference preservation. List mini preparation as NOT_RUN (user skipped it)
and policy loading as NOT_RUN (no trained model supplied), outside the required
initialization rows. Neither is a reason to import local scientific data.

Include commands, return codes, start/end times, transfer/resume logs, selected
versions, effective host/container paths, per-file expected/actual sizes/hashes,
file types, copy independence, quota accounting, and a final artifact-location
table. State explicitly which assets were downloaded, imported, or extracted.
Log locally without leaking credentials. Preserve unfinished-transfer evidence
and identify exact missing bytes/files/checks instead of calling an attempt done.

Only report **fresh-download and file-initialization validation passed** when all
required rows pass. Any required unavailable package, incomplete transfer,
missing authoritative integrity evidence, unverified real staging step, or
external flow-input dependency leaves this target incomplete. A missing trained
policy is an explicitly separate limitation: do not claim deployment readiness.
Continue independent checks where possible, but report remaining failures clearly.

Even a complete pass does not establish numerical reproduction, solver startup,
MPI communication, training stability, or deployment performance. Confirm that
no jobs, simulations, training, evaluation, inference, or real MPI launchers ran.
