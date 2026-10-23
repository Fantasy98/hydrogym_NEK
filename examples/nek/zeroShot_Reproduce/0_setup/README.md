# Container and optional mini-channel preparation

`01_pull_hydrogym_image.sh` obtains the pinned OCI image under
`ZS_ROOT/containers/`. An existing SIF can instead be selected with `ZS_IMAGE`.
The image is about 34.5 GB; registry transfer can be larger. No pull was performed
while assembling this bundle.

`02_prepare_nek_workspace.sh` downloads the mini-channel environment at the fixed
dataset revision, stages `train_run`, and prepares `SESSION.NAME`. Its Python
helpers are included here, so it does not depend on `getting_started/`.
For this stage the bundle is mounted at `/work` inside the container; paths in
its generated session/config files therefore refer to `/work/0_setup/train_run`.
The paired smoke launcher uses exactly that same mapping.

The mini-channel test is optional and uses opposition control, `-obs[1::2]`.
It can be skipped entirely when starting large-channel training. After the image
pull, run `1_tcf_train/01_prepare_tcf.sh --download` from the bundle root instead.
It is not the large-channel training case and must not be run during the sandbox
preparation audit. Inspect its allocation with `./tools/host_python.sh tools/submit.py smoke`
from the bundle root. HF metadata confirms this optional package exists at the
pinned revision; a complete mini-channel download/preparation has not been tested.
