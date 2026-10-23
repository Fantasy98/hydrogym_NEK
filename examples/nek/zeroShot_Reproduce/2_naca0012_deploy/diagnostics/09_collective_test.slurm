#!/usr/bin/env bash
# Standalone MPI benchmark. Preview: python3 tools/submit.py collectives
# Does not launch HydroGym/Nek or read, copy, archive, or advance restart data.

set -euo pipefail
: "${WING_CASE_DIR:?Use tools/submit.py to supply the case path}"
source "${WING_CASE_DIR}/00_settings.sh"
[[ "${SLURM_NTASKS:?Submit with sbatch}" == 4097 && "${SLURM_JOB_NUM_NODES:?}" == "${WING_NODES}" ]] || {
    echo 'This test requires the production layout: the configured node count and 4097 tasks.' >&2
    exit 2
}
output="${WING_CASE_DIR}/diagnostics/collectives-${SLURM_JOB_ID}"
mkdir "${output}"
cp "${WING_CASE_DIR}/diagnostics/collective_bench.c" "${output}/"
cp "${WING_CASE_DIR}/diagnostics/collective_rank.sh" "${output}/"

# Open MPI 5.0.5 legacy ASCII rules: collective 2 = Allreduce.
# Communicator-size lower bounds 4096 and 4097 restrict changes to size 4096.
# Algorithm 0 restores fixed decisions outside the specified message interval.
cat > "${output}/small_rd.rules" <<'RULES'
1
2
3
1
1
0 0 0 0
4096
2
0 3 0 0
16384 0 0 0
4097
1
0 0 0 0
RULES
cat > "${output}/large_ring.rules" <<'RULES'
1
2
3
1
1
0 0 0 0
4096
3
0 0 0 0
1048576 4 0 0
8388608 0 0 0
4097
1
0 0 0 0
RULES

apptainer exec \
    --bind "$(zs_bind_paths)" \
    "${WING_IMAGE}" bash --noprofile --norc -c '
        set -eo pipefail
        source /usr/share/lmod/lmod/init/bash
        module --force purge
        module load "$1"
        mpicc -O2 -Wall -Wextra -Werror "$2/collective_bench.c" -o "$2/collective_bench"
        ompi_info --version > "$2/mpi-version.txt"
        ompi_info --param coll all --level 9 > "$2/collective-parameters.txt"
        ucx_info -v > "$2/ucx-version.txt"
    ' bash "${WING_NEK_MODULE}" "${output}"
scontrol show job "${SLURM_JOB_ID}" > "${output}/allocation.txt"
echo "Results: ${output}"

# Same srun transport and binding as production. New MPI processes are needed
# for each profile because MCA settings are read during MPI initialization.
run_profile() {
    local profile="$1" pass="$2" phase="$3"
    echo "Running ${phase}: ${profile}, pass ${pass}"
    srun --mpi="${ZS_MPI}" --ntasks=4097 --cpu-bind=cores --kill-on-bad-exit=1 \
        --time=00:08:00 \
        --output="${output}/${profile}-${phase}-${pass}.out" \
        --error="${output}/${profile}-${phase}-${pass}.err" \
        bash "${output}/collective_rank.sh" "$profile" "$pass" "$phase" "${output}"
}
# Probe default selection and the forced tuned path; do not use probe timings.
run_profile default 0 probe
run_profile small_rd 0 probe
# Reverse order on the second pass to expose drift/order effects.
for profile in default tuned_fixed small_rd large_ring; do run_profile "$profile" 1 timing; done
for profile in large_ring small_rd tuned_fixed default; do run_profile "$profile" 2 timing; done
"${ZS_ROOT}/tools/host_python.sh" "${WING_CASE_DIR}/diagnostics/summarize_collectives.py" "${output}"
