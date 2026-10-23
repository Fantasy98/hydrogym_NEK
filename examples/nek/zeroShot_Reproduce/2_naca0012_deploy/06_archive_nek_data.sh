#!/usr/bin/env bash
# Classify and retain Nek5000 outputs in the traditional results/res_X layout.
# Adapted from marl_energy_cost/utils/mv-data. Moving within the same
# filesystem is fast; restart snapshots are copied to preserve live state.

set -euo pipefail
shopt -s nullglob

RUN_ROOT=""
RESULTS_ROOT=""
CASE_NAME="naca_wing"
SOLVER_EXIT_CODE="0"
SOLVER_LOCK_HELD=false

usage() {
    cat <<EOF
Usage: $(basename "$0") --run-root PATH [OPTIONS]

  --run-root PATH          r/j-<jobid> directory (required)
  --results-root PATH      parent of sequential res_X directories (required)
  --case-name NAME         Nek case name                    [${CASE_NAME}]
  --solver-exit-code CODE  exit status returned by srun     [${SOLVER_EXIT_CODE}]
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --run-root)        RUN_ROOT="$2"; shift 2 ;;
        --results-root)    RESULTS_ROOT="$2"; shift 2 ;;
        --case-name)       CASE_NAME="$2"; shift 2 ;;
        --solver-exit-code) SOLVER_EXIT_CODE="$2"; shift 2 ;;
        --solver-lock-held) SOLVER_LOCK_HELD=true; shift ;;
        -h|--help)         usage; exit 0 ;;
        *) echo "[ARCHIVE] unknown argument: $1" >&2; exit 2 ;;
    esac
done

[[ -n "${RUN_ROOT}" ]] || { echo "[ARCHIVE] --run-root is required" >&2; exit 2; }
[[ -n "${RESULTS_ROOT}" ]] || { echo "[ARCHIVE] --results-root is required" >&2; exit 2; }
[[ "${SOLVER_EXIT_CODE}" =~ ^[0-9]+$ ]] || {
    echo "[ARCHIVE] invalid solver exit code: ${SOLVER_EXIT_CODE}" >&2; exit 2; }
[[ -d "${RUN_ROOT}" ]] || { echo "[ARCHIVE] run root not found: ${RUN_ROOT}" >&2; exit 1; }

RUN_ROOT="$(cd "${RUN_ROOT}" && pwd)"
SOLVER_DIR="${RUN_ROOT}/s"
[[ -d "${SOLVER_DIR}" ]] || { echo "[ARCHIVE] solver directory not found: ${SOLVER_DIR}" >&2; exit 1; }
SOLVER_DIR="$(cd "${SOLVER_DIR}" && pwd -P)"

# The job passes its inherited fd 7. Manual archives acquire the same lock.
if [[ "${SOLVER_LOCK_HELD}" == true ]]; then
    [[ "$(readlink "/proc/$$/fd/7")" == "${SOLVER_DIR}.lock" ]] || {
        echo "[ARCHIVE] missing inherited solver lock" >&2; exit 1;
    }
else
    exec 7>"${SOLVER_DIR}.lock"
fi
flock -n 7 || { echo "[ARCHIVE] solver is in use: ${SOLVER_DIR}" >&2; exit 1; }

# Old job links point at live state. Never attribute a later job's fields to
# an older job; a completed historical archive can still be queried safely.
if [[ -s "${RUN_ROOT}/RESULT_PATH.txt" ]]; then
    IFS= read -r saved_result < "${RUN_ROOT}/RESULT_PATH.txt"
    if [[ -f "${saved_result}/ARCHIVE_COMPLETE" ]]; then
        echo "[ARCHIVE] already complete: ${saved_result}"
        exit 0
    fi
fi
if [[ -s "${SOLVER_DIR}.last-run" ]]; then
    IFS= read -r last_run < "${SOLVER_DIR}.last-run"
    [[ "${last_run}" == "${RUN_ROOT}" ]] || {
        echo "[ARCHIVE] live solver outputs belong to ${last_run}, not ${RUN_ROOT}" >&2; exit 1;
    }
fi

mkdir -p "${RESULTS_ROOT}"
RESULTS_ROOT="$(cd "${RESULTS_ROOT}" && pwd)"

# A job keeps its allocation in RESULT_PATH.txt. This makes a manual retry
# idempotent. The global lock prevents simultaneous jobs from choosing the
# same next res_X number.
result_path_file="${RUN_ROOT}/RESULT_PATH.txt"
exec 8>"${RESULTS_ROOT}/.res-index.lock"
flock 8

if [[ -s "${result_path_file}" ]]; then
    IFS= read -r ARCHIVE_DIR < "${result_path_file}"
    result_name="$(basename "${ARCHIVE_DIR}")"
    if [[ "$(dirname "${ARCHIVE_DIR}")" != "${RESULTS_ROOT}" || \
          ! "${result_name}" =~ ^res_[0-9]+$ ]]; then
        echo "[ARCHIVE] invalid saved result path: ${ARCHIVE_DIR}" >&2
        exit 1
    fi
    mkdir -p "${ARCHIVE_DIR}"
else
    last_index=0
    for existing in "${RESULTS_ROOT}"/res_*; do
        [[ -e "${existing}" || -L "${existing}" ]] || continue
        existing_name="$(basename "${existing}")"
        if [[ "${existing_name}" =~ ^res_([0-9]+)$ ]] && \
           (( 10#${BASH_REMATCH[1]} > last_index )); then
            last_index=$((10#${BASH_REMATCH[1]}))
        fi
    done
    result_name="res_$((last_index + 1))"
    ARCHIVE_DIR="${RESULTS_ROOT}/${result_name}"
    mkdir "${ARCHIVE_DIR}"
    result_path_tmp="${RUN_ROOT}/.RESULT_PATH.txt.$$"
    printf '%s\n' "${ARCHIVE_DIR}" > "${result_path_tmp}"
    mv -f -- "${result_path_tmp}" "${result_path_file}"
fi

flock -u 8

result_link="${RUN_ROOT}/result"
if [[ -L "${result_link}" ]]; then
    if [[ "$(readlink -f "${result_link}")" != "${ARCHIVE_DIR}" ]]; then
        echo "[ARCHIVE] existing result link points elsewhere: ${result_link}" >&2
        exit 1
    fi
elif [[ -e "${result_link}" ]]; then
    echo "[ARCHIVE] refusing to replace existing path: ${result_link}" >&2
    exit 1
else
    ln -s "${ARCHIVE_DIR}" "${result_link}"
fi

exec 9>"${ARCHIVE_DIR}/archive.lock"
flock 9

if [[ -f "${ARCHIVE_DIR}/ARCHIVE_COMPLETE" ]]; then
    echo "[ARCHIVE] already complete: ${ARCHIVE_DIR}"
    exit 0
fi

LA2_DATA="${ARCHIVE_DIR}/la2_data"
STS_DATA="${ARCHIVE_DIR}/sts_data"
PTS_DATA="${ARCHIVE_DIR}/pts_data"
RS_DATA="${ARCHIVE_DIR}/rs_data"
REG_DATA="${ARCHIVE_DIR}/reg_data"
TXT_DATA="${ARCHIVE_DIR}/txt_data"
DRL_DATA="${ARCHIVE_DIR}/drl_data"
REC_DATA="${DRL_DATA}/drlrec"
POLICY_DATA="${DRL_DATA}/policy"
MONITOR_DATA="${DRL_DATA}/monitor"
CONFIG_DATA="${ARCHIVE_DIR}/config_data"
MISC_DATA="${ARCHIVE_DIR}/misc_data"
mkdir -p "${LA2_DATA}" "${STS_DATA}" "${PTS_DATA}" "${RS_DATA}" \
         "${REG_DATA}" "${TXT_DATA}" "${DRL_DATA}" "${REC_DATA}" \
         "${POLICY_DATA}" "${MONITOR_DATA}" "${CONFIG_DATA}" "${MISC_DATA}"

date --iso-8601=seconds > "${ARCHIVE_DIR}/ARCHIVE_IN_PROGRESS"

move_one() {
    local source="$1" destination="$2"
    local target="${destination}/$(basename "${source}")"
    if [[ -e "${target}" || -L "${target}" ]]; then
        echo "[ARCHIVE] refusing filename collision: ${target}" >&2
        return 1
    fi
    mv -- "${source}" "${destination}/"
}

move_pattern() {
    local pattern="$1" destination="$2" label="$3"
    local source count=0
    while IFS= read -r -d '' source; do
        move_one "${source}" "${destination}"
        count=$((count + 1))
    done < <(find "${SOLVER_DIR}" -maxdepth 1 -type f -name "${pattern}" -print0)
    (( count > 0 )) && echo "[ARCHIVE] ${label}: ${count} file(s)"
    return 0
}

copy_config() {
    local name="$1"
    if [[ -f "${SOLVER_DIR}/${name}" ]]; then
        cp -p -- "${SOLVER_DIR}/${name}" "${CONFIG_DATA}/"
    fi
}

move_directory() {
    local source="$1" destination="$2" label="$3"
    local item count=0
    [[ -d "${source}" ]] || return 0
    while IFS= read -r -d '' item; do
        move_one "${item}" "${destination}"
        count=$((count + 1))
    done < <(find "${source}" -mindepth 1 -maxdepth 1 -print0)
    (( count > 0 )) && echo "[ARCHIVE] ${label}: ${count} item(s)"
    return 0
}

echo "[ARCHIVE] classify ${SOLVER_DIR} -> ${ARCHIVE_DIR}"

# Preserve the exact small configuration inputs used by this run.  The solver
# copies remain in s/, while these copies travel with the generated data.
copy_config "${CASE_NAME}.rea"
copy_config "${CASE_NAME}.restart"
copy_config "runtime-config.yml"
copy_config "current_conf.yml"
copy_config "SESSION.NAME"
copy_config "restart-input.json"

# Same field families used by marl_energy_cost/utils/mv-data.
move_pattern "la2${CASE_NAME}0.f?????" "${LA2_DATA}" "LA2 fields"
move_pattern "pts${CASE_NAME}0.f?????" "${PTS_DATA}" "point fields"
move_pattern "sts${CASE_NAME}0.f?????" "${STS_DATA}" "statistics fields"
move_pattern "c2D${CASE_NAME}0.f?????" "${STS_DATA}" "C2D fields"
move_pattern "${CASE_NAME}0.f?????" "${REG_DATA}" "regular fields"

# Match marl_energy_cost: COPY restart fields and retain the live selector.
# Reflinks are independent on write; unsupported filesystems use real copies.
for source in "${SOLVER_DIR}"/rs[0-9]"${CASE_NAME}"0.f?????; do
    [[ -f "${source}" ]] || continue
    target="${RS_DATA}/$(basename "${source}")"
    cp --reflink=auto -p -- "${source}" "${target}.partial"
    mv -f -- "${target}.partial" "${target}"
done
cp -p -- "${SOLVER_DIR}/${CASE_NAME}.restart" "${RS_DATA}/"

move_pattern "*txt*" "${TXT_DATA}" "text diagnostics"
move_pattern "logfile*" "${TXT_DATA}" "Nek logs"
move_pattern "*.log" "${TXT_DATA}" "log files"
move_pattern "*.npz" "${DRL_DATA}" "NumPy records"
move_pattern "*vars_record*" "${DRL_DATA}" "DRL variable records"
move_pattern "reward_monitor*.dat" "${MONITOR_DATA}" "reward monitors"
move_directory "${SOLVER_DIR}/drlrec" "${REC_DATA}" "embedded DRL records"
move_directory "${SOLVER_DIR}/history" "${DRL_DATA}" "HydroGym history"

# Policy inputs, if a future executable writes them, are small provenance
# files.  Copy them while leaving the solver directory self-describing.
for policy_file in \
    "${SOLVER_DIR}"/actor_*.pol \
    "${SOLVER_DIR}/drl_policy.in" \
    "${SOLVER_DIR}/drl_record.in"; do
    [[ -f "${policy_file}" ]] || continue
    cp -p -- "${policy_file}" "${POLICY_DATA}/"
done

# Catch new Nek output types without silently losing them.  Known staged input
# files remain in s/; every other top-level regular file is moved to misc_data.
while IFS= read -r -d '' source; do
    name="$(basename "${source}")"
    case "${name}" in
        nek5000|SIZE|SESSION.NAME|runtime-config.yml|current_conf.yml|forparam.i|restart-input.json|\
        rs[0-9]"${CASE_NAME}"0.f?????|\
        "${CASE_NAME}.rea"|"${CASE_NAME}.re2"|"${CASE_NAME}.map"|\
        "${CASE_NAME}.wall"|"${CASE_NAME}.restart"|\
        "mask_${CASE_NAME}0.f00002"|stat_pts.in|actor_*.pol|drl_policy.in|drl_record.in)
            continue ;;
    esac
    move_one "${source}" "${MISC_DATA}"
    echo "[ARCHIVE] unclassified output -> misc_data: ${name}"
done < <(find "${SOLVER_DIR}" -maxdepth 1 -type f -print0)

# Record the immutable input links separately from generated output data.
: > "${ARCHIVE_DIR}/input_symlinks.tsv"
while IFS= read -r -d '' link; do
    printf '%s\t%s\n' "$(basename "${link}")" "$(readlink "${link}")" \
        >> "${ARCHIVE_DIR}/input_symlinks.tsv"
done < <(find "${SOLVER_DIR}" -maxdepth 1 -type l -print0 | sort -z)

finished="$(date --iso-8601=seconds)"
printf '{\n  "archived_at": "%s",\n  "result": "%s",\n  "run_root": "%s",\n  "solver_exit_code": %s\n}\n' \
    "${finished}" "${result_name}" "${RUN_ROOT}" "${SOLVER_EXIT_CODE}" \
    > "${ARCHIVE_DIR}/archive_status.json"
find "${ARCHIVE_DIR}" -type f \
    ! -name 'archive_manifest.tsv' \
    ! -name 'ARCHIVE_IN_PROGRESS' \
    ! -name 'ARCHIVE_COMPLETE' \
    -printf '%P\t%s\n' | sort > "${ARCHIVE_DIR}/archive_manifest.tsv"
printf 'archived_at=%s\nsolver_exit_code=%s\n' \
    "${finished}" "${SOLVER_EXIT_CODE}" > "${ARCHIVE_DIR}/ARCHIVE_COMPLETE"
rm -f -- "${ARCHIVE_DIR}/ARCHIVE_IN_PROGRESS"

echo "[ARCHIVE] complete: ${ARCHIVE_DIR}"
