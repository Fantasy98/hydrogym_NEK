#!/usr/bin/env python3
"""Read CPU affinity of an existing job's Nek/controller processes on this node.

Run one probe per node through a short --overlap --mpi=none Slurm step.
No process affinity, environment, or solver files are modified.
"""

import argparse
from collections import Counter
from datetime import datetime, timezone
from functools import lru_cache
import json
import os
from pathlib import Path
import re
import socket


@lru_cache(None)
def topology(cpu):
    path = Path('/sys/devices/system/cpu/cpu{}'.format(cpu))
    return {
        'cpu': cpu,
        'socket': int((path / 'topology/physical_package_id').read_text()),
        'core': int((path / 'topology/core_id').read_text()),
        'numa': [int(p.name[4:]) for p in path.glob('node[0-9]*')],
    }


def memory_pages(pid):
    pages = Counter()
    policies = Counter()
    try:
        for line in (Path('/proc') / str(pid) / 'numa_maps').read_text().splitlines():
            if 'anon=' not in line:
                continue
            fields = line.split()
            policies[fields[1]] += 1
            for node, amount in re.findall(r'\bN(\d+)=(\d+)', line):
                pages[node] += int(amount)
        return {'anonymous_pages_by_numa': dict(pages), 'vma_policies': dict(policies)}
    except OSError as error:
        return {'error': str(error)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--job-id', required=True)
    args = parser.parse_args()
    rows, errors = [], []
    for proc in Path('/proc').iterdir():
        if not proc.name.isdigit():
            continue
        try:
            if proc.stat().st_uid != os.getuid():
                continue
            comm = (proc / 'comm').read_text().strip()
            if comm != 'nek5000' and not comm.startswith('python'):
                continue
            command = (proc / 'cmdline').read_bytes().replace(b'\0', b' ').decode(errors='replace')
            is_controller = '03_deploy.py' in command
            if comm != 'nek5000' and not is_controller:
                continue
            env = dict(item.split('=', 1) for item in (proc / 'environ').read_text().split('\0') if '=' in item)
            if env.get('SLURM_JOB_ID') != args.job_id:
                continue
            pid = int(proc.name)
            affinity = sorted(os.sched_getaffinity(pid))
            cpus = [topology(cpu) for cpu in affinity]
            thread_masks = Counter()
            for task in (proc / 'task').iterdir():
                try:
                    mask = ','.join(map(str, sorted(os.sched_getaffinity(int(task.name)))))
                    thread_masks[mask] += 1
                except ProcessLookupError:
                    pass
            rows.append({
                'pid': pid, 'rank': env.get('SLURM_PROCID'),
                'local_rank': env.get('SLURM_LOCALID'),
                'role': 'controller' if is_controller else 'solver',
                'cpus': affinity,
                'physical_cores': sorted(set((cpu['socket'], cpu['core']) for cpu in cpus)),
                'numa_nodes': sorted(set(node for cpu in cpus for node in cpu['numa'])),
                'thread_masks': dict(thread_masks),
                'settings': {key: env.get(key) for key in (
                    'OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS',
                    'OMPI_MCA_pml', 'UCX_TLS', 'UCX_NET_DEVICES',
                    'SLURM_CPU_BIND', 'SLURM_CPU_BIND_TYPE', 'SLURM_CPU_BIND_LIST',
                    'SLURM_DISTRIBUTION', 'SLURM_HINT', 'SLURM_THREADS_PER_CORE')},
            })
        except (OSError, ValueError) as error:
            errors.append({'pid': proc.name, 'error': str(error)})

    rows.sort(key=lambda row: int(row['rank'] or -1))
    sampled_solver = False
    core_users = {}
    for row in rows:
        for core in row['physical_cores']:
            core_users.setdefault(':'.join(map(str, core)), []).append(row['rank'])
        if row['role'] == 'controller' or not sampled_solver:
            row['memory_sample'] = memory_pages(row['pid'])
            if row['role'] == 'solver':
                sampled_solver = True
    result = {
        'host': socket.gethostname(), 'job_id': args.job_id,
        'captured_utc': datetime.now(timezone.utc).isoformat(),
        'process_count': len(rows),
        'role_counts': dict(Counter(row['role'] for row in rows)),
        'distinct_physical_cores': len(core_users),
        'shared_physical_cores': {core: ranks for core, ranks in core_users.items() if len(ranks) > 1},
        'processes': rows, 'errors': errors,
    }
    print(json.dumps(result), flush=True)


if __name__ == '__main__':
    main()
