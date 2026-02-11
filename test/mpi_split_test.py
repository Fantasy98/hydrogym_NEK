#!/usr/bin/env python3
from mpi4py import MPI


def main() -> None:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    # Require at least 2 processes for a meaningful split test
    if size < 2:
        if rank == 0:
            print(
                f"ERROR: This test requires at least 2 MPI processes, but only {size} process(es) detected.\n"
                f"Run with: mpirun -n 4 python3 {__file__}",
                flush=True,
            )
        return

    color = rank % 2
    split_comm = comm.Split(color=color, key=rank)

    split_rank = split_comm.Get_rank()
    split_size = split_comm.Get_size()

    # Validate membership in each split communicator.
    local_ranks = split_comm.gather(rank, root=0)
    if split_rank == 0:
        local_ranks = sorted(local_ranks)
        expected = list(range(color, size, 2))
        assert local_ranks == expected, (
            f"color {color}: expected ranks {expected}, got {local_ranks}"
        )

    # Validate split sizes for each color without empty slices.
    all_colors = comm.gather(color, root=0)
    all_split_sizes = comm.gather(split_size, root=0)
    if rank == 0:
        expected_sizes = {c: len([1 for r in range(size) if r % 2 == c]) for c in (0, 1)}
        actual_sizes = {0: set(), 1: set()}
        for c, s in zip(all_colors, all_split_sizes):
            actual_sizes[c].add(s)
        actual_sizes = {c: max(actual_sizes[c]) if actual_sizes[c] else 0 for c in (0, 1)}
        assert actual_sizes == expected_sizes, (
            f"expected split sizes {expected_sizes}, got {actual_sizes}"
        )

    comm.Barrier()
    print(
        f"world_rank={rank} world_size={size} "
        f"color={color} split_rank={split_rank} split_size={split_size}",
        flush=True,
    )

    split_comm.Free()


if __name__ == "__main__":
    main()
