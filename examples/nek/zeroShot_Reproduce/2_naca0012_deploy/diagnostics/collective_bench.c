/* Standalone diagnostic: never opens HydroGym configuration or restart files. */
#define _GNU_SOURCE
#include <mpi.h>
#include <sched.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void check(int rc) {
    if (rc != MPI_SUCCESS) MPI_Abort(MPI_COMM_WORLD, rc);
}

static void measure(MPI_Comm comm, const char *profile, int pass,
                    int count, int probe) {
    int rank, size;
    MPI_Comm_rank(comm, &rank);
    MPI_Comm_size(comm, &size);
    double *send = calloc(count ? count : 1, sizeof(double));
    double *recv = calloc(count ? count : 1, sizeof(double));
    if (!send || !recv) MPI_Abort(MPI_COMM_WORLD, 2);
    for (int i = 0; i < count; ++i) send[i] = rank + 1;
    int iterations = probe ? 1 : (count >= 131072 ? 32 : 200);
    int batches = probe ? 1 : 3;
    int warmups = probe ? 0 : 5;
    for (int batch = 0; batch < batches; ++batch) {
        for (int i = 0; i < warmups; ++i) {
            if (count) check(MPI_Allreduce(send, recv, count, MPI_DOUBLE, MPI_SUM, comm));
            else check(MPI_Barrier(comm));
        }
        check(MPI_Barrier(comm));
        double start = MPI_Wtime();
        for (int i = 0; i < iterations; ++i) {
            if (count) check(MPI_Allreduce(send, recv, count, MPI_DOUBLE, MPI_SUM, comm));
            else check(MPI_Barrier(comm));
        }
        double elapsed = MPI_Wtime() - start, maximum;
        /* Validation and timing aggregation are outside the timed section. */
        double expected = (double)size * (size + 1) / 2;
        for (int i = 0; i < count; ++i) {
            if (recv[i] != expected) {
                fprintf(stderr, "Incorrect reduction on rank %d, element %d\n", rank, i);
                MPI_Abort(MPI_COMM_WORLD, 3);
            }
        }
        check(MPI_Reduce(&elapsed, &maximum, 1, MPI_DOUBLE, MPI_MAX, 0, comm));
        if (rank == 0 && !probe) {
            printf("RESULT,%s,%d,%s,%d,%d,%d,%d,%.9g\n", profile, pass,
                   count ? "allreduce_sum" : "barrier", count * (int)sizeof(double),
                   size, batch, iterations, maximum / iterations);
            fflush(stdout);
        }
    }
    free(send);
    free(recv);
}

int main(int argc, char **argv) {
    check(MPI_Init(&argc, &argv));
    int rank, world_size;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &world_size);
    if (argc != 5 || world_size < 3) {
        if (rank == 0) fprintf(stderr, "Usage: collective_bench PROFILE PASS probe|timing EXPECTED_WORLD_SIZE\n");
        MPI_Abort(MPI_COMM_WORLD, 4);
    }
    if (world_size != atoi(argv[4])) MPI_Abort(MPI_COMM_WORLD, 5);
    int probe = !strcmp(argv[3], "probe");
    MPI_Comm solver;
    /* Production has a Python controller at world rank zero. */
    check(MPI_Comm_split(MPI_COMM_WORLD, rank == 0 ? MPI_UNDEFINED : 1, rank, &solver));
    if (rank != 0) {
        check(MPI_Comm_set_name(solver, "WING_SOLVER_BENCH"));
        MPI_Comm local;
        check(MPI_Comm_split_type(solver, MPI_COMM_TYPE_SHARED, rank, MPI_INFO_NULL, &local));
        int local_size, minimum, maximum;
        MPI_Comm_size(local, &local_size);
        MPI_Allreduce(&local_size, &minimum, 1, MPI_INT, MPI_MIN, solver);
        MPI_Allreduce(&local_size, &maximum, 1, MPI_INT, MPI_MAX, solver);
        cpu_set_t mask;
        CPU_ZERO(&mask);
        int cores = sched_getaffinity(0, sizeof(mask), &mask) == 0 ? CPU_COUNT(&mask) : -1;
        int min_cores, max_cores;
        MPI_Allreduce(&cores, &min_cores, 1, MPI_INT, MPI_MIN, solver);
        MPI_Allreduce(&cores, &max_cores, 1, MPI_INT, MPI_MAX, solver);
        if (rank == 1) {
            printf("LAYOUT,%s,solver_ranks=%d,local_ranks_min=%d,local_ranks_max=%d,affinity_cpus_min=%d,affinity_cpus_max=%d\n",
                   argv[1], world_size - 1, minimum, maximum, min_cores, max_cores);
            fflush(stdout);
        }
        MPI_Comm_free(&local);
        /* 641536 doubles is the exact wing reward-reduction buffer length.
           Smaller sizes sample latency-sensitive solver reductions. */
        const int counts[] = {0, 1, 8, 64, 512, 2048, 4096, 131072, 641536};
        for (unsigned i = 0; i < sizeof(counts) / sizeof(counts[0]); ++i)
            measure(solver, argv[1], atoi(argv[2]), counts[i], probe);
        MPI_Comm_free(&solver);
        if (rank == 1) check(MPI_Send(NULL, 0, MPI_BYTE, 0, 17, MPI_COMM_WORLD));
    } else {
        check(MPI_Recv(NULL, 0, MPI_BYTE, 1, 17, MPI_COMM_WORLD, MPI_STATUS_IGNORE));
    }
    check(MPI_Finalize());
    return 0;
}
