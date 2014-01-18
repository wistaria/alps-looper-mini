#ifndef LOOPER_AFFINITY_H
#define LOOPER_AFFINITY_H

#include <map>
#include <omp.h>
#include <sys/types.h>   // cpu_set_t
#include <sched.h>       // sched_setaffinity

// EXPERIMENTAL
// Use in a parallel region
namespace looper {
  bool set_affinity(int rank, int mpi_pocs_per_node, int omp_num_threads, int thread_id = 0) {
    const int current_process = 0; // execute sched_setaafinity() in the current process/thread.
    const int cpu_map = (rank % mpi_pocs_per_node) * omp_num_threads + thread_id;
    cpu_set_t cpu_flag;

    CPU_ZERO(&cpu_flag);
    CPU_SET(cpu_map, &cpu_flag);

    return sched_setaffinity(current_process, sizeof(cpu_set_t), &cpu_flag);
  }

  bool set_custom_affinity(int rank, int mpi_procs_per_node, int omp_num_threads,
			   std::map<int, int> cpu_map, int thread_id = 0,
			   int num_logical_cores = 1 /* per physical core */) {
    const int current_thread = 0;
    const int id_off_set = (rank % mpi_procs_per_node) * omp_num_threads * num_logical_cores;
    cpu_set_t cpu_flag;

    CPU_ZERO(&cpu_flag);
    CPU_SET(cpu_map[rank + thread_id] + id_off_set, &cpu_flag);

    return sched_setaffinity(current_thread, sizeof(cpu_set_t), &cpu_flag);
  }
}

#endif // LOOPER_AFFINITY_H
