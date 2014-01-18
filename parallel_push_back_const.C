#define ALPS_INDEP_SOURCE
#ifndef ALPS_ENABLE_TIMER
# define ALPS_ENABLE_TIMER
#endif

#include <boost/lexical_cast.hpp>
#include "timer_mpi.hpp"
#include "parallel.h"

#include <sys/time.h> /* gettimeofday */

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include <sstream>
#include <fstream>

#include <boost/random.hpp>
#include <boost/shared_ptr.hpp>

#include <mpi.h>
#include <omp.h>

struct myclock_t {
  static double get_time() {
    struct timeval tm;
    gettimeofday(&tm, NULL);
    return tm.tv_sec + tm.tv_usec * 1.0e-6;
  }
};

#if defined(ALPS_ENABLE_FAPP) || defined(ALPS_ENABLE_FAPP_PA)
template<>
char alps::parapack::detail::timer_base<alps::parapack::detail::clock_mpi>::msgs[512][256]
 = {"dummy string for allocation"};
#endif

int main(int argc, char* argv[]) {  
	int provided;
  MPI_Init(&argc, &argv);
  int num_processes, process_id;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

  alps::parapack::timer_mpi timer(MPI_COMM_WORLD);
  // timer.registrate(1, 1, "fill_times", timer.detailed | timer.start_barrier);
  timer.registrate(1, 1, "fill_times", timer.detailed);

  const int num_threads = omp_get_max_threads();  
    
  std::string filename;
  std::stringstream spid;
  spid << process_id;
  filename = "output_" + spid.str();
  
  
  std::vector<std::vector<double> > times_g(num_threads);
  for (int mcs = 0; mcs < 255; ++mcs) {
    // std::ofstream ofs(filename.c_str(), std::ios::app);    
    timer.start(1);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::vector<double>& times = times_g[tid];
      times.reserve(30000000);
      
      times.resize(0);
      double t0;
      for (int i = 0; i < 10000000; ++i) {
        t0 = 0.1;
        times.push_back(t0);
      }
    }
    timer.stop(1);

    timer.detailed_report();
  }

  timer.summarize();  
  MPI_Finalize();
  return 0;
}
