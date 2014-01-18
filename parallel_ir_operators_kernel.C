#if defined(LOOPER_ENABLE_OPENMP) && defined(_OPENMP) && !defined(LOOPER_OPENMP)
# define LOOPER_OPENMP
#endif

#define ALPS_INDEP_SOURCE
#ifndef ALPS_ENABLE_TIMER
# define ALPS_ENABLE_TIMER
#endif

#include "atomic_impl.h"
#include "chain_lattice.h"
#include "expand.h"
#include "observable.h"
#include "operator.h"
#include "options.h"
#include "parallel.h"
#include "union_find.h"
#include "timer_mpi.hpp"

#include <boost/random.hpp>
#include <boost/timer.hpp>
#include <boost/assign.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef LOOPER_OPENMP
# include "lattice_sharing.h"
# include "subaccumulate.h"
# include <boost/shared_ptr.hpp>
#endif

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
  timer.registrate(4, 1, "__fill_times", timer.detailed);
  timer.registrate(6, 1, "__for_insert_or_remove_operators", timer.detailed);

  // parameters
  options p(argc, argv, 8, 0.2, true, process_id == 0);
  if (!p.valid) { MPI_Finalize(); std::exit(-1); }
  unsigned int sweeps = p.sweeps;
  unsigned int therm = p.therm;
  double beta = 1 / p.temperature;
  double tau0 = 1. * process_id / num_processes;
  double tau1 = 1. * (process_id+1) / num_processes;
#ifdef LOOPER_OPENMP
  // set default number of threads to 1
  if (getenv("OMP_NUM_THREADS") == 0) omp_set_num_threads(1);
  const int num_threads = omp_get_max_threads();
#else
  const int num_threads = 1;
#endif
  if (process_id == 0) std::cout << "Number of Threads         = " << num_threads << std::endl;

  // lattice
  chain_lattice lattice(p.length);
  // square_lattice lattice(p.length);
  lattice_sharing sharing(lattice);
    
  std::string filename;
  std::stringstream spid;
  spid << process_id;
  filename = "output_" + spid.str();
  
  // random number generators
  typedef boost::mt19937 engine_t;
  typedef boost::uniform_01<engine_t&> random01_t;
  typedef boost::exponential_distribution<> expdist_t;
  std::vector<boost::shared_ptr<engine_t> > engine_g(num_threads);
  std::vector<boost::shared_ptr<random01_t> > random01_g(num_threads);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    unsigned int seed = 29833  ^ (113 * process_id) ^ (tid << 11);
    engine_g[tid].reset(new engine_t(seed));
    random01_g[tid].reset(new random01_t(*engine_g[tid]));
  }
  
  // vector of operators
  std::vector<std::vector<local_operator_t> > operators_g(num_threads), operators_pg(num_threads);
  std::vector<std::vector<double> > times_g(num_threads);

  // spin configuration at t = tau0 (1 for down and 0 for up)
  std::vector<int> spins(lattice.num_sites(), 0 /* all up */);
  std::vector<int> spins_c(lattice.num_sites());
  std::vector<int> current(lattice.num_sites());

  // current time of each thread
  double current_times[16][16];   // only support 1-16 threads.
  if(omp_get_max_threads() > 16) {
      std::cerr << "Error: Over 16 threads is not supported!!!\n";
      std::exit(-1);
  }

  // cluster information
  typedef looper::union_find::node fragment_t;
  std::vector<fragment_t> fragments;
  std::vector<int> num_fragments_g(num_threads);
  std::vector<int> fragment_offset_g(num_threads);
  std::vector<collector_t> coll_g(num_threads);

  // vector reserve
  for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
    int base_size = beta * (tau1 - tau0) * sharing.num_bonds_local(tid) / 2; 
    int times_reserve_size = base_size + 10 * sqrt(base_size);
    int operators_reserve_size = times_reserve_size;
    times_g[tid].reserve(times_reserve_size);
    operators_g[tid].reserve(operators_reserve_size);
    operators_pg[tid].reserve(operators_reserve_size);
  }
  int fragments_reserve_size = 3 * beta * (tau1 - tau0) * lattice.num_sites();
  fragments.reserve(fragments_reserve_size);

  // check capacity
  if(process_id == 0) {
    for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
      std::cout << "Info: times_g[" << tid << "].capacity      : " << times_g[tid].capacity() 
                << std::endl;
      std::cout << "Info: operators_g[" << tid << "].capacity  : " << operators_g[tid].capacity() 
                << std::endl;
      std::cout << "Info: operators_pg[" << tid << "].capacity : " << operators_pg[tid].capacity() 
                << std::endl;
    }
    std::cout << "Info: fragments.capacity       : " << fragments.capacity() 
              << std::endl;
  }

  for (unsigned int mcs = 0; mcs < therm + sweeps; ++mcs) {
    if(mcs <= therm) timer.clear();
    timer.start(2);

    //
    // diagonal update and cluster construction
    //

    // initialize operator information
    timer.start(3);
    #ifdef LOOPER_OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef LOOPER_OPENMP
      int tid = omp_get_thread_num();
      std::vector<local_operator_t>& operators = operators_g[tid];
      std::vector<local_operator_t>& operators_p = operators_pg[tid];
      current_times[tid][0] = 0;
      #endif
      std::swap(operators, operators_p); operators.resize(0);
      // insert a diagonal operator at the end of operators_p
      operators_p.push_back(local_operator_t(0, tau1));
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) spins_c[s] = spins[s];
    }
    timer.stop(3);

    // fill times
    timer.start(4);
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::vector<double>& times = times_g[tid];
      
      random01_t& random01 = *(random01_g[tid]);
      expdist_t expdist(beta * sharing.num_bonds_local(tid) / 2);
      times.resize(0);
      double t = tau0;
      while (t < tau1) {
        t += expdist(random01);
        times.push_back(t);
      } // a sentinel (t >= tau1) will be appended
    }
    timer.stop(4);

    // initialize cluster information
    timer.start(5);
    int n = 0;
#if LOOPER_FRAGMENT_ORDER == 0
    int bottom_offset = n; n += lattice.num_sites();
    int top_offset = n; n += lattice.num_sites();
#elif LOOPER_FRAGMENT_ORDER == 1
    int top_offset = n; n += lattice.num_sites();
    int bottom_offset = n; n += lattice.num_sites();
#elif LOOPER_FRAGMENT_ORDER == 2
    int bottom_offset = n; n += lattice.num_sites();
#endif
    for (int r = 0; r < num_threads; ++r) {
      fragment_offset_g[r] = n; n += operators_pg[r].size() + times_g[r].size();
    }

#if LOOPER_FRAGMENT_ORDER == 2
    int top_offset = n; n += lattice.num_sites();
#endif
    looper::expand(fragments, n);
    #ifdef LOOPER_OPENMP
    #pragma omp parallel
    #endif
    {
      fragment_t fragment_init;
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) {
        fragments[bottom_offset + s] = fragment_init;
        current[s] = bottom_offset + s;
        fragments[top_offset + s] = fragment_init;
      }
    }
    timer.stop(5);

    // physical quantities
    timer.start(6);
#ifdef LOOPER_OPENMP
    #pragma omp parallel for schedule(static)
    for (int r = 0; r < num_threads; ++r) {
      coll_g[r] = collector_t();
    }
#else
    coll = collector_t();
#endif

    // insert/remove operators
    #ifdef LOOPER_OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef LOOPER_OPENMP
      int tid = omp_get_thread_num();
      std::vector<local_operator_t>& operators = operators_g[tid];
      std::vector<local_operator_t>& operators_p = operators_pg[tid];
      std::vector<double>& times = times_g[tid];
      int& num_fragments = num_fragments_g[tid];
      int& fragment_offset = fragment_offset_g[tid];
      random01_t& random01 = *(random01_g[tid]);
      int bond_offset = sharing.bond_offset(tid);
      int num_bonds_local = sharing.num_bonds_local(tid);
      #else
      int bond_offset = 0;
      int num_bonds_local = lattice.num_bonds();
      #endif
      int fid = fragment_offset;
      std::vector<double>::iterator tmi = times.begin();
      for (std::vector<local_operator_t>::iterator opi = operators_p.begin();
           opi != operators_p.end();) {
        // diagonal update
        if (*tmi < opi->time) {
          #ifdef LOOPER_OPENMP
          current_times[tid][0] = *tmi;
          #endif
          int b = static_cast<int>(num_bonds_local * random01() + bond_offset);
          #ifdef LOOPER_OPENMP
          // wait for other threads
          int nid = sharing(b);
          if (nid != tid) {
            do {
              #pragma omp flush (current_times)
            } while (current_times[nid][0] < *tmi);
          }
          #endif
          if (spins_c[lattice.source(b)] != spins_c[lattice.target(b)]) {
            operators.push_back(local_operator_t(b, *tmi));
            ++tmi;
          } else {
            ++tmi;
            continue;
          }
        } else {
          #ifdef LOOPER_OPENMP
          current_times[tid][0] = opi->time;
          #endif
          if (opi->type == diagonal) {
            ++opi;
            continue;
          } else {
            operators.push_back(*opi);
            #ifdef LOOPER_OPENMP
            // wait for other threads
            int nid = sharing(opi->bond);
            if (nid != tid) {
              do {
                #pragma omp flush (current_times)
              } while (current_times[nid][0] < opi->time);
            }
            #endif
            ++opi;
          }
        }
        std::vector<local_operator_t>::iterator oi = operators.end() - 1;
        int s0 = lattice.source(oi->bond);
        int s1 = lattice.target(oi->bond);
        if (oi->type == offdiagonal) {
          spins_c[s0] ^= 1;
          spins_c[s1] ^= 1;
        }
        fragments[fid] = fragment_t();
        oi->lower_cluster = unify(fragments, current[s0], current[s1]);
        oi->upper_cluster = current[s0] = current[s1] = fid++;
      }
      num_fragments = fid - fragment_offset;
    }
    timer.stop(6);

    timer.detailed_report();
  }
  // check capacity
  if(process_id == 0) {
    for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
      std::cout << "Info: times_g[" << tid << "].capacity      : " << times_g[tid].capacity() 
                << std::endl;
      std::cout << "Info: operators_g[" << tid << "].capacity  : " << operators_g[tid].capacity() 
                << std::endl;
      std::cout << "Info: operators_pg[" << tid << "].capacity : " << operators_pg[tid].capacity() 
                << std::endl;
    }
    std::cout << "Info: fragments.capacity       : " << fragments.capacity() 
              << std::endl;

    for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
      std::cout << "Info: times_g[" << tid << "].size          : " << times_g[tid].size() 
                << std::endl;
      std::cout << "Info: operators_g[" << tid << "].size      : " << operators_g[tid].size() 
                << std::endl;
      std::cout << "Info: operators_pg[" << tid << "].size     : " << operators_pg[tid].size() 
                << std::endl;
    }
    std::cout << "Info: fragments.size           : " << fragments.size() 
              << std::endl;
  }

  timer.summarize();  
  MPI_Finalize();
  return 0;
}
