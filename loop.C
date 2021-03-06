/*****************************************************************************
*
* ALPS/looper mini: MiniAppli for multi-cluster quantum Monte Carlo algorithms
*
* Copyright (C) 1997-2014 by Synge Todo <wistaria@comp-phys.org>,
*                            Haruhiko Matsuo <halm@looper.t.u-tokyo.ac.jp>
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*
*****************************************************************************/

// Spin-1/2 Antiferromagnetic Heisenberg Chain
// [continuous time path integral; using std::vector<> for operator string]
// OpenMP parallelization (real-space direction for OpenMP)

#if defined(LOOPER_ENABLE_OPENMP) && defined(_OPENMP) && !defined(LOOPER_OPENMP)
# define LOOPER_OPENMP
#endif

#define ALPS_INDEP_SOURCE
// #define ALPS_ENABLE_TIMER

#include "atomic_impl.h"
#include "chain_lattice.h"
#include "expand.h"
#include "observable.h"
#include "operator.h"
#include "options.h"
#include "union_find.h"
#include "timer_maprof.h"

#include <boost/random.hpp>
#include <boost/timer.hpp>

#include <algorithm>
#include <cstdlib>
#include <iostream>
#include <vector>

#ifdef LOOPER_OPENMP
# include "lattice_sharing.h"
# include "subaccumulate.h"
# include <boost/shared_ptr.hpp>
#endif

int main(int argc, char* argv[]) {
  alps::parapack::timer timer;
  timer.registrate( 1, 4, "main");
  timer.registrate( 2, 3, "_Monte_Carlo_steps", timer.detailed);
  timer.registrate( 3, 1, "__initialize_operators");
  timer.registrate( 4, 1, "__fill_times", timer.detailed);
  timer.registrate( 5, 1, "__initialize_cluster_infomation");
  timer.registrate( 6, 1, "__for_insert_or_remove_operators", timer.detailed);
  timer.registrate( 7, 1, "__for_fragments_current_unify");
  timer.registrate( 8, 1, "__for_fragments_unify");
  timer.registrate(10, 1, "__for_assign_cluster_id", timer.detailed);
  timer.registrate(11, 1, "__for_accumulate_estimates", timer.detailed);
  timer.registrate(12, 1, "__for_accumulate_coll", timer.detailed);
  timer.registrate(13, 1, "__cluster_flip_or_not", timer.detailed);
  timer.registrate(15, 1, "__flip_operator_and_spins", timer.detailed);
  timer.registrate(16, 2, "__Monte_Carlo_steps_whole_tid_3_13", timer.detailed);

  timer.start(1);

  // parameters
  options p(argc, argv, 8, 0.2);
  if (!p.valid) std::exit(-1);
  unsigned int sweeps = p.sweeps;
  unsigned int therm = p.therm;
  double beta = 1 / p.temperature;
#ifdef LOOPER_OPENMP
  // set default number of threads to 1
  if (getenv("OMP_NUM_THREADS") == 0) omp_set_num_threads(1);
  const int num_threads = omp_get_max_threads();
#else
  const int num_threads = 1;
#endif
  std::cout << "Number of Threads         = " << num_threads << std::endl;
  timer.set_parameter("L", p.length);
  timer.set_parameter("beta", p.beta);
  timer.set_parameter("sweeps", p.sweeps);
  timer.set_parameter("therm", p.therm);
  timer.set_parameter("total_sweeps", p.therm + p.sweeps);
  timer.set_repstdout(p.repstdout);

  // lattice
  chain_lattice lattice(p.length);

  // random number generators
  typedef boost::mt19937 engine_t;
  typedef boost::uniform_01<engine_t&> random01_t;
  typedef boost::exponential_distribution<> expdist_t;
#ifdef LOOPER_OPENMP
  std::vector<boost::shared_ptr<engine_t> > engine_g(num_threads);
  std::vector<boost::shared_ptr<random01_t> > random01_g(num_threads);
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    unsigned int seed = 29833 ^ (tid << 11);
    engine_g[tid].reset(new engine_t(seed));
    random01_g[tid].reset(new random01_t(*engine_g[tid]));
  }
#else
  engine_t engine(29833u);
  random01_t random01(engine);
#endif

  // vector of operators
#ifdef LOOPER_OPENMP
  std::vector<std::vector<local_operator_t> > operators_g(num_threads), operators_pg(num_threads);
  std::vector<std::vector<double> > times_g(num_threads);
#else
  std::vector<local_operator_t> operators, operators_p;
  std::vector<double> times;
#endif

  // spin configuration at t = 0 (1 for down and 0 for up)
  std::vector<int> spins(lattice.num_sites(), 0 /* all up */);
  std::vector<int> current(lattice.num_sites());
#ifdef LOOPER_OPENMP
  // current time of each thread
  double current_times[16][16];   // only support 1-16 threads.
  if(omp_get_max_threads() > 16) {
      std::cerr << "Error: Over 16 threads is not supported!!!\n";
      std::exit(-1);
  }
#endif

  // cluster information
  typedef looper::union_find::node fragment_t;
  std::vector<fragment_t> fragments;
#ifdef LOOPER_OPENMP
  std::vector<int> num_fragments_g(num_threads);
  std::vector<int> fragment_offset_g(num_threads);
  std::vector<std::vector<estimate_t> > estimates_g(num_threads);
  std::vector<collector_t> coll_g(num_threads);
#else
  int num_fragments;
  int fragment_offset;
  std::vector<estimate_t> estimates;
  collector_t coll;
#endif

  int nc;
#ifdef LOOPER_OPENMP
  std::vector<int> nc_g(num_threads);
#endif

  // oservables
  observable num_clusters;
  observable energy;
  observable usus; // uniform susceptibility
  observable smag; // staggered magnetizetion^2
  observable ssus; // staggered susceptibility

#ifdef LOOPER_OPENMP
  lattice_sharing sharing(lattice);
#endif

  //
  // Monte Carlo steps
  //

  boost::timer tm;

  for (unsigned int mcs = 0; mcs < therm + sweeps; ++mcs) {
    if(mcs <= therm) timer.clear();
    timer.start(2);

    //
    // diagonal update and cluster construction
    //

    // initialize operator information
    timer.start(16);
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
      operators_p.push_back(local_operator_t(0, 1));
    }
    timer.stop(3);

    // fill times
    timer.start(4);
    #ifdef LOOPER_OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef LOOPER_OPENMP
      int tid = omp_get_thread_num();
      std::vector<double>& times = times_g[tid];
      random01_t& random01 = *(random01_g[tid]);
      expdist_t expdist(beta * sharing.num_bonds_local(tid) / 2);
      #else
      expdist_t expdist(beta * lattice.num_bonds() / 2);
      #endif
      times.resize(0);
      double t = 0;
      while (t < 1) {
        t += expdist(random01);
        times.push_back(t);
      } // a sentinel (t >= 1) will be appended
    }
    timer.stop(4);

    // initialize cluster information
    timer.start(5);
    int n = 0;
    int bottom_offset = n; n += lattice.num_sites();
#ifdef LOOPER_OPENMP
    for (int r = 0; r < num_threads; ++r) {
      fragment_offset_g[r] = n; n += operators_pg[r].size() + times_g[r].size();
    }
#else
    fragment_offset = n; n += operators_p.size() + times.size();
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
          if (spins[lattice.source(b)] != spins[lattice.target(b)]) {
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
          spins[s0] ^= 1;
          spins[s1] ^= 1;
        }
        fragments[fid] = fragment_t();
        oi->lower_cluster = unify(fragments, current[s0], current[s1]);
        oi->upper_cluster = current[s0] = current[s1] = fid++;
      }
      num_fragments = fid - fragment_offset;
    }
    timer.stop(6);

    timer.start(7);
    #ifdef LOOPER_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int s = 0; s < lattice.num_sites(); ++s) {
      unify(fragments, s, current[s]);
    }
    timer.stop(7);

    //
    // cluster flip
    //

    // assign cluster id
    timer.start(10);
#ifdef LOOPER_OPENMP
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int ncl;
      nc_g[tid] = count_root_p(fragments, 0, lattice.num_sites()) +
        count_root(fragments, fragment_offset_g[tid], num_fragments_g[tid]);
      #pragma omp barrier
      ncl = looper::subaccumulate(nc_g, tid);
      ncl = set_id_p(fragments, 0, lattice.num_sites(), ncl);
      ncl = set_id(fragments, fragment_offset_g[tid], num_fragments_g[tid], ncl);
      if (tid + 1 == num_threads) nc = ncl;
      #pragma omp barrier
      copy_id_p(fragments, 0, lattice.num_sites());
      copy_id(fragments, fragment_offset_g[tid], num_fragments_g[tid]);
    }
#else
    nc = set_id(fragments, 0, fragment_offset + num_fragments, 0);
    copy_id(fragments, 0, fragment_offset + num_fragments);
#endif
    timer.stop(10);

    // accumulate physical property of clusters
    timer.start(11);
    #ifdef LOOPER_OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef LOOPER_OPENMP
      int tid = omp_get_thread_num();
      std::vector<local_operator_t>& operators = operators_g[tid];
      std::vector<estimate_t>& estimates = estimates_g[tid];
      #endif
      looper::expand(estimates, nc);
      estimate_t estimate_init;
      estimate_init = estimate_t();
      for (int c = 0; c < nc; ++c) estimates[c] = estimate_init;
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) {
        int id = fragments[s].id();
        estimates[id].mag += 1 - 2 * spins[s];
        estimates[id].size += 1;
        estimates[id].length += 1;
      }
      for (std::vector<local_operator_t>::iterator opi = operators.begin();
           opi != operators.end(); ++opi) {
        double t = opi->time;
        estimates[fragments[opi->lower_cluster].id()].length += 2 * t;
        estimates[fragments[opi->upper_cluster].id()].length -= 2 * t;
      }
    }
    timer.stop(11);

    timer.start(12);
#ifdef LOOPER_OPENMP
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      std::vector<estimate_t>& estimates = estimates_g[0];
      collector_t coll = coll_g[tid];
      coll.set_num_operators(operators_g[tid].size());
      #pragma omp for schedule(static)
      for (int c = 0; c < nc; ++c) {
        for (int r = 1; r < num_threads; ++r) estimates[c] += estimates_g[r][c];
        coll += estimates[c];
      }
      coll_g[tid] = coll;
    }
    collector_t& coll = coll_g[0];
    for (int r = 1; r < num_threads; ++r) coll += coll_g[r];
#else
    coll.set_num_operators(operators.size());
    for (int c = 0; c < nc; ++c) coll += estimates[c];
#endif
    coll.set_num_clusters(nc);
    timer.stop(12);

    // determine whether clusters are flipped or not
    timer.start(13);
    #ifdef LOOPER_OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef LOOPER_OPENMP
      std::vector<estimate_t>& estimates = estimates_g[0];
      random01_t& random01 = *(random01_g[omp_get_thread_num()]);
      #pragma omp for schedule(static)
      #endif
      for (int c = 0; c < nc; ++c) estimates[c].to_flip = (random01() < 0.5);
    }
    timer.stop(13);
    timer.stop(16);

    // flip operators & spins
    timer.start(15);
    #ifdef LOOPER_OPENMP
    #pragma omp parallel
    #endif
    {
      #ifdef LOOPER_OPENMP
      int tid = omp_get_thread_num();
      std::vector<local_operator_t>& operators = operators_g[tid];
      std::vector<estimate_t>& estimates = estimates_g[0];
      #endif
      for (std::vector<local_operator_t>::iterator opi = operators.begin();
           opi != operators.end(); ++opi) {
        if ((estimates[fragments[opi->lower_cluster].id()].to_flip ^
             estimates[fragments[opi->upper_cluster].id()].to_flip) & 1) opi->flip();
      }
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) {
        if (estimates[fragments[bottom_offset + s].id()].to_flip & 1) spins[s] ^= 1;
      }
    }
    timer.stop(15);

    //
    // measurements
    //

    if (mcs >= therm) {
      num_clusters << coll.num_clusters();
      energy << (0.25 * lattice.num_bonds() - coll.num_operators() / beta) / lattice.num_sites();
      usus << 0.25 * beta * coll.usus / lattice.num_sites();
      smag << 0.25 * coll.smag;
      ssus << 0.25 * beta * coll.ssus / lattice.num_sites();
    }
    timer.stop(2);
  }

  double elapsed = tm.elapsed() / num_threads;
  std::clog << "Elapsed time = " << elapsed << " sec\n"
            << "Speed = " << (therm + sweeps) / elapsed << " MCS/sec\n";
  std::cout << "Number of Clusters        = "
            << num_clusters.mean() << " +- " << num_clusters.error() << std::endl
            << "Energy Density            = "
            << energy.mean() << " +- " << energy.error() << std::endl
            << "Uniform Susceptibility    = "
            << usus.mean() << " +- " << usus.error() << std::endl
            << "Staggered Magnetization^2 = "
            << smag.mean() << " +- " << smag.error() << std::endl
            << "Staggered Susceptibility  = "
            << ssus.mean() << " +- " << ssus.error() << std::endl;

  timer.stop(1);
  timer.summarize();
}
