/*****************************************************************************
*
* ALPS/looper: multi-cluster quantum Monte Carlo algorithms for spin systems
*
* Copyright (C) 1997-2012 by Synge Todo <wistaria@comp-phys.org>,
*                            Haruhiko Matsuo <halm@looper.t.u-tokyo.ac.jp>
*
* This software is published under the ALPS Application License; you
* can use, redistribute it and/or modify it under the terms of the
* license, either version 1 or (at your option) any later version.
* 
* You should have received a copy of the ALPS Application License
* along with this software; see the file LICENSE. If not, the license
* is also available from http://alps.comp-phys.org/.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
* FITNESS FOR A PARTICULAR PURPOSE, TITLE AND NON-INFRINGEMENT. IN NO EVENT 
* SHALL THE COPYRIGHT HOLDERS OR ANYONE DISTRIBUTING THE SOFTWARE BE LIABLE 
* FOR ANY DAMAGES OR OTHER LIABILITY, WHETHER IN CONTRACT, TORT OR OTHERWISE, 
* ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER 
* DEALINGS IN THE SOFTWARE.
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
// #define ALPS_ENABLE_TIMER_TRACE
// #define ALPS_ENABLE_TIMER_DETAILED

#include "atomic_impl.h"
#include "capacity.h"
#include "chain_lattice.h"
#include "expand.h"
#include "observable.h"
#include "operator.h"
#include "options.h"
#include "union_find.h"
#include "timer.hpp"

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

#if defined(ALPS_ENABLE_FAPP) || defined(ALPS_ENABLE_FAPP_PA)
template<>
char alps::parapack::detail::timer_base<alps::parapack::detail::clock>::msgs[512][256]
 = {"dummy string for allocation"};
#endif

int main(int argc, char* argv[]) {
  alps::parapack::timer timer;
  timer.registrate( 1, 4, "main");
  timer.registrate( 2, 3, "_Monte_Carlo_steps", timer.detailed);
  timer.registrate( 3, 1, "__initialize_operators");
  timer.registrate( 4, 1,"__fill_times", timer.detailed);
  timer.registrate( 5, 1, "__initialize_cluster_infomation");
  timer.registrate( 6, 1, "__for_insert_or_remove_operators", timer.detailed);
  timer.registrate( 7, 1, "__for_fragments_current_unify");
  timer.registrate( 8, 1, "__for_fragments_unify");
  timer.registrate(10, 1, "__for_assign_cluster_id", timer.detailed);
  timer.registrate(11, 1, "__for_accumulate_estimates", timer.detailed);
  timer.registrate(12, 1, "__for_accumulate_coll", timer.detailed);
  timer.registrate(13, 1, "__cluster_flip_or_not", timer.detailed);
  timer.registrate(15, 1, "__flip_operator_and_spins", timer.detailed);
  timer.registrate(16, 2, "__Monte_Carlo_steps_whole-tid_3_13", timer.detailed);

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
  std::vector<estimate_un_t> estimates;
  std::vector<collector_un_t> coll_g(num_threads);
#else
  int num_fragments;
  int fragment_offset;
  std::vector<estimate_un_t> estimates;
  collector_un_t coll;
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

  // vector reservation
  if (p.reserve_times || p.reserve_operators || p.reserve_estimates || p.reserve_fragments) {
    for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
      times_g[tid].reserve(p.reserve_times);
      operators_g[tid].reserve(p.reserve_operators);
      operators_pg[tid].reserve(p.reserve_operators);
    }
    estimates.reserve(p.reserve_estimates);
    fragments.reserve(p.reserve_fragments);
    looper::vector_capacity capacity(times_g, operators_g, operators_pg, estimates, fragments);
    capacity.report();
  }

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
    #pragma omp parallel
    #endif
    {
      #ifdef LOOPER_OPENMP
      collector_un_t coll;
      #else
      coll = collector_un_t();
      #endif
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) {
        coll.umag_0 += 1 - 2 * spins[s];
        coll.smag_0 += lattice.phase(s) * (1 - 2 * spins[s]);
      }
      #ifdef LOOPER_OPENMP
      coll_g[omp_get_thread_num()] = coll;
      #endif
    }

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
      collector_un_t coll = coll_g[tid];
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
        double t = oi->time;
        int s0 = lattice.source(oi->bond);
        int s1 = lattice.target(oi->bond);
        if (oi->type == offdiagonal) {
          coll.smag_a += 2 * t * lattice.phase(s0) * (1 - 2 * spins[s0]);
          coll.smag_a += 2 * t * lattice.phase(s1) * (1 - 2 * spins[s1]);
          spins[s0] ^= 1;
          spins[s1] ^= 1;
        }
        fragments[fid] = fragment_t();
        oi->lower_cluster = unify(fragments, current[s0], current[s1]);
        oi->upper_cluster = current[s0] = current[s1] = fid++;
      }
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) {
        coll.smag_a += lattice.phase(s) * (1 - 2 * spins[s]);
      }
      num_fragments = fid - fragment_offset;
      #ifdef LOOPER_OPENMP
      coll_g[tid] = coll;
      #endif
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
    looper::expand(estimates, nc);
    timer.stop(11);

    timer.start(12);
#ifdef LOOPER_OPENMP
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      coll_g[tid].set_num_operators(operators_g[tid].size());
    }
    collector_un_t& coll = coll_g[0];
    for (int r = 1; r < num_threads; ++r) coll += coll_g[r];
#else
    coll.set_num_operators(operators.size());
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
      usus << 0.25 * beta * coll.umag_0 * coll.umag_0 / lattice.num_sites();
      smag << 0.25 * coll.smag_0 * coll.smag_0;
      ssus << 0.25 * beta * coll.smag_a * coll.smag_a / lattice.num_sites();
    }
    timer.stop(2);
    timer.detailed_report(p.dtime_interval);
  }

  // check vector capcity
  if (p.reserve_times || p.reserve_operators || p.reserve_estimates || p.reserve_fragments) {
    looper::vector_capacity capacity(times_g, operators_g, operators_pg, estimates, fragments);
    capacity.report();
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
