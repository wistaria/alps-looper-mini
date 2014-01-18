/*****************************************************************************
*
* ALPS/looper: multi-cluster quantum Monte Carlo algorithms for spin systems
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>,
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
// Parallel Union-Find
#if defined(LOOPER_ENABLE_OPENMP) && defined(_OPENMP) && !defined(LOOPER_OPENMP)
# define LOOPER_OPENMP
#endif

#define ALPS_INDEP_SOURCE
// #define ALPS_ENABLE_TIMER
// #define ALPS_ENABLE_TIMER_TRACE
// #define ALPS_ENABLE_TIMER_DETAILED
// #define ALPS_ENABLE_TIMER_BARRIER

// #define LOOPER_FRAGMENT_ORDER xx
// 0: [bottom] -> [top] -> [thread 0] -> ... -> [thread N-1] (default)
// 1: [top] -> [bottom] -> [thread 0] -> ... -> [thread N-1]
// 2: [bottom] -> [thread 0] -> ... -> [thread N-1] -> [top] (overhead in pack_tree and chunk::set)
#ifndef LOOPER_FRAGMENT_ORDER
# define LOOPER_FRAGMENT_ORDER 0
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

template <typename FRAGMENT>
void fragment_output(int pid, FRAGMENT fragments, int bottom_offset, std::vector<int> global_id, int num_sites) {
  std::cout << "-------------------------" << std::endl;
  for (int i = 0; i < fragments.size(); ++i) {
    std::cout << "fout::PID: " << pid << " "
	      << "FID: " << i + bottom_offset << " "
	      << "CID: " << fragments[i + bottom_offset].id() << " "
	      << "parent: " << fragments[i + bottom_offset].parent() << " "
	      << "is_root: " << fragments[i + bottom_offset].is_root() << " ";
    if (i < num_sites) {
	    std::cout << "gid: " << global_id[fragments[bottom_offset + i].id()] << std::endl;
    } else {
	    std::cout << "gid --" << std::endl;
    }
  }
}

template <typename FRAGMENT>
void fragment_output(int pid, FRAGMENT fragments, int s, int t, int b, int bottom_offset, std::vector<int> global_id, int num_sites) {
  std::cout << "-------------------------" << std::endl;
  std::cout << "b = " << b << " s = " << s << " t = " << t << std::endl;
  for (int i = 0; i < fragments.size(); ++i) {
    std::cout << "fout::PID: " << pid << " "
	      << "FID: " << i + bottom_offset << " "
	      << "CID: " << fragments[bottom_offset + i].id() << " "
	      << "parent: " << fragments[bottom_offset + i].parent() << " "
	      << "is_root: " << fragments[bottom_offset + i].is_root() << " ";
    if (i < num_sites) {
	    std::cout << "gid: " << global_id[fragments[bottom_offset + i].id()] << std::endl;
    } else {
	    std::cout << "gid --" << std::endl;
    }
  }
}

#if defined(ALPS_ENABLE_FAPP) || defined(ALPS_ENABLE_FAPP_PA)
template<>
char alps::parapack::detail::timer_base<alps::parapack::detail::clock_mpi>::msgs[512][256]
 = {"dummy string for allocation"};
#endif

typedef looper::parallel_cluster_unifier<estimate_t, collector_t> unifier_t;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int num_processes, process_id;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

  alps::parapack::timer_mpi timer(MPI_COMM_WORLD);
  timer.registrate(1, 1, "main");
  unifier_t::init_timer(timer);

  timer.start(1);
  
  // parameters
  options p(argc, argv, 4, 0.2, true, process_id == 0);
  if (!p.valid) { MPI_Finalize(); std::exit(-1); }
  double beta = 1;
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

  std::vector<int> global_cid;

  // preset times
  typedef boost::tuple<double, int> times_t;
  std::vector<times_t> preset_times(7);
 preset_times[0] = boost::make_tuple(0.2, 0);  //  (time, bond)
 preset_times[1] = boost::make_tuple(0.3, 2); 
 preset_times[2] = boost::make_tuple(0.4, 1); 
 preset_times[3] = boost::make_tuple(0.55, 3); 
 preset_times[4] = boost::make_tuple(0.6, 1); 
 preset_times[5] = boost::make_tuple(0.7, 0); 
 preset_times[6] = boost::make_tuple(0.8, 3); 

  // vector of operators
#ifdef LOOPER_OPENMP
  //  std::vector<std::vector<local_operator_t> > operators_g(num_threads), operators_pg(num_threads);
  std::vector<std::vector<times_t> > times_g(num_threads);
#else
  //  std::vector<local_operator_t> operators, operators_p;
  std::vector<times_t> times;
#endif

  // spin configuration at t = tau0 (1 for down and 0 for up)
  std::vector<int> current(lattice.num_sites());
#ifdef LOOPER_OPENMP
  // current time of each thread
  double current_times[16][16];   // only support 1-16 threads.
  if(omp_get_max_threads() > 16) {
      std::cerr << "Error: Over 16 threads is not supported!!!\n";
      std::exit(-1);
  }
#endif

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

  int nc; // total number of (open and close) clusters
  int noc; // total number of open clusters
#ifdef LOOPER_OPENMP
  std::vector<int> noc_g(num_threads); // number of open clusters
  std::vector<int> ncc_g(num_threads); // number of close clusters
#endif

  // oservables
  observable num_clusters;
  observable energy;
  observable usus; // uniform susceptibility
  observable smag; // staggered magnetizetion^2
  observable ssus; // staggered susceptibility

  // vector capacity
  std::vector<int> capacity_times_s(num_threads);
  std::vector<int> capacity_times_e(num_threads);
  //  std::vector<int> capacity_operators_g_s(num_threads);
  //  std::vector<int> capacity_operators_g_e(num_threads);
  //  std::vector<int> capacity_operators_pg_s(num_threads);
  //  std::vector<int> capacity_operators_pg_e(num_threads);
  std::vector<int> capacity_estimates_g_s(num_threads);
  std::vector<int> capacity_estimates_g_e(num_threads);
  int capacity_fragments_s, capacity_fragments_e;

  // vector reserve
  for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
    times_g[tid].reserve(p.reserve_times);
    //    operators_g[tid].reserve(p.reserve_operators);
    //    operators_pg[tid].reserve(p.reserve_operators);
    estimates_g[tid].reserve(p.reserve_estimates);
  }
  fragments.reserve(p.reserve_fragments);

  // save vector capacity of start
  for (int tid = 0; tid < omp_get_max_threads(); ++tid) {
    capacity_times_s[tid]        = times_g[tid].capacity();
    //    capacity_operators_g_s[tid]  = operators_g[tid].capacity();
    //    capacity_operators_pg_s[tid] = operators_pg[tid].capacity();
    capacity_estimates_g_s[tid]  = estimates_g[tid].capacity();
  }
  capacity_fragments_s           = fragments.capacity();

  // cluster information
  unifier_t unifier(MPI_COMM_WORLD, timer, lattice.num_sites(), p.partition, p.duplex);

#ifdef LOOPER_OPENMP
  lattice_sharing sharing(lattice);
#endif

  //
  // Parallel Union-Find
  //

  MPI_Barrier(MPI_COMM_WORLD);
  boost::timer tm;

  // initialize operator information
  #ifdef LOOPER_OPENMP
  #pragma omp parallel
  #endif
  {
    #ifdef LOOPER_OPENMP
    int tid = omp_get_thread_num();
    current_times[tid][0] = 0;
    #endif
  }

  // fill times
  #ifdef LOOPER_OPENMP
  #pragma omp parallel
  #endif
  {
    int tid = omp_get_thread_num();
    std::vector<times_t>& times = times_g[tid];
    times.resize(0);
    for (int i = 0; i < preset_times.size(); ++i) {
      double t = preset_times[i].get<0>();
      int b = preset_times[i].get<1>();
      if (tau0 < t && t < tau1) {
	if (sharing.bond_offset(tid) <= b && b < sharing.bond_offset(tid) + sharing.num_bonds_local(tid)) {
	  times.push_back(preset_times[i]);
	}
      }
    } // a sentinel (t >= tau1) will be appended
  }

  // initialize cluster information
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
#ifdef LOOPER_OPENMP
  for (int r = 0; r < num_threads; ++r) {
    fragment_offset_g[r] = n; n += times_g[r].size();
  }
#else
  fragment_offset = n; n += times.size();
#endif
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

  // physical quantities
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
    
    std::vector<times_t>& times = times_g[tid];
    int& num_fragments = num_fragments_g[tid];
    int& fragment_offset = fragment_offset_g[tid];
    
    int bond_offset = sharing.bond_offset(tid);
    int num_bonds_local = sharing.num_bonds_local(tid);
    #else
    int bond_offset = 0;
    int num_bonds_local = lattice.num_bonds();
    #endif
    int fid = fragment_offset;
    std::vector<times_t>::iterator tmi = times.begin();

    for (std::vector<times_t>::iterator tmi = times.begin(); tmi != times.end(); ++tmi) {
      #ifdef LOOPER_OPENMP
      current_times[tid][0] = tmi->get<0>();
      #endif
      int b = tmi->get<1>();
	    
      #ifdef LOOPER_OPENMP
      // wait for other threads
      int nid = sharing(b);
      if (nid != tid) {
	do {
          #pragma omp flush (current_times)
	} while (current_times[nid][0] < tmi->get<0>());
      }
      #endif

      int s0 = lattice.source(b);
      int s1 = lattice.target(b);
	
      fragments[fid] = fragment_t();
      unify(fragments, current[s0], current[s1]);
      current[s0] = current[s1] = fid++;
    }
    num_fragments = fid - fragment_offset;
  }
  

  #ifdef LOOPER_OPENMP
  #pragma omp parallel for schedule(static)
  #endif
  for (int s = 0; s < lattice.num_sites(); ++s) {
    unify(fragments, current[s], top_offset + s);
  }

  if (num_processes == 1) {
    #ifdef LOOPER_OPENMP
    #pragma omp parallel for schedule(static)
    #endif
    for (int s = 0; s < lattice.num_sites(); ++s)
      unify(fragments, bottom_offset + s, top_offset + s);
  } else {
    #if LOOPER_FRAGMENT_ORDER == 0 || LOOPER_FRAGMENT_ORDER == 1
    pack_tree(fragments, 2 * lattice.num_sites());
    #elif LOOPER_FRAGMENT_ORDER == 2
    pack_tree(fragments, lattice.num_sites(), top_offset);
    #endif
  }

  // assign cluster id
  #ifdef LOOPER_OPENMP
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    int ncl;
    noc_g[tid] = count_root_p(fragments, bottom_offset, lattice.num_sites()) +
      count_root_p(fragments, top_offset, lattice.num_sites());
    ncc_g[tid] = count_root(fragments, fragment_offset_g[tid], num_fragments_g[tid]);
    #pragma omp barrier
    ncl = looper::subaccumulate(noc_g, tid);
    ncl = set_id_p(fragments, bottom_offset, lattice.num_sites(), ncl);
    ncl = set_id_p(fragments, top_offset, lattice.num_sites(), ncl);
    if (tid + 1 == num_threads) noc = ncl;
    #pragma omp barrier
    ncl = noc + looper::subaccumulate(ncc_g, tid);
    ncl = set_id(fragments, fragment_offset_g[tid], num_fragments_g[tid], ncl);
    if (tid + 1 == num_threads) nc = ncl;
    #pragma omp barrier
    copy_id_p(fragments, bottom_offset, lattice.num_sites());
    copy_id_p(fragments, top_offset, lattice.num_sites());
    copy_id(fragments, fragment_offset_g[tid], num_fragments_g[tid]);
  }
  if (num_processes == 1) noc = 0;
  #else
  nc = set_id(fragments, bottom_offset, lattice.num_sites(), 0);
  nc = set_id(fragments, top_offset, lattice.num_sites(), nc);
  noc = (num_processes == 1) ? 0 : nc;
  nc = set_id(fragments, fragment_offset, num_fragments, nc);
  copy_id(fragments, bottom_offset, lattice.num_sites());
  copy_id(fragments, top_offset, lattice.num_sites());
  copy_id(fragments, fragment_offset, num_fragments);
  #endif

  // accumulate physical property of clusters
  #ifdef LOOPER_OPENMP
  #pragma omp parallel
  #endif
  {
    #ifdef LOOPER_OPENMP
    int tid = omp_get_thread_num();
    //    std::vector<local_operator_t>& operators = operators_g[tid];
    std::vector<estimate_t>& estimates = estimates_g[tid];
    #endif
    looper::expand(estimates, nc);
    estimate_t estimate_init;
    estimate_init = estimate_t();
    for (int c = 0; c < nc; ++c) estimates[c] = estimate_init;
    if (process_id == 0) {
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) {
	int id = fragments[bottom_offset + s].id();
	estimates[id].mag += 1;
	estimates[id].size += 1;
	estimates[id].length -= tau0;
      }
    } else {
      #ifdef LOOPER_OPENMP
      #pragma omp for schedule(static)
      #endif
      for (int s = 0; s < lattice.num_sites(); ++s) {
	int id = fragments[bottom_offset + s].id();
	estimates[id].length -= tau0;
      }
    }

    #ifdef LOOPER_OPENMP
    #pragma omp for schedule(static)
    #endif
    for (int s = 0; s < lattice.num_sites(); ++s) {
      int id = fragments[top_offset + s].id();
      estimates[id].length += tau1;
    }
  }

  #ifdef LOOPER_OPENMP
  #pragma omp parallel
  {
    int tid = omp_get_thread_num();
    std::vector<estimate_t>& estimates = estimates_g[0];
    collector_t coll = coll_g[tid];
    coll.set_num_operators(times_g[tid].size());
    #pragma omp for schedule(static)
    for (int c = 0; c < noc; ++c) {
      for (int r = 1; r < num_threads; ++r) estimates[c] += estimates_g[r][c];
    }
    #pragma omp for schedule(static)
    for (int c = noc; c < nc; ++c) {
      for (int r = 1; r < num_threads; ++r) estimates[c] += estimates_g[r][c];
      coll += estimates[c];
    }
    coll_g[tid] = coll;
  }
  collector_t& coll = coll_g[0];
  for (int r = 1; r < num_threads; ++r) coll += coll_g[r];
  #else
  coll.set_num_operators(times.size());
  for (int c = noc; c < nc; ++c) coll += estimates[c];
  #endif
  coll.set_num_open_clusters(noc);
  coll.set_num_clusters(nc - noc);

  // global unification of open clusters
  if (num_processes > 1) {
    #ifdef LOOPER_OPENMP
    unifier.unify(coll, fragments, bottom_offset, top_offset, estimates_g, global_cid, timer);
    #else
    unifier.unify(coll, fragments, bottom_offset, top_offset, estimates, global_cid, timer);
    #endif
  }

  MPI_Barrier(MPI_COMM_WORLD);
  if (process_id == 0)  fragment_output(process_id, fragments, bottom_offset, global_cid, lattice.num_sites());
  MPI_Barrier(MPI_COMM_WORLD);
  if (process_id == 1)  fragment_output(process_id, fragments, bottom_offset, global_cid, lattice.num_sites());
  MPI_Barrier(MPI_COMM_WORLD);
  if (process_id == 2)  fragment_output(process_id, fragments, bottom_offset, global_cid, lattice.num_sites());
  MPI_Barrier(MPI_COMM_WORLD);
  if (process_id == 3)  fragment_output(process_id, fragments, bottom_offset, global_cid, lattice.num_sites());
  MPI_Barrier(MPI_COMM_WORLD);
  if (process_id == 0 )
    std::cerr << "Number of Clusters = " << coll.num_clusters();
  
  timer.stop(1);
  timer.summarize();
  MPI_Finalize();
}
