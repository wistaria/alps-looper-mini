/*****************************************************************************
*
* ALPS/looper: multi-cluster quantum Monte Carlo algorithms for spin systems
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>
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

// Swendsen-Wang Algorithm for Square-Lattice Ising Model

#define ALPS_INDEP_SOURCE
// #define ALPS_ENABLE_TIMER
// #define ALPS_ENABLE_TIMER_TRACE
// #define ALPS_ENABLE_TIMER_DETAILED

// for debugging
// #define SERIALIZE_RANDOM_NUMBER_GENERATION

#include "atomic_impl.h"
#include "observable.h"
#include "options.h"
#include "square_lattice.h"
#include "union_find.h"
#include "timer.hpp"

#include <boost/foreach.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#if defined(LOOPER_OPENMP) && defined(_OPENMP)
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
  timer.registrate( 1, 3, "main");
  timer.registrate( 2, 2, "_Monte_Carlo_steps");
  timer.registrate( 3, 1, "__initialize_cluster_information");
  timer.registrate( 4, 1, "__random_number_generation_for_test");
  timer.registrate( 5, 1, "__union_find");
  timer.registrate( 6, 1, "__assign_cluster_id");
  timer.registrate( 7, 1, "__cluster_flip_or_not_for_test");
  timer.registrate( 8, 1, "__cluster_flip_or_not");
  timer.registrate( 9, 1, "__flip_spins");
  timer.registrate(10, 1, "__measurements");

  timer.start(1);

  // parameters
  options p(argc, argv, 8, 2.27);
  if (!p.valid) std::exit(-1);
  const unsigned int sweeps = p.sweeps;
  const unsigned int therm = p.therm;
  const double beta = 1 / p.temperature;
  const double prob = 1 - std::exp(- 2 * beta);
#if defined(LOOPER_OPENMP) && defined(_OPENMP)
  // set default number of threads to 1
  if (getenv("OMP_NUM_THREADS") == 0) omp_set_num_threads(1);
  const int num_threads = omp_get_max_threads();
#else
  const int num_threads = 1;
#endif
  std::cout << "Number of Threads         = " << num_threads << std::endl;

  // sqaure lattice
  square_lattice lattice(p.length);

  // random number generators
  typedef boost::mt19937 engine_t;
  typedef boost::uniform_01<engine_t&> random01_t;
#if defined(LOOPER_OPENMP) && defined(_OPENMP) && !defined(SERIALIZE_RANDOM_NUMBER_GENERATION)
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

  // spin configuration
  std::vector<int> spins(lattice.num_sites());
  std::fill(spins.begin(), spins.end(), 0 /* all up */);
#ifdef SERIALIZE_RANDOM_NUMBER_GENERATION
  std::vector<double> rn(std::max(lattice.num_sites(), lattice.num_bonds()));
#endif

  // cluster information
  typedef looper::union_find::node fragment_t;
  std::vector<fragment_t> fragments(lattice.num_sites());
  std::vector<bool> to_flip(lattice.num_sites());

  // oservables
  observable num_clusters;
  observable usus; // uniform susceptibility
  int nc;
#if defined(LOOPER_OPENMP) && defined(_OPENMP)
  std::vector<int> nc_g(num_threads);
#endif
  double mag2;

  //
  // Monte Carlo steps
  //

  boost::timer tm;

  for (unsigned int mcs = 0; mcs < therm + sweeps; ++mcs) {
    if(mcs <= therm) timer.clear();
    timer.start(2);

    //
    // cluster construction
    //

    // initialize cluster information
    timer.start(3);
    #if defined(LOOPER_OPENMP) && defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int s = 0; s < lattice.num_sites(); ++s) fragments[s] = fragment_t();
    timer.stop(3);

#ifdef SERIALIZE_RANDOM_NUMBER_GENERATION
    timer.start(4);
    for (int b = 0; b < lattice.num_bonds(); ++b) rn[b] = random01();
    timer.stop(4);
    timer.start(5);
    #if defined(LOOPER_OPENMP) && defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int b = 0; b < lattice.num_bonds(); ++b) {
      if (spins[lattice.source(b)] == spins[lattice.target(b)] && rn[b] < prob) {
        unify(fragments, lattice.source(b), lattice.target(b));
      }
    }
    timer.stop(5);
#else
    timer.start(5);
    #if defined(LOOPER_OPENMP) && defined(_OPENMP)
    #pragma omp parallel
    #endif
    {
      #if defined(LOOPER_OPENMP) && defined(_OPENMP)
      random01_t& random01 = *(random01_g[omp_get_thread_num()]);
      #pragma omp for schedule(static)
      #endif
      for (int b = 0; b < lattice.num_bonds(); ++b) {
        if (spins[lattice.source(b)] == spins[lattice.target(b)] && random01() < prob) {
          unify(fragments, lattice.source(b), lattice.target(b));
        }
      }
    }
    timer.stop(5);
#endif

    //
    // cluster flip
    //

    // assign cluster id & accumulate cluster properties
    timer.start(6);
    nc = 0;
    mag2 = 0;
#if defined(LOOPER_OPENMP) && defined(_OPENMP)
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int ncl;
      nc_g[tid] = count_root_p(fragments, 0, fragments.size());
      #pragma omp barrier
      ncl = looper::subaccumulate(nc_g, tid);
      #pragma omp for schedule(static) reduction(+:mag2)
      for (int f = 0; f < fragments.size(); ++f) {
        if (fragments[f].is_root()) {
          fragments[f].set_id(ncl++);
          mag2 += fragments[f].weight() * fragments[f].weight();
        }
      }
      if (tid == num_threads - 1) nc = ncl;
      #pragma omp for schedule(static)
      for (int f = 0; f < fragments.size(); ++f) fragments[f].set_id(cluster_id(fragments, f));
    }
#else
    for (int f = 0; f < fragments.size(); ++f) {
      if (fragments[f].is_root()) {
        fragments[f].set_id(nc++);
        mag2 += fragments[f].weight() * fragments[f].weight();
      }
    }
    for (int f = 0; f < fragments.size(); ++f) fragments[f].set_id(cluster_id(fragments, f));
#endif
    timer.stop(6);

    // determine whether clusters are flipped or not
#ifdef SERIALIZE_RANDOM_NUMBER_GENERATION
    // should be run in serial
    timer.start(7);
    for (int s = 0; s < lattice.num_sites(); ++s) rn[s] = random01();
    for (int c = 0; c < nc; ++c) to_flip[c] = false;
    for (int s = 0; s < lattice.num_sites(); ++s) {
      int c = cluster_id(fragments, s);
      to_flip[c] = to_flip[c] ^ (rn[s] < 0.5);
    }
    timer.stop(7);
#else
    timer.start(8);
    #if defined(LOOPER_OPENMP) && defined(_OPENMP)
    #pragma omp parallel
    #endif
    {
      #if defined(LOOPER_OPENMP) && defined(_OPENMP)
      random01_t& random01 = *(random01_g[omp_get_thread_num()]);
      #pragma omp for schedule(static)
      #endif
      for (int c = 0; c < nc; ++c) to_flip[c] = (random01() < 0.5);
    }
    timer.stop(8);
#endif

    // flip spins
    timer.start(9);
    #if defined(LOOPER_OPENMP) && defined(_OPENMP)
    #pragma omp parallel for schedule(static)
    #endif
    for (int s = 0; s < lattice.num_sites(); ++s) if (to_flip[fragments[s].id()]) spins[s] ^= 1;
    timer.stop(9);

    //
    // measurements
    //

    timer.start(10);
    if (mcs >= therm) {
      num_clusters << nc;
      usus << beta * mag2 / lattice.num_sites();
    }
    timer.stop(10);
    timer.stop(2);
    timer.detailed_report();
  }

  double elapsed = tm.elapsed() / num_threads;
  std::clog << "Elapsed time = " << elapsed << " sec\n"
            << "Speed = " << (therm + sweeps) / elapsed << " MCS/sec\n";
  std::cout << "Number of Clusters        = "
            << num_clusters.mean() << " +- " << num_clusters.error() << std::endl
            << "Uniform Susceptibility    = "
            << usus.mean() << " +- " << usus.error() << std::endl;
  timer.stop(1);
  timer.summarize();
}
