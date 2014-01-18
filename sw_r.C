/*****************************************************************************
*
* ALPS/looper: multi-cluster quantum Monte Carlo algorithms for spin systems
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>
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

// Swendsen-Wang Algorithm for Square-Lattice Ising Model

#define ALPS_INDEP_SOURCE
// #define UNION_BY_WEIGHT
#define VISUALIZE

#include "atomic_impl.h"
#include "observable.h"
#include "options.h"
#include "square_lattice.h"

#ifdef UNION_BY_WEIGHT
# include "union_find.h"
#else
# include "union_find_r.h"
#endif // UNION_BY_WEIGHT

#include "visualize.h"

#include <boost/foreach.hpp>
#include <boost/random.hpp>
#include <boost/timer.hpp>

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

int main(int argc, char* argv[]) {
  // parameters
  options p(argc, argv, 8, 2.27);
  if (!p.valid) std::exit(-1);
  const unsigned int sweeps = p.sweeps;
  const unsigned int therm = p.therm;
  const double beta = 1 / p.temperature;
  const double prob = 1 - std::exp(- 2 * beta);

  // sqaure lattice
  square_lattice lattice(p.length);

  // random number generators
  typedef boost::mt19937 engine_t;
  typedef boost::uniform_01<engine_t&> random01_t;
  engine_t engine(29833u);
  random01_t random01(engine);

  // spin configuration
  std::vector<int> spins(lattice.num_sites());
  std::fill(spins.begin(), spins.end(), 0 /* all up */);

  // cluster information
#ifdef UNION_BY_WEIGHT
  typedef looper::union_find::node fragment_t;
#else
  typedef looper::union_find::node_rank fragment_t;
#endif // UNION_BY_WEIGHT

  std::vector<fragment_t> fragments(lattice.num_sites());
  std::vector<bool> to_flip(lattice.num_sites());

  // oservables
  observable num_clusters;
  int nc;

  //
  // Monte Carlo steps
  //

  boost::timer tm;

  for (unsigned int mcs = 0; mcs < therm + sweeps; ++mcs) {
    //
    // cluster construction
    //

    // initialize cluster information

    for (int s = 0; s < lattice.num_sites(); ++s) fragments[s] = fragment_t();

    for (int b = 0; b < lattice.num_bonds(); ++b) {
      if (spins[lattice.source(b)] == spins[lattice.target(b)] && random01() < prob) {
	unify(fragments, lattice.source(b), lattice.target(b));
      }
    }

    //
    // cluster flip
    //

    // assign cluster id & accumulate cluster properties
    nc = 0;
    for (int f = 0; f < fragments.size(); ++f) {
      if (fragments[f].is_root()) {
        fragments[f].set_id(nc++);
      }
    }
    for (int f = 0; f < fragments.size(); ++f) fragments[f].set_id(cluster_id(fragments, f));


    // determine whether clusters are flipped or not
    for (int c = 0; c < nc; ++c) to_flip[c] = (random01() < 0.5);

    // flip spins
    for (int s = 0; s < lattice.num_sites(); ++s) if (to_flip[fragments[s].id()]) spins[s] ^= 1;

    //
    // measurements
    //

    if (mcs >= therm) {
      num_clusters << nc;
    }

// #ifdef VISUALIZE
//     if (mcs == outputflag) {
//       graph_out(fragments);
//     }
// #endif // VISUALIZE

  }

  std::cout << "Speed = " << (therm + sweeps) / tm.elapsed()
            << " MCS/sec\n";
  std::cout << "Number of Clusters        = "
            << num_clusters.mean() << " +- " << num_clusters.error() << std::endl;
}
