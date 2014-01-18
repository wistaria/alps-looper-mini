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

// Bond Percolation Problem on Square Lattice

#define ALPS_INDEP_SOURCE

#include "square_lattice.h"
#include "union_find.h"

#include <boost/random.hpp>

#include <cmath>
#include <iostream>
#include <vector>

int main() {
  // parameters
  const int sweeps = 8;
  const int length = 1024;
  const double prob = 0.5;

  // sqaure lattice
  square_lattice lattice(length);

  // random number generators
  boost::mt19937 eng(29833u);
  boost::variate_generator<boost::mt19937&, boost::uniform_real<> >
    r_uniform(eng, boost::uniform_real<>());

  // cluster information
  typedef looper::union_find::node node;
  std::vector<node> fragments(lattice.num_sites());

  // oservables
  double num_clusters = 0;

  // Monte Carlo steps
  for (int mcs = 0; mcs < sweeps; ++mcs) {

    // initialize cluster information
    std::fill(fragments.begin(), fragments.end(), node());

    // connect each bond with probability prob
    for (int b = 0; b < lattice.num_bonds(); ++b) {
      if (r_uniform() < prob) {
        // find root of lattice.source(b) with path-halving
        int r0 = lattice.source(b);
        if (!fragments[r0].is_root()) {
          while (true) {
            int p = fragments[r0].parent();
            if (fragments[p].is_root()) { r0 = p; break; }
            fragments[r0].set_parent(fragments[p].parent());
            r0 = p;
          }
        }
        // find root of lattice.target(b) with path-halving
        int r1 = lattice.target(b);
        if (!fragments[r1].is_root()) {
          while (true) {
            int p = fragments[r1].parent();
            if (fragments[p].is_root()) { r1 = p; break; }
            fragments[r1].set_parent(fragments[p].parent());
            r1 = p;
          }
        }
        // unify two clusters
        if (r0 != r1) {
          if (fragments[r0].weight() < fragments[r1].weight()) std::swap(r0, r1);
          fragments[r0].set_weight(fragments[r0].weight() + fragments[r1].weight());
          fragments[r1].set_parent(r0);
        }
      }
    }

    // count the number of clusters
    for (int f = 0; f < fragments.size(); ++f) if (fragments[f].is_root()) num_clusters += 1;
  }

  std::cout << "Number of Sites    = " << lattice.num_sites() << std::endl
            << "Number of Clusters = " << num_clusters / sweeps << std::endl;
}
