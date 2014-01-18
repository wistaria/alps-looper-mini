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

#ifndef LOOPER_STANDALONE_LATTICE_SHARING_H
#define LOOPER_STANDALONE_LATTICE_SHARING_H

#if defined(LOOPER_ENABLE_OPENMP) && defined(_OPENMP)

#include <algorithm>
#include <iostream>
#include <vector>
#include <omp.h>

class lattice_sharing {
public:
  template<class LATTICE>
  lattice_sharing(LATTICE const& lat)
    : bond_offset_(omp_get_max_threads()), num_bonds_local_(omp_get_max_threads()),
      share_(lat.num_bonds()),
      plq_offset_(omp_get_max_threads()), num_plqs_local_(omp_get_max_threads()),
      plq_share_(lat.num_plqs()) {
    std::vector<int> bond_owner(lat.num_bonds());
    std::vector<int> plq_owner(lat.num_plqs());
    std::vector<std::vector<int> > site_owners(lat.num_sites());
    #pragma omp parallel
    {
      int tid = omp_get_thread_num();
      int n = 0;
      # pragma omp for schedule(static)
      for (int b = 0; b < lat.num_bonds(); ++b) {
        ++n;
        bond_owner[b] = tid;
      }
      num_bonds_local_[tid] = n;
      int m = 0;
      # pragma omp for schedule(static)
      for (int q = 0; q < lat.num_plqs(); ++q) {
        ++m;
        plq_owner[q] = tid;
      }
      num_plqs_local_[tid] = m;
    }
    bond_offset_[0] = 0;
    plq_offset_[0] = 0;
    for (int t = 1; t < omp_get_max_threads(); ++t) {
      bond_offset_[t] = bond_offset_[t-1] + num_bonds_local_[t-1];
      plq_offset_[t] = plq_offset_[t-1] + num_plqs_local_[t-1];
    }

    for (int b = 0; b < lat.num_bonds(); ++b) { // TODO: parallelization
      site_owners[lat.source(b)].push_back(bond_owner[b]);
      site_owners[lat.target(b)].push_back(bond_owner[b]);
    }
    for (int q = 0; q < lat.num_plqs(); ++q) { // TODO: parallelization
      site_owners[lat.source(lat.plq2bond0(q))].push_back(plq_owner[q]);
      site_owners[lat.target(lat.plq2bond0(q))].push_back(plq_owner[q]);
      site_owners[lat.source(lat.plq2bond1(q))].push_back(plq_owner[q]);
      site_owners[lat.target(lat.plq2bond1(q))].push_back(plq_owner[q]);
    }

    #pragma omp parallel for schedule(static)
    for (int b = 0; b < lat.num_bonds(); ++b) {
      bool found_other = false;
      int other_owner = 0;
      for (int i = 0; i < 2; ++i) {
        int s = (i == 0) ? lat.source(b) : lat.target(b);
        for (int k = 0; k < site_owners[s].size(); ++k) {
          if (site_owners[s][k] != bond_owner[b]) {
            if (found_other) {
              if (site_owners[s][k] != other_owner) {
                std::cerr << "Error: This lattice is not supported!!!\n";
                std::exit(-1);
              }
            } else {
              found_other = true;
              other_owner = site_owners[s][k];
            }
          }
        }
      }
      if (found_other) {
        share_[b] = other_owner;
      } else {
        share_[b] = bond_owner[b];
      }
    }

    #pragma omp parallel for schedule(static)
    for (int q = 0; q < lat.num_plqs(); ++q) {
      bool found_other = false;
      int other_owner = 0;
      for (int i = 0; i < 4; ++i) {
        int s;
        switch (i) {
        case 0 : s = lat.source(lat.plq2bond0(q)); break;
        case 1 : s = lat.target(lat.plq2bond0(q)); break;
        case 2 : s = lat.source(lat.plq2bond1(q)); break;
        case 3 : s = lat.target(lat.plq2bond1(q)); break;
        default : break;
        }
        for (int k = 0; k < site_owners[s].size(); ++k) {
          if (site_owners[s][k] != plq_owner[q]) {
            if (found_other) {
              if (site_owners[s][k] != other_owner) {
                std::cerr << "Error: This lattice is not supported!!!\n";
                std::exit(-1);
              }
            } else {
              found_other = true;
              other_owner = site_owners[s][k];
            }
          }
        }
      }
      if (found_other) {
        plq_share_[q] = other_owner;
      } else {
        plq_share_[q] = plq_owner[q];
      }
    }
    // for (int b = 0; b < lat.num_bonds(); ++b) std::cerr << b << ' ' << share_[b] << std::endl;
    // for (int q = 0; q < lat.num_plqs(); ++q) std::cerr << q << ' ' << plq_share_[q] << std::endl;
  }

  int bond_offset(int tid) const { return bond_offset_[tid]; }
  int num_bonds_local(int tid) const { return num_bonds_local_[tid]; }
  int operator()(int b) const { return share_[b]; }

  int plq_offset(int tid) const { return plq_offset_[tid]; }
  int num_plqs_local(int tid) const { return num_plqs_local_[tid]; }
  int plq(int q) const { return plq_share_[q]; }

private:
  std::vector<int> bond_offset_;
  std::vector<int> num_bonds_local_;
  std::vector<int> share_;
  std::vector<int> plq_offset_;
  std::vector<int> num_plqs_local_;
  std::vector<int> plq_share_;
};

#endif

#endif
