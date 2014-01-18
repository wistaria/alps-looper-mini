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

#ifndef LOOPER_UNION_FIND_R_H
#define LOOPER_UNION_FIND_R_H

#ifndef ALPS_INDEP_SOURCE
# include <alps/config.h>
# if defined(LOOPER_ENABLE_OPENMP) && defined(ALPS_ENABLE_OPENMP_WORKER) && !defined(LOOPER_OPENMP)
#  define LOOPER_OPENMP
# endif
#else
# if defined(LOOPER_ENABLE_OPENMP) && defined(_OPENMP) && !defined(LOOPER_OPENMP)
#  define LOOPER_OPENMP
# endif
#endif

#include <iostream>
#include <iomanip>
#include <vector>
#include "atomic.h"

#ifdef LOOPER_OPENMP
# include <omp.h>
#endif

namespace looper {
namespace union_find {

class node_rank {
public:
  node_rank () : rank_(0), parent_(-1), id_(-1) {}
  ~node_rank () {}

  void set_parent(int parent) { parent_ = parent + 1; }
  void set_rank(int rank) { rank_ = rank; }
  void inc_rank() { ++rank_; }
  void set_id (int id) { id_ = id; }

  int id() const {return id_; }
  int rank() const { return rank_; }
  int parent() const { return parent_ - 1; }
  bool is_root() const { return parent_ <= 0; }
  int lock_root() {
    int p = parent_;
    if(p < 0 && compare_and_swap(parent_, p, 0)) {
      return p - 1;
    } else {
      return 0;
    }
  }

private:
  int id_; // cluster_id to which the node belongs.
  int rank_;
  int parent_; // negative weight for root
};

// add a new vertex to a graph
// thread-unsafe  
template<class T>
inline int add(std::vector<T>& v) {
  v.push_back(T());
  return v.size() - 1; // return index of new node
}

// Find Root
template<class T>
inline int root_index(std::vector<T> const& v, int g) {
#ifdef LOOPER_OPENMP
  T c = v[g];
  while (!c.is_root()) {
    g = c.parent();
    c = v[g];
  }
#else
  while (!v[g].is_root()) {
    g = v[g].parent();
  }
#endif
  return g;
}

template<class T>
inline T const& root(std::vector<T> const& v, int g) { return v[root_index(v, g)]; }

template<class T>
inline T const& root(std::vector<T> const& v, T const& n) {
  return n.is_root() ? n : root(v, n.parent());
}

template<class T>
inline int cluster_id(std::vector<T> const& v, int g) { return root(v, g).id(); }

template<class T>
inline int cluster_id(std::vector<T> const& v, T const& n) { return root(v, n).id(); }

// template<class T>
// inline void set_root(std::vector<T>& v, int g) {
// #ifdef LOOPER_OPENMP
//   while(true) {
//     int r = root_index(v, g);
//     if (r == g) {
//       return;      
//     } else {
//       int rk = v[r].lock.root();
//       if (rk != 0) {
//         v[g].set_rank(rk);
//         v[r].set_parent(g); // release lock
//         return;
//       }
//     }
//   }
// #else
//   int r = root_index(v, g);
//   if (r != g) {
//     v[g] = v[r]; // unneeded operation?
//     v[r].set_parent(g);
//   }
// #endif
// }

// path-compression
template<class T>
inline void update_link(std::vector<T>& v, int g, int r) {
  while(g != r) {
    int p = v[g].parent();
    v[g].set_parent(r);
    g = p;
  }
}

// Find root & Path halving
template<class T>
inline int root_index_ph(std::vector<T>& v, int g) {
  if (v[g].is_root()) return g;
  while(true) {
    int p = v[g].parent();
    if (v[p].is_root()) return p;
    v[g].set_parent(v[p].parent());
    g = p;
  }
}

//  input: g1 is not root and r0 >= r1
// output: g1 is root,  or g1 is not root with r0 <= r1
// g0 == g1 is possible
template<class T>
inline int find_above_rank(std::vector<T>& v, const int g0, int g1) {
  const int rk0 = v[g0].rank();
  while (rk0 >= v[g1].rank()) {
    g1 = v[g1].parent();
    if (v[g1].is_root() || (g0 == g1)) break;
  }
  return g1;
}

// thread-safe Union-by-Rank
template<class T>
inline int unify_by_rank(std::vector<T>& v, int g0, int g1) {
  using std::swap;
  int rt0 = root_index(v, g0);
  int rt1 = root_index(v, g1);

#ifdef LOOPER_OPENMP
  int lrt0 = 0;
  int lrt1 = 0;
  while (true) {
    if (rt0 == rt1) return rt0;
    lrt0 = v[rt0].lock_root(); // lrt0 := locked rt0
    lrt1 = v[rt1].lock_root();

    if (lrt0 != 0 && lrt1 != 0) break;
    if (lrt0 != 0) v[rt0].set_parent(lrt0);
    if (lrt1 != 0) v[rt1].set_parent(lrt1);

    rt0 = root_index(v, g0);
    rt1 = root_index(v, g1);
  }
  int rk0 = v[rt0].rank();
  int rk1 = v[rt1].rank();
  if (rk0 < rk1) swap(rk0, rk1);
  v[rt0].set_parent(lrt0); // release lock
  v[rt1].set_parent(rt0); // release lock
  if (rk0 == rk1) v[rt0].inc_rank();
#else
  if (rt0 != rt1) {
    int rk0 = v[rt0].rank();
    int rk1 = v[rt1].rank();
    if (rk0 < rk1) swap(rt0, rt1);    
    v[rt1].set_parent(rt0);
    if (rk0 == rk1) v[rt0].inc_rank();
  }
#endif
  return rt0;
}

// thread-unsafe Union-by-Rank & Path compression
template<class T>
inline int unify_by_rank_cmp(std::vector<T>& v, int g0, int g1) {
  using std::swap;
  int rt0 = root_index(v, g0);
  int rt1 = root_index(v, g1);

  if (rt0 != rt1) {
    int rk0 = v[rt0].rank();
    int rk1 = v[rt1].rank();
    if (rk0 < rk1) swap(rt0, rt1);    
    v[rt1].set_parent(rt0);
    if (rk0 == rk1) v[rt0].inc_rank();
  }

  update_link(v, g0, rt0);
  update_link(v, g1, rt0);

  return rt0;
}

// thead-safe Union-by-Rank & Path halving
template<class T>
inline int unify_by_rank_ph(std::vector<T>& v, int g0, int g1) {
  using std::swap;
  int rt0 = root_index_ph(v, g0);
  int rt1 = root_index_ph(v, g1);

#ifdef LOOPER_OPENMP
  int lrt0 = 0;
  int lrt1 = 0;
  while (true) {
    if (rt0 == rt1) return rt0;
    lrt0 = v[rt0].lock_root(); // lrt0 := locked rt0
    lrt1 = v[rt1].lock_root();

    if (lrt0 != 0 && lrt1 != 0) break;
    if (lrt0 != 0) v[rt0].set_parent(lrt0);
    if (lrt1 != 0) v[rt1].set_parent(lrt1);

    rt0 = root_index_ph(v, g0);
    rt1 = root_index_ph(v, g1);
  }
  int rk0 = v[rt0].rank();
  int rk1 = v[rt1].rank();
  if (rk0 < rk1) swap(rk0, rk1);
  if (rk0 == rk1) v[rt0].inc_rank();
  v[rt0].set_parent(lrt0); // release lock
  v[rt1].set_parent(rt0); // release lock
#else
  if (rt0 != rt1) {
    int rk0 = v[rt0].rank();
    int rk1 = v[rt1].rank();
    if (rk0 < rk1) swap(rt0, rt1);    
    v[rt1].set_parent(rt0);
    if (rk0 == rk1) v[rt0].inc_rank();
  }
#endif
  return rt0;
}


// thread-unsafe unify by zigzag algorithm
// zigzag algorithm doesn't return the new root-node
template<class T>  
inline int unify_by_rank_zz(std::vector<T>& v, int g0, int g1) {
  using std::swap;

  while (true) {
    if (v[g0].rank() < v[g1].rank()) swap(g0, g1);
    if (v[g0].rank() > v[g1].rank()) {
      if (v[g1].is_root()) {
	v[g1].set_parent(g0);
	return g0;
      } else {
	g1 = v[g1].parent();	
	continue;
      }
    } else { // v[g0].rank() == v[g1].rank()
      if (g0 == g1) {
	return g0;
      } else if (!v[g0].is_root()) {
	g0 = v[g0].parent();
	continue;
      } else if (!v[g1].is_root()) {
	g1 = v[g1].parent();
	continue;
      } else { // both g0 and g1 are root.
	v[g0].inc_rank();
	v[g1].set_parent(g0);
	return g0;
      }
    }
  }  
}

// thread-safe unify by zigzag algorithm
// zigzag algorithm doesn't return the new root-node
template<class T>  
inline int unify_by_rank_zz_omp(std::vector<T>& v, int g0, int g1) {
  using std::swap;
  int lrt0 = 0;
  int lrt1 = 0;

  // debug_out_omp(v, g0, g1, lrt0, lrt1, __LINE__);
  while (true) {
    if (v[g0].rank() < v[g1].rank()) swap(g0, g1);
    if (v[g0].rank() > v[g1].rank()) {
      if (v[g1].is_root()) {
	lrt1 = v[g1].lock_root();
	if (lrt1) {
	  if (v[g0].rank() > v[g1].rank()) {
	    v[g1].set_parent(g0); // release lock
	    return g0;
	  } else {
	    v[g1].set_parent(lrt1); // release lock
	    continue;
	  }
	} else {
	  continue;
	}
      } else {
	g1 = v[g1].parent();
	continue;
      }
    } else { // v[g0].rank() == v[g1].rank()
      if (g0 == g1) {
	return g0;
      } else if (!v[g0].is_root()) {
	g0 = v[g0].parent();
	continue;
      } else if (!v[g1].is_root()) {
	g1 = v[g1].parent();
	continue;
      } else { // both g0 and g1 are root.
	lrt0 = v[g0].lock_root();
	lrt1 = v[g1].lock_root();
	
	if (lrt0 && lrt1 && (v[g0].rank() == v[g1].rank())) {
	  v[g0].inc_rank();
	  v[g0].set_parent(lrt0); // release lock
	  v[g1].set_parent(g0);   // release lock
	  return g0;
	} else {
	  if (lrt0) v[g0].set_parent(lrt0); // release lock
	  if (lrt1) v[g1].set_parent(lrt1); // release lock
	  continue;
	}
      }
    }
  }
}


template<class T>
inline int unify(std::vector<T>& v, int g0, int g1) {
  return unify_by_rank_zz_omp(v, g0, g1);
  // return unify_by_rank_ph(v, g0, g1);
}


template<class T>
void debug_out(const std::vector<T>& v) {
  std::cerr << "    id: ";
  for (int i = 0; i < v.size(); ++i) {
    std::cerr << std::setw(2) << i << " ";
  }
  std::cerr << std::endl;
  std::cerr << "  rank: ";
  for (int i = 0; i < v.size(); ++i) {
    std::cerr << std::setw(2) << v[i].rank() << " ";
  }
  std::cerr << std::endl;
  std::cerr << "parent: ";
  for (int i = 0; i < v.size(); ++i) {
    std::cerr << std::setw(2) << v[i].parent() << " ";
  }
  std::cerr << std::endl;
  std::cerr << "   cid: ";
  for (int i = 0; i < v.size(); ++i) {
    std::cerr << std::setw(2) << v[i].id() << " ";
  }
  std::cerr << std::endl << std::endl;
}

template<class T>
void debug_out_omp(const std::vector<T>& v,
		   const int g0, const int g1, const int lrt0, const int lrt1,
		   const int linenum) {
  #pragma omp critical
  {
    int tid = omp_get_thread_num();
    std::cerr << "TID: " << tid << " **L" << linenum << " g0, g1 = " << g0 << ", " << g1
	      << " lrt0, lrt1 = " <<lrt0 << ", " << lrt1 << std::endl << std::flush;
    debug_out(v);
  }
}


template<typename T>
int count_root(std::vector<T>& v, int start, int n) {
  int nc = 0;
  for (int i = start; i < start + n; ++i)
    if (v[i].is_root()) ++nc;
  return nc;
}

template<typename T>
int count_root_p(std::vector<T>& v, int start, int n) {
  int nc = 0;
  #pragma omp for schedule(static) nowait
  for (int i = start; i < start + n; ++i)
    if (v[i].is_root()) ++nc;
  return nc;
}

template<typename T>
int set_id(std::vector<T>& v, int start, int n, int nc) {
  for (int i = start; i < start +n; ++i)
    if (v[i].is_root()) v[i].set_id(nc++);
  return nc;
}

template<typename T>
int set_id_p(std::vector<T>& v, int start, int n, int nc) {
  #pragma omp for schedule(static) nowait
  for (int i = start; i < start + n; ++i)
    if (v[i].is_root()) v[i].set_id(nc++);
  return nc;
}

template<typename T>
void copy_id(std::vector<T>& v, int start, int n) {
  for (int i = start; i < start + n; ++i)
    v[i].set_id(cluster_id(v, i));
}

template<typename T>
void copy_id_p(std::vector<T>& v, int start, int n) {
  #pragma omp for schedule(static) nowait
  for (int i = start; i < start + n; ++i)
    v[i].set_id(cluster_id(v, i));
}

} // end namespace union_find_r
} // end namespace looper

#endif
