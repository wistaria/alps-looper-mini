#define ALPS_INDEP_SOURCE
#include <iostream>
#include <vector>
#include "atomic_impl.h"
#include "union_find_r.h"
#include "visualize.h"

int main() {
  // typedef looper::union_find::node node_t;
  typedef looper::union_find::node_rank node_t;
  using looper::union_find::debug_out;
  
  std::vector<node_t> v(10);
  // {0, 1}, {2,5}, {3,4,8,9}, {6}, {7}
  
  int r;
  for (int i = 0; i < 5; ++i) {
    v[i].set_parent(i+1);
    v[i+1].set_rank(v[i].rank() + 1);
  }
  for (int i = 6; i < 8; ++i) {
    v[i].set_parent(i+1);
    v[i+1].set_rank(v[i].rank() + 1);
  }

  debug_out(v);
  unify(v, 0, 2);
  debug_out(v);
  unify(v, 3, 9);
  debug_out(v);
  unify(v, 3, 7);
  debug_out(v);
  unify(v, 2, 4);
  debug_out(v);

  graph_out(v);
  
  return 0;
}
