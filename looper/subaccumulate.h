/*****************************************************************************
*
* ALPS/looper mini: MiniAppli for multi-cluster quantum Monte Carlo algorithms
*
* Copyright (C) 1997-2010 by Synge Todo <wistaria@comp-phys.org>
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*
*****************************************************************************/

#ifndef LOOPER_SUBACCUMULATE_H
#define LOOPER_SUBACCUMULATE_H

#include <vector>

namespace looper {

template<typename T>
inline int subaccumulate(std::vector<T> const& v, int tid) {
  int n = T(0);
  for (int p = 0; p < tid; ++p) n += v[p];
  return n;
}

} // end namespace looper

#endif // LOOPER_SUBACCUMULATE_H
