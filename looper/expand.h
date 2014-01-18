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

#ifndef LOOPER_EXPAND_H
#define LOOPER_EXPAND_H

#include <vector>

namespace looper {

template<typename T>
inline void expand(std::vector<T>& vec, int n, T const& t = T()) {
  if (vec.size() < n) vec.resize(n, t);
}

} // end namespace looper

#endif // LOOPER_EXPAND_H
