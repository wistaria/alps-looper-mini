/*****************************************************************************
*
* ALPS/looper mini: MiniAppli for multi-cluster quantum Monte Carlo algorithms
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>,
*                            Haruhiko Matsuo <halm@looper.t.u-tokyo.ac.jp>
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*
*****************************************************************************/

#ifndef LOOPER_PRIME_FACTORIZATION_H
#define LOOPER_PRIME_FACTORIZATION_H

#include <vector>

namespace looper {

inline std::vector<int> prime_factorization(int n) {
  int num_primes = 6;
  int primes[] = { 2, 3, 5, 7, 11, 13 };
  std::vector<int> factors;
  if (n <= 1) {
    factors.push_back(n);
  } else {
    for (int i = 0; i < num_primes; ++i) {
      int p = primes[i];
      while (n % p == 0) {
        n = n / p;
        factors.push_back(p);
      }
    }
    while (n > 1) {
      n = ((n - 1) / 2) + 1;
      factors.insert(factors.begin(), 2);
    }
  }
  return factors;
}

} // end namespace looper

#endif // LOOPER_PRIME_FACTORIZATION_H
