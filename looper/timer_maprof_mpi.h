/*****************************************************************************
*
* ALPS/looper mini: MiniAppli for multi-cluster quantum Monte Carlo algorithms
*
* Copyright (C) 2013-2014 by Synge Todo <wistaria@comp-phys.org>
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
*
*****************************************************************************/

#ifndef TIMER_MAPROF_MPI_H
#define TIMER_MAPROF_MPI_H

#include "timer_maprof.h"
#include <mpi.h>

#ifdef ALPS_ENABLE_TIMER

namespace alps {
namespace parapack {

class timer_mpi : public timer_base {
public:
  BOOST_STATIC_CONSTANT(int, start_barrier = (1 << 1));
  BOOST_STATIC_CONSTANT(int, stop_barrier = (1 << 2));
  BOOST_STATIC_CONSTANT(int, barrier = (start_barrier | stop_barrier));
  timer_mpi(MPI_Comm comm) { timer_base::setup(); }
  void summarize(std::ostream& = std::clog) const {
    timer_base::finalize();
    if (timer_base::repstdout()) {
      maprof_print(0, timer_base::label(0).c_str());
      for (int id = 0; id < timer_base::size(); ++id) {
        if (!timer_base::label(id).empty())
          maprof_print_time_mpi(id, timer_base::label(id).c_str());
      }
      maprof_print_time_mpi_full(0, timer_base::label(0).c_str());
    }
  }
};

} // namespace parapack
} // namespace alps

#else

namespace alps {
namespace parapack {

class timer_mpi : public timer {
public:
  BOOST_STATIC_CONSTANT(int, start_barrier = (1 << 1));
  BOOST_STATIC_CONSTANT(int, stop_barrier = (1 << 2));
  BOOST_STATIC_CONSTANT(int, barrier = (start_barrier | stop_barrier));
  timer_mpi(MPI_Comm comm) {}
};

} // namespace parapack
} // namespace alps

#endif

#endif // ALPS_PARAPACK_TIMER_MPI_H
