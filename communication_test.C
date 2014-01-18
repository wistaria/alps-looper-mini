/*****************************************************************************
*
* ALPS/looper: multi-cluster quantum Monte Carlo algorithms for spin systems
*
* Copyright (C) 2011 by Synge Todo <wistaria@comp-phys.org>
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

#if defined(LOOPER_ENABLE_OPENMP) && defined(_OPENMP) && !defined(LOOPER_OPENMP)
# define LOOPER_OPENMP
#endif

#define ALPS_INDEP_SOURCE
#ifndef ALPS_ENABLE_TIMER
# define ALPS_ENABLE_TIMER
#endif
// #define ALPS_TRACE_TIMER
#define COMMUNICATION_TEST
// #define COMMUNICATION_DEBUG_OUTPUT

#include "options.h"
#include "parallel.h"
#include "timer_mpi.hpp"

#include <boost/timer.hpp>

#include <iostream>
#include <vector>

#if defined(ALPS_ENABLE_FAPP) || defined(ALPS_ENABLE_FAPP_PA)
template<>
char alps::parapack::detail::timer_base<alps::parapack::detail::clock_mpi>::msgs[512][256]
 = {"dummy string for aloocation"};
#endif

typedef looper::parallel_cluster_unifier<int, int> unifier_t;

int main(int argc, char* argv[]) {
  MPI_Init(&argc, &argv);
  int num_processes, process_id;
  MPI_Comm_size(MPI_COMM_WORLD, &num_processes);
  MPI_Comm_rank(MPI_COMM_WORLD, &process_id);

  alps::parapack::timer_mpi timer(MPI_COMM_WORLD);
  timer.registrate( 1, 2, "main");
  timer.registrate( 2, 1, "_parameters");
  timer.registrate( 3, 1, "_unifier__constructor");
  timer.registrate( 4, 1, "_MPI_Barrier");
  timer.registrate( 5, 1, "_boost__timer");
  timer.registrate( 6, 1, "_dummy_variables");
  timer.registrate( 7, 1, "_main_loop");
  timer.registrate( 8, 1, "_Monte_Carlo_steps");
  timer.registrate( 9, 1, "_MPI_Barrier");
  timer.registrate(10, 1, "_boost__timer_output");
  unifier_t::init_timer(timer);

  timer.start(1);

  timer.start(2);
  // parameters
  options p(argc, argv, 8, 0.2, true, process_id == 0);
  if (!p.valid) { MPI_Finalize(); std::exit(-1); }
  unsigned int sweeps = p.sweeps;
  unsigned int therm = p.therm;
  if (process_id == 0) {
    std::cout << "Number of processes = " << num_processes << std::endl
              << "Number of sweeps = " << sweeps << std::endl
              << "Number of warming up sweeps = " << therm << std::endl
              << "Data size = " << (double)(sizeof(int) * 2 * p.length) / 1048576.0 << " MByte\n";
  }
  timer.stop(2);

  timer.start(3);
  unifier_t unifier(MPI_COMM_WORLD, timer, p.length, p.partition, p.duplex);
  timer.stop(3);

  //
  // Monte Carlo steps
  //

  timer.start(4);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.stop(4);

  timer.start(5);
  boost::timer tm;
  timer.stop(5);

  // dummy variables
  timer.start(6);
  int coll;
  std::vector<int> fragments;
#ifdef LOOPER_OPENMP
  std::vector<std::vector<int> > estimates;
#else
  std::vector<int> estimates;
#endif
  timer.stop(6);

  timer.start(7);
  for (unsigned int mcs = 0; mcs < therm + sweeps; ++mcs) {
    if(mcs <= therm) timer.clear();
    timer.start(8);
    if (num_processes > 1) {
      unifier.unify(coll, fragments, 0, 0, estimates, timer);
    }
    timer.stop(8);
  }
  timer.stop(7);
  timer.start(9);
  MPI_Barrier(MPI_COMM_WORLD);
  timer.stop(9);
  timer.start(10);
  if (process_id == 0) {
    double elapsed = tm.elapsed();
    std::cout << "Elapsed time = " << elapsed << " sec\n"
              << "Speed = " << (therm + sweeps) / elapsed << " MCS/sec\n";
  }
  timer.stop(10);
  timer.stop(1);
  timer.summarize(std::cout);
  MPI_Finalize();
}
