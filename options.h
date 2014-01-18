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

// default & command line options

#include <iostream>
#include <string>
#include <boost/lexical_cast.hpp>

struct options {
  unsigned int length;
  double temperature;
  double beta;
#ifdef LOOPER_ENABLE_PLAQUETTE
  double J;
  double Q;
#endif
  unsigned int sweeps;
  unsigned int therm;
  std::string partition;
  bool duplex;
  unsigned int reserve_times;
  unsigned int reserve_operators;
  unsigned int reserve_fragments;
  unsigned int reserve_estimates;
//fj>
  unsigned int reserve_chunk_links;
  unsigned int reserve_chunk_estimates;
//fj<
  unsigned int dtime_interval;
  bool valid;
  bool t_is_set, b_is_set;
  bool therm_is_set;

  options(unsigned int argc, char *argv[], unsigned int len_def, double temp_def,
    bool parallel = false, bool print = true) :
    // default parameters
    length(len_def), temperature(temp_def), beta(0),
#ifdef LOOPER_ENABLE_PLAQUETTE
    J(1.0), Q(0.0),
#endif
    sweeps(1 << 16), therm(sweeps >> 3), partition(""), duplex(true),
    reserve_times(0), reserve_operators(0), reserve_fragments(0), reserve_estimates(0),
    //fj>
    reserve_chunk_links(0), reserve_chunk_estimates(0),
    //fj<
    dtime_interval(1), 
    valid(true) {

    t_is_set = 0; b_is_set = 0;
    therm_is_set = 0;

    for (unsigned int i = 1; i < argc; ++i) {
      switch (argv[i][0]) {
      case '-' :
        switch (argv[i][1]) {
        case 'l' :
          if (++i == argc) { usage(parallel, print); return; }
          length = boost::lexical_cast<unsigned int>(argv[i]); break;
        case 't' :
          if (++i == argc) { usage(parallel, print); return; }
          temperature = boost::lexical_cast<double>(argv[i]);
          t_is_set = 1; break;
        case 'b' :
          if (++i == argc) { usage(parallel, print); return; }
          beta = boost::lexical_cast<double>(argv[i]);
          b_is_set = 1; break;
        case 'm' :
          if (++i == argc) { usage(parallel, print); return; }
          therm = boost::lexical_cast<unsigned int>(argv[i]);
          therm_is_set = 1; break;
        case 'n' :
          if (++i == argc) { usage(parallel, print); return; }
          sweeps = boost::lexical_cast<unsigned int>(argv[i]);
          if (!therm_is_set) therm = sweeps >> 3; break;
        case 'p' :
          if (!parallel) { usage(parallel, print); return; }
          if (++i == argc) { usage(parallel, print); return; }
          partition = argv[i]; break;
        case 'r' :
          if (argv[i][2] == 't') {
            if (++i == argc) { usage(parallel, print); return; }
            reserve_times = boost::lexical_cast<unsigned int>(argv[i]); break;
          } else if (argv[i][2] == 'o') {
            if (++i == argc) { usage(parallel, print); return; }
            reserve_operators = boost::lexical_cast<unsigned int>(argv[i]); break;
          } else if (argv[i][2] == 'f') {
            if (++i == argc) { usage(parallel, print); return; }
            reserve_fragments = boost::lexical_cast<unsigned int>(argv[i]); break;
          } else if (argv[i][2] == 'e') {
            if (++i == argc) { usage(parallel, print); return; }
            reserve_estimates = boost::lexical_cast<unsigned int>(argv[i]); break;
          } else if (argv[i][2] == 'c') {
            if (argv[i][3] == 'l') {
              if (++i == argc) { usage(parallel, print); return; }
              reserve_chunk_links = boost::lexical_cast<unsigned int>(argv[i]); break;
            } else if (argv[i][3] == 'e') {
              if (++i == argc) { usage(parallel, print); return; }
              reserve_chunk_estimates = boost::lexical_cast<unsigned int>(argv[i]); break;
            } else {
              usage(parallel, print); return; 
            }
          } else {
            usage(parallel, print); return; 
          }
        case 's' :
          if (!parallel) { usage(parallel, print); return; }
          duplex = false; break;
        case 'd' :
          if (argv[i][2] == 'i') {
            if (++i == argc) { usage(parallel, print); return; }
            dtime_interval = boost::lexical_cast<unsigned int>(argv[i]); break;
          }
#ifdef LOOPER_ENABLE_PLAQUETTE
        case 'j' :
          if (++i == argc) { usage(parallel, print); return; }
          J = boost::lexical_cast<double>(argv[i]); break;
        case 'q' :
          if (++i == argc) { usage(parallel, print); return; }
          Q = boost::lexical_cast<double>(argv[i]); break;
#endif
        case 'h' :
          usage(parallel, print, std::cout); return;
        default :
          usage(parallel, print); return;
        }
        break;
      default :
        usage(parallel, print); return;
      }
    }

    if (length % 2 == 1 || temperature <= 0. || sweeps == 0) {
      std::cerr << "invalid parameter\n"; usage(parallel, print); return;
    }

    if ( t_is_set && b_is_set) {
      std::cerr << "You can't set temperature and beta at the same time.\n";
      usage(parallel, print); return;
    }
    if (b_is_set) {
      temperature = 1 / beta;
    }

    if (print) {
      std::cout << "System Linear Size        = " << length << '\n'
#ifdef LOOPER_ENABLE_PLAQUETTE
                << "J                         = " << J << '\n'
                << "Q                         = " << Q << '\n'
#endif
                << "Temperature               = " << temperature << '\n'
                << "MCS for Thermalization    = " << therm << '\n'
                << "MCS for Measurement       = " << sweeps << '\n'
                << "Detailed timer interval   = " << dtime_interval << '\n';
        if (reserve_times)
          std::cout << "Reserved size for times            = " << reserve_times << '\n';
        if (reserve_operators)
          std::cout << "Reserved size for operators        = " << reserve_operators << '\n';
        if (reserve_fragments)
          std::cout << "Reserved size for fragments        = " << reserve_fragments << '\n';
        if (reserve_estimates)
          std::cout << "Reserved size for estimates        = " << reserve_estimates << '\n';
        if (reserve_chunk_links)
          std::cout << "Reserved size for chunk::links     = " << reserve_chunk_links << '\n';
        if (reserve_chunk_estimates)
          std::cout << "Reserved size for chunk::estimates = " << reserve_chunk_estimates << '\n';
      if (parallel) {
        if (partition.size())
          std::cout << "Process Partition         = " << partition << '\n';
        std::cout << "Communication mode        = " << (duplex ? "duplex" : "simplex") << '\n';
      }
    }
  }

  void usage(bool parallel, bool print, std::ostream& os = std::cerr) {
    if (print) {
      os << "[command line options]\n\n"
         << "  -l int     System Linear Size\n"
         << "  -t double  Temperature\n"
         << "  -b double  Beta (inverse temperature)\n"
#ifdef LOOPER_ENABLE_PLAQUETTE
         << "  -j double  Coupling Constant J\n"
         << "  -q double  Coupling Constant Q\n"
#endif
         << "  -m int     MCS for Thermalization\n"
         << "  -n int     MCS for Measurement\n"
         << "  -i int     MCS between detailed timer measurement\n"
         << "  -rt int    Reserved size for times\n"
         << "  -ro int    Reserved size for operators\n"
         << "  -rf int    Reserved size for fragments\n"
         << "  -re int    Reserved size for estimates\n"
         << "  -rcl int   Reserved size for chunk::links\n"
         << "  -rce int   Reserved size for chunk::estimates\n";
      if (parallel)
        os << "  -p string  Process Partition\n"
           << "  -s         Use simplex communication mode instead of duplex one\n";
      os << "  -h         this help\n\n";
    }
    valid = false;
  }
};
