/*****************************************************************************
*
* ALPS/looper mini: MiniAppli for multi-cluster quantum Monte Carlo algorithms
*
* Copyright (C) 1997-2011 by Synge Todo <wistaria@comp-phys.org>
*
* Distributed under the Boost Software License, Version 1.0. (See accompanying
* file LICENSE_1_0.txt or copy at http://www.boost.org/LICENSE_1_0.txt)
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
  unsigned int sweeps;
  unsigned int therm;
  std::string partition;
  bool duplex;
  bool valid;
  bool t_is_set, b_is_set;
  bool therm_is_set;

  options(unsigned int argc, char *argv[], unsigned int len_def, double temp_def,
    bool parallel = false, bool print = true) :
    // default parameters
    length(len_def), temperature(temp_def), beta(0),
    sweeps(1 << 16), therm(sweeps >> 3), partition(""), duplex(true),
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
        case 's' :
          if (!parallel) { usage(parallel, print); return; }
          duplex = false; break;
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
                << "Temperature               = " << temperature << '\n'
                << "MCS for Thermalization    = " << therm << '\n'
                << "MCS for Measurement       = " << sweeps << '\n';
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
         << "  -m int     MCS for Thermalization\n"
         << "  -n int     MCS for Measurement\n"
         << "  -i int     MCS between detailed timer measurement\n";
      if (parallel)
        os << "  -p string  Process Partition\n"
           << "  -s         Use simplex communication mode instead of duplex one\n";
      os << "  -h         this help\n\n";
    }
    valid = false;
  }
};
