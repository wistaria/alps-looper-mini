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

#ifndef TIMER_MAPROF_H
#define TIMER_MAPROF_H

// ALPS_ENBALE_TIMER

#include <iostream>
#include <string>

#ifdef ALPS_ENABLE_TIMER

#include <maprof.h>

#ifndef LOOPER_VERSION
# define LOOPER_VERSION "unknown"
#endif

namespace alps {
namespace parapack {

class timer_base {
public:
  BOOST_STATIC_CONSTANT(int, detailed = (1 << 0));
  void setup() {
    this->registrate(0, 0, "all");
    this->start(0);
    maprof_setup("ALPS/looper mini", LOOPER_VERSION);
  }
  void set_repstdout(bool rep) { repstdout_ = rep; }
  void set_parameter(std::string const& key, std::string const& str) const {
    maprof_profile_add_str(key.c_str(), str.c_str());
  }
  void set_parameter(std::string const& key, unsigned int n) const {
    maprof_profile_add_int(key.c_str(), n);
  }
  void set_parameter(std::string const& key, int n) const {
    maprof_profile_add_int(key.c_str(), n);
  }
  void set_parameter(std::string const& key, double r) const {
    maprof_profile_add_float(key.c_str(), r);
  }
  void clear() {}
  void registrate(std::size_t id, std::size_t, std::string const& label, int = 0) {
    if (label.empty()) {
      std::cerr << "Error: empty label\n";
      boost::throw_exception(std::invalid_argument("empty label"));
    }
    if (id >= labels_.size()) {
      labels_.resize(id + 1, "");
    }
    if (!labels_[id].empty()) {
      std::cerr << "Error: duplicated id: " << id << std::endl;
      boost::throw_exception(std::invalid_argument("duplicated id"));
    }
    labels_[id] = label;
    maprof_add_section(label.c_str(), id);
  }
  void start(std::size_t id, bool = true) const { maprof_time_start(id); }
  void stop(std::size_t id, bool = true) const { maprof_time_stop(id); }
  void finalize(std::ostream& = std::clog) const {
    this->stop(0);
    maprof_output();
  }
protected:
  std::size_t size() const { return labels_.size(); }
  std::string const& label(std::size_t id) const { return labels_[id]; }
  bool repstdout() const { return repstdout_; }
private:
  bool repstdout_;
  std::vector<std::string> labels_;
};

class timer : public timer_base {
public:
  timer() { timer_base::setup(); }
  void summarize(std::ostream& = std::clog) const {
    timer_base::finalize();
    if (timer_base::repstdout()) {
      maprof_print(0, timer_base::label(0).c_str());
      for (int id = 0; id < timer_base::size(); ++id) {
        if (!timer_base::label(id).empty()) maprof_print_time(id, timer_base::label(id).c_str());
      }
    }
  }
};

} // namespace parapack
} // namespace alps

#else

namespace alps {
namespace parapack {

class timer {
public:
  BOOST_STATIC_CONSTANT(int, detailed = (1 << 0));
  void set_repstdout(bool) {}
  void set_parameter(std::string const&, std::string const&) const {}
  void set_parameter(std::string const&, unsigned int) const {}
  void set_parameter(std::string const&, int) const {}
  void set_parameter(std::string const&, double) const {}
  void clear() {}
  void registrate(std::size_t, std::size_t, std::string const&, int = 0) {}
  void start(std::size_t, bool = true) {}
  void stop(std::size_t, bool = true) {}
  void summarize(std::ostream& = std::clog) const {}
};

} // namespace parapack
} // namespace alps

#endif

#endif // ALPS_PARAPACK_TIMER_H
