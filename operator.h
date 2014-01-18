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

#ifndef LOOPER_STANDALONE_COMMON_H
#define LOOPER_STANDALONE_COMMON_H

#include <algorithm>
#include <iostream>
#include <vector>

enum operator_type { diagonal = 0, offdiagonal };

#ifndef LOOPER_ENABLE_PLAQUETTE

struct local_operator_t {
  local_operator_t() {}
  local_operator_t(unsigned int b, double t) : type(diagonal), bond(b), time(t) {}
  void flip() { type = (type == diagonal ? offdiagonal : diagonal); }
  void flip(int N, int k) {
    if (k >= 0)
      type = (type + k) % N;
    else {
      type = N - ((N - k - type) % N);
      if (type == N) type = 0;
    }
  }
  int type;
  unsigned int bond;
  unsigned int upper_cluster, lower_cluster;
  double time;
};

#else

struct local_operator_t {
  enum { bond_term = -1 };
  local_operator_t() {}
  local_operator_t(unsigned int b, double t, int q = bond_term) :
    type(diagonal), bond(b), time(t), plq(q) {}
  void flip() { type = (type == diagonal ? offdiagonal : diagonal); }
  void flip(int N, int k) {
    if (k >= 0)
      type = (type + k) % N;
    else {
      type = N - ((N - k - type) % N);
      if (type == N) type = 0;
    }
  }
  int type;
  unsigned int bond;
  int plq; // =bond_term: J-term, >=0: Q-term
  unsigned int upper_cluster, lower_cluster;
  double time;
};

#endif

struct estimate_t {
  estimate_t() : mag(0), size(0), length(0), to_flip(0) {}
  estimate_t& operator+=(estimate_t const& est) {
    mag += est.mag;
    size += est.size;
    length += est.length;
    to_flip += est.to_flip;
    return *this;
  }
  int to_flip;
  double mag;
  double size;
  double length;
};

struct estimate_un_t {
  estimate_un_t() : to_flip(0) {}
  estimate_un_t& operator+=(estimate_un_t const& est) {
    to_flip += est.to_flip;
    return *this;
  }
  int to_flip;
  double padding; // workaround for FCCpx
};

struct collector_t {
  collector_t() : nop_(0), nc_(0), range_(std::make_pair(1, 0)), noc_(0),
                  usus(0), smag(0), ssus(0) {}
  collector_t& operator+=(collector_t const& coll) {
    nop_ += coll.nop_;
    nc_ += coll.nc_;
    if (!coll.empty())
      range_ = std::make_pair(std::min(range_.first, coll.range_.first),
                              std::max(range_.second, coll.range_.second));
    usus += coll.usus;
    smag += coll.smag;
    ssus += coll.ssus;
    return *this;
  }
  collector_t& operator+=(estimate_t const& est) {
    usus += est.mag * est.mag;
    smag += est.size * est.size;
    ssus += est.length * est.length;
    return *this;
  }
  void set_num_clusters(unsigned int n) { nc_ = n; }
  void inc_num_clusters(unsigned int n) { nc_ += n; }
  double num_clusters() const { return nc_; }
  void set_num_operators(unsigned int n) { nop_ = n; }
  double num_operators() const { return nop_; }

  // for parallel QMC
  void set_num_open_clusters(unsigned int n) { noc_ = n; }
  unsigned int num_open_clusters() const { return noc_; }
  void clear_range() { range_ = std::make_pair(1, 0); }
  void set_range(int pos) { range_ = std::make_pair(pos, pos); }
  void set_range(collector_t const& coll) { range_ = coll.range_; }
  std::pair<int, int> const& range() const { return range_; }
  bool empty() const { return range_.first > range_.second; }

  double nop_; // total number of operators
  double nc_; // total number of (closed) clusters
  std::pair<int, int> range_; // configuration range (for parallel QMC)
  unsigned int noc_; // number of open clusters (for parallel QMC)
  double usus;
  double smag;
  double ssus;
};

struct collector_un_t {
  collector_un_t() : nop_(0), nc_(0), range_(std::make_pair(1, 0)), noc_(0),
                     umag_0(0), smag_0(0), smag_a(0) {}
  collector_un_t& operator+=(collector_un_t const& coll) {
    nop_ += coll.nop_;
    nc_ += coll.nc_;
    range_ = std::make_pair(std::min(range_.first, coll.range_.first),
                            std::max(range_.second, coll.range_.second));
    umag_0 += coll.umag_0;
    smag_0 += coll.smag_0;
    smag_a += coll.smag_a;
    return *this;
  }
  collector_un_t& operator+=(estimate_un_t const&) {
    // nothing to do
    return *this;
  }
  void set_num_clusters(unsigned int n) { nc_ = n; }
  void inc_num_clusters(unsigned int n) { nc_ += n; }
  double num_clusters() const { return nc_; }
  void set_num_operators(unsigned int n) { nop_ = n; }
  double num_operators() const { return nop_; }

  // for parallel QMC
  void set_num_open_clusters(unsigned int n) { noc_ = n; }
  unsigned int num_open_clusters() const { return noc_; }
  void clear_range() { range_ = std::make_pair(1, 0); }
  void set_range(int pos) { range_ = std::make_pair(pos, pos); }
  void set_range(collector_un_t const& coll) { range_ = coll.range_; }
  std::pair<int, int> const& range() const { return range_; }
  bool empty() const { return range_.first > range_.second; }

  double nop_; // total number of operators
  double nc_; // total number of (closed) clusters
  std::pair<int, int> range_; // configuration range (for parallel QMC)
  unsigned int noc_; // number of open clusters (for parallel QMC)
  double umag_0;
  double smag_0;
  double smag_a;
};

#endif
