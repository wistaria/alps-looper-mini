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

// observable class

#ifndef LOOPER_STANDALONE_OBSERVABLE
#define LOOPER_STANDALONE_OBSERVABLE

#include <cmath> // for std::sqrt
#ifdef LOOPER_ENABLE_BOOST_ARCHIVE
# include <boost/archive/text_oarchive.hpp>
# include <boost/archive/text_iarchive.hpp>
#endif

class observable {
public:
  observable() : count_(0), sum_(0), esq_(0) {}
  void operator<<(double x) { sum_ += x; esq_ += x * x; ++count_; }
  double mean() const { return (count_ > 0) ? (sum_ / count_) : 0.; }
  double error() const {
    return (count_ > 1) ?
      std::sqrt((esq_ / count_ - mean() * mean()) / (count_ - 1)) : 0.;
  }
private:
#ifdef LOOPER_ENABLE_BOOST_ARCHIVE
  friend class boost::serialization::access;
  template<class Archive>
  void serialize(Archive & ar, const unsigned int version){
    ar & sum_;
    ar & esq_;
    ar & count_;
  }
#endif
  unsigned int count_;
  double sum_, esq_;
};

#endif // LOOPER_STANDALONE_OBSERVABLE
