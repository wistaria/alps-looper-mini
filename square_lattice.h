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

#ifndef LOOPER_STANDALONE_SQUARE_LATTICE_H
#define LOOPER_STANDALONE_SQUARE_LATTICE_H

#include <vector>

class square_lattice {
public:
  explicit square_lattice(unsigned int L, bool has_4nn = false) {
    init(L, L, has_4nn);
  }
  explicit square_lattice(unsigned int Lx, unsigned int Ly, bool has_4nn = false) {
    init(Lx, Ly, has_4nn);
  }
  unsigned int length_x() const { return length_x_; }
  unsigned int length_y() const { return length_y_; }
  unsigned int num_sites() const { return num_sites_; }
  double site_phase(unsigned int s) const { return site_phase_[s]; }

  unsigned int num_bonds() const { return num_bonds_; }
  unsigned int num_bonds_4nn() const { return num_bonds_4nn_; }
  unsigned int source(unsigned int b) const { return source_[b]; }
  unsigned int target(unsigned int b) const { return target_[b]; }
  double bond_phase(unsigned int b) const { return bond_phase_[b]; }
  
  unsigned int num_plqs() const { return 2 * num_sites(); }
  unsigned int plq2bond0(unsigned int p) const { return p; }
  unsigned int plq2bond1(unsigned int p) const { return target_[p + 1 - (p % 2) * 2] * 2 + p % 2; }

protected:
  int site_coordinate_x(unsigned int s) const { return s % length_x_; }
  int site_coordinate_y(unsigned int s) const { return s / length_x_; }
  unsigned int site_index(int x, int y) const {
    while (x < 0) x += length_x_;
    x = x % length_x_;
    while (y < 0) y += length_y_;
    y = y % length_y_;
    return y * length_x_ + x;
  }
  void init(unsigned int Lx, unsigned int Ly, bool has_4nn) {
    length_x_ = Lx;
    length_y_ = Ly;
    num_sites_ = length_x_ * length_y_;
    num_bonds_ = 2 * num_sites_;
    num_bonds_4nn_ = (has_4nn ? 4 * num_sites_ : 0);
    source_.resize(num_bonds_ + num_bonds_4nn_);
    target_.resize(num_bonds_ + num_bonds_4nn_);
    site_phase_.resize(num_sites_);
    bond_phase_.resize(num_bonds_);
    for (unsigned int s = 0; s < num_sites_; ++s) {
      int x = site_coordinate_x(s);
      int y = site_coordinate_y(s);
      site_phase_[s] = 2 * ((x + y) % 2) - 1;
    }
    for (unsigned int b = 0; b < num_bonds_; ++b) {
      unsigned int s = b / 2;
      int x = site_coordinate_x(s);
      int y = site_coordinate_y(s);
      unsigned int t;
      // . 1 .
      // . x 0
      if (b % 2 == 0) {
        t = site_index(x + 1, y);
        bond_phase_[b] = 2.0 * ((b / 2) % 2) - 1.0;
      } else {
        t = site_index(x, y + 1);
        bond_phase_[b] = 2.0 * ((b / length_x_ / 2) % 2) - 1.0;
      }
      source_[b] = s;
      target_[b] = t;
    }
    for (unsigned int b = 0; b < num_bonds_4nn_; ++b) {
      unsigned int s = b / 4;
      int x = site_coordinate_x(s);
      int y = site_coordinate_y(s);
      unsigned int t;
      // . 2 . 1 .
      // 3 . . . 0
      // . . x . .
      if (b % 4 == 0) {
        t = site_index(x + 2, y + 1);
      } else if (b % 4 == 1) {
        t = site_index(x + 1, y + 2);
      } else if (b % 4 == 2) {
        t = site_index(x - 1, y + 2);
      } else if (b % 4 == 3) {
        t = site_index(x - 2, y + 1);
      }
      source_[num_bonds_ + b] = s;
      target_[num_bonds_ + b] = t;
    }
  }

private:
  unsigned int length_x_, length_y_;
  unsigned int num_sites_, num_bonds_, num_bonds_4nn_;
  std::vector<unsigned int> source_;
  std::vector<unsigned int> target_;
  std::vector<double> site_phase_;
  std::vector<double> bond_phase_;
};

#endif
