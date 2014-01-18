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

#ifndef LOOPER_STANDALONE_CHAIN_LATTICE_H
#define LOOPER_STANDALONE_CHAIN_LATTICE_H

class chain_lattice {
public:
  chain_lattice(unsigned int L) : length_(L) {}
  unsigned int num_sites() const { return length_; }
  unsigned int num_bonds() const { return num_sites(); }
  unsigned int source(unsigned int b) const { return b; }
  unsigned int target(unsigned int b) const { return (b == length_-1) ? 0 : b+1; }
  double phase(unsigned int s) const { return (s & 1) ? 1.0 : -1.0; }

  // dummy functions
  unsigned int num_plqs() const { return 0; }
  unsigned int plq2bond0(unsigned int p) const { return 0; }
  unsigned int plq2bond1(unsigned int p) const { return 0; }

private:
  unsigned int length_;
};

#endif
