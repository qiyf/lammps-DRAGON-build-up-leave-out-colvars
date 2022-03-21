/* -*- c++ -*- ----------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

#ifdef PAIR_CLASS

PairStyle(tanhlr/cut/idealh,PairTanhlrCutIdealh)

#else

#ifndef LMP_PAIR_TANHLR_CUT_IDEALH_H
#define LMP_PAIR_TANHLR_CUT_IDEALH_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTanhlrCutIdealh : public Pair {
 public:
  PairTanhlrCutIdealh(class LAMMPS *);
  ~PairTanhlrCutIdealh();

  virtual void compute(int, int);
  virtual double single(int, int, int, int, double, double, double, double &);
  virtual void settings(int, char **);
  virtual void coeff(int, char **);
  virtual double init_one(int, int);
  virtual void write_restart(FILE *);
  virtual void read_restart(FILE *);
  virtual void write_restart_settings(FILE *);
  virtual void read_restart_settings(FILE *);
  virtual double memory_usage();

 protected:
  double cut_global;
  double **cut;
  double **htanh,**sigmah,**rmh, **rmh4;
  double **ptanh,**offset;

  int nideal, nbead, same_mol_flag, nHistoneMark, **active;
  int hm_flag1, hm_flag2;
  double *ideal_potential,***hm_potential;

  // hm_potential1 stands for 0-0
  // hm_potential2 stands for 0-1 and 1-0 (0 would be treated same)
  // hm_potential3 stands for 1-1
  // 0-0 is 1d, and 0-1 would be a 2-d vector, while 1-1 is 3-d
  double *hm_potential1,**hm_potential2,***hm_potential3;

  void allocate();
};

}

#endif
#endif
