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

PairStyle(tanhlr/cut/ideal,PairTanhlrCutIdeal)

#else

#ifndef LMP_PAIR_TANHLR_CUT_IDEAL_H
#define LMP_PAIR_TANHLR_CUT_IDEAL_H

#include "pair.h"

namespace LAMMPS_NS {

class PairTanhlrCutIdeal : public Pair {
 public:
  PairTanhlrCutIdeal(class LAMMPS *);
  ~PairTanhlrCutIdeal();

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

  int nideal, **ctcfList, *ctcfLoc, nctcf;
  double **ideal_potential, cg_factor, *scale_matrix, *scale_matrix_q, *qcumsum, *bindingProb, **cumProb;
  double qratio_cut;

  void allocate();
};

}

#endif
#endif
