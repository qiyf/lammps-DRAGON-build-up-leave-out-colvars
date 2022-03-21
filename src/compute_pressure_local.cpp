/* ----------------------------------------------------------------------
   LAMMPS - Large-scale Atomic/Molecular Massively Parallel Simulator
   http://lammps.sandia.gov, Sandia National Laboratories
   Steve Plimpton, sjplimp@sandia.gov

   Copyright (2003) Sandia Corporation.  Under the terms of Contract
   DE-AC04-94AL85000 with Sandia Corporation, the U.S. Government retains
   certain rights in this software.  This software is distributed under
   the GNU General Public License.

   See the README file in the top-level LAMMPS directory.
------------------------------------------------------------------------- */

/* ----------------------------------------------------------------------
   Contributing author: Naveen Michaud-Agrawal (Johns Hopkins U)
     K-space terms added by Stan Moore (BYU)

     Hijacked the compute-group-group to compute virial. Ignoring the 
     angular contributions for now.
------------------------------------------------------------------------- */

#include <mpi.h>
#include <string.h>
#include "compute_pressure_local.h"
#include "atom.h"
#include "update.h"
#include "force.h"
#include "pair.h"
#include "bond.h"
#include "neighbor.h"
#include "neigh_request.h"
#include "neigh_list.h"
#include "group.h"
#include "kspace.h"
#include "error.h"
#include <math.h>
#include "comm.h"
#include "domain.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

#define SMALL 0.00001

/* ---------------------------------------------------------------------- */

ComputePressureLocal::ComputePressureLocal(LAMMPS *lmp, int narg, char **arg) :
  Compute(lmp, narg, arg)
{
  if (narg < 3) error->all(FLERR,"Illegal compute pressure/local command");
  if (igroup) error->all(FLERR,"Compute pressure/local must use group all");
  // to compute the pressure, we really need the whole group

  scalar_flag = vector_flag = 1;
  size_vector = 3;
  extscalar = 1;
  extvector = 1;

  pairflag = 1;

  int iarg = 3;
  while (iarg < narg) {
    if (strcmp(arg[iarg],"pair") == 0) {
      if (iarg+2 > narg)
        error->all(FLERR,"Illegal compute pressure/local command");
      if (strcmp(arg[iarg+1],"yes") == 0) pairflag = 1;
      else if (strcmp(arg[iarg+1],"no") == 0) pairflag = 0;
      else error->all(FLERR,"Illegal compute pressure/local command");
      iarg += 2;
    } else if (strcmp(arg[iarg],"center") == 0) {
      if (iarg+5 > narg)
        error->all(FLERR,"Illegal compute pressure/local command");
      origin[0] = force->numeric(FLERR,arg[iarg+1]);
      origin[1] = force->numeric(FLERR,arg[iarg+2]);
      origin[2] = force->numeric(FLERR,arg[iarg+3]);
      radii  = force->numeric(FLERR,arg[iarg+4]);
      radii_sq = radii*radii;
      iarg += 5;
      //printf("pressure/local debug: %f %f %f %f\n", origin[0], origin[1], origin[2], radii);
    } else error->all(FLERR,"Illegal compute pressure/local command");
  }

  vector = new double[3];
}

/* ---------------------------------------------------------------------- */

ComputePressureLocal::~ComputePressureLocal()
{
  delete [] group2;
  delete [] vector;
}

/* ---------------------------------------------------------------------- */

void ComputePressureLocal::init()
{
  // if non-hybrid, then error if single_enable = 0
  // if hybrid, let hybrid determine if sub-style sets single_enable = 0

  if (pairflag && force->pair == NULL)
    error->all(FLERR,"No pair style defined for compute pressure/local");
  if (force->pair_match("hybrid",0) == NULL && force->pair->single_enable == 0)
    error->all(FLERR,"Pair style does not support compute pressure/local");

  if (pairflag) {
    pair = force->pair;
    cutsq = force->pair->cutsq;
  } else pair = NULL;

  // need an occasional half neighbor list

  if (pairflag) {
    int irequest = neighbor->request(this,instance_me);
    neighbor->requests[irequest]->pair = 0;
    neighbor->requests[irequest]->compute = 1;
    neighbor->requests[irequest]->occasional = 1;
  }
}

/* ---------------------------------------------------------------------- */

void ComputePressureLocal::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

double ComputePressureLocal::compute_scalar()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  vector[0] = vector[1] = vector[2] = 0.0;

  if (pairflag) pair_contribution();

  return scalar;
}

/* ---------------------------------------------------------------------- */

void ComputePressureLocal::compute_vector()
{
  invoked_scalar = invoked_vector = update->ntimestep;

  scalar = 0.0;
  vector[0] = vector[1] = vector[2] = 0.0;

  if (pairflag) pair_contribution();
}

/* ---------------------------------------------------------------------- */

void ComputePressureLocal::pair_contribution()
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz;
  double rsq,eng,fpair,factor_coul,factor_lj;
  int *ilist,*jlist,*numneigh,**firstneigh;

  double **x = atom->x;
  int *type = atom->type;
  int *mask = atom->mask;
  int nlocal = atom->nlocal;
  double *special_coul = force->special_coul;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  // 
  int btype;
  double f1[3], fbond, invol_fraction;

  // invoke half neighbor list (will copy or build if necessary)

  neighbor->build_one(list);

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms
  // skip if I,J are not in 2 groups

  double one[4];
  one[0] = one[1] = one[2] = one[3] = 0.0;

  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];

    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      factor_coul = special_coul[sbmask(j)];
      j &= NEIGHMASK;

      if (factor_lj == 0.0 && factor_coul == 0.0) continue;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      if (rsq < cutsq[itype][jtype]) {

        invol_fraction = compute_invol_fraction(x[i], x[j], rsq);
        //printf("pressure/local debug: %d %d %f %f\n", i, j, invol_fraction, rsq);

        eng = pair->single(i,j,itype,jtype,rsq,factor_coul,factor_lj,fpair);

        // energy only computed once so tally full amount
        // force tally is jgroup acting on igroup

        f1[0] = delx*fpair;
        f1[1] = dely*fpair;
        f1[2] = delz*fpair;
        if (newton_pair || j < nlocal) {
          one[0] += (delx*f1[0]+dely*f1[1]+delz*f1[2])*invol_fraction;

        // energy computed twice so tally half amount
        // only tally force if I own igroup atom

        } else {
          one[0] += 0.5*(delx*f1[0]+dely*f1[1]+delz*f1[2])*invol_fraction;
        }
      }
    }
  }

  //printf("pressure/local pair virial debug: %f\n", one[0]);

  // bond part //
  // copied from /home/binz/Packages/lammps-1Feb14/src/compute_stress_spatial.cpp
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;
  int i1,i2,n;
  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    btype = bondlist[n][2];

    delx = x[i1][0] - x[i2][0];
    dely = x[i1][1] - x[i2][1];
    delz = x[i1][2] - x[i2][2];
    rsq = delx*delx + dely*dely + delz*delz;
    eng = force->bond->single(btype,rsq,i1,i2,fbond);
    f1[0] = delx*fbond;
    f1[1] = dely*fbond;
    f1[2] = delz*fbond;

    invol_fraction = compute_invol_fraction(x[i1], x[i2], rsq);
    one[0] += (delx*f1[0]+dely*f1[1]+delz*f1[2])*invol_fraction;
  }

  //printf("pressure/local bond virial debug: %f\n", one[0]);

  double all[4];
  MPI_Allreduce(one,all,4,MPI_DOUBLE,MPI_SUM,world);
  scalar += all[0];
  vector[0] += all[1]; vector[1] += all[2]; vector[2] += all[3];

  //printf("pressure/local total virial debug: %f\n", scalar);
}

double ComputePressureLocal::compute_invol_fraction(double *x1, double *x2, double rsq)
{
  // determine the portion of the distance that's within the sphere
  // I have a note on this in the OneNote Chromosome/Nucleolus/Revision
  /*
   * o----------o
   *    \  |
   *     \ |
   *       o
   */

  double cx,cy,cz, l1sq,l2sq, invol_fraction, l1,l2;
  double dist_l, dist_r, dist_from_com_sq, r_l;
  int invol_flag_1, invol_flag_2;

  invol_flag_1 = 0; invol_flag_2 = 0;

  cx = origin[0]-x1[0]; 
  cy = origin[1]-x1[1]; 
  cz = origin[2]-x1[2];
  l1sq = cx*cx + cy*cy + cz*cz;
  if (l1sq<=radii_sq) invol_flag_1 = 1;
  l1 = sqrt(l1sq);

  double delx1 = x2[0] - x1[0];
  double dely1 = x2[1] - x1[1];
  double delz1 = x2[2] - x1[2];
  double r1 = sqrt(delx1*delx1 + dely1*dely1 + delz1*delz1);

  double c = delx1*cx + dely1*cy + delz1*cz;
  c /= r1*l1;
  if (c > 1.0) c = 1.0;
  if (c < -1.0) c = -1.0;

  double theta_l = acos(c);

  cx = origin[0]- x2[0];
  cy = origin[1]- x2[1];
  cz = origin[2]- x2[2];
  l2sq = cx*cx + cy*cy + cz*cz;
  if (l2sq<=radii_sq) invol_flag_2 = 1;


  if (invol_flag_1 && invol_flag_2) {
      invol_fraction = 1.0;
  } else {
      // determine the distance of the line from the center
      r_l = (rsq + l1sq - l2sq) / (2.0*sqrt(rsq));
      dist_from_com_sq = l1sq - r_l*r_l;
      //printf("pressure/local debug: %f %f\n", dist_from_com_sq, l1sq*sin(theta_l)*sin(theta_l));

      if (dist_from_com_sq >= radii_sq) {
        invol_fraction = 0.0;
      } else {

        // will the line cross the sphere or not?
        double c = -delx1*cx - dely1*cy - delz1*cz;
        l2 = sqrt(l2sq);
        c /= r1*l2;
        if (c > 1.0) c = 1.0;
        if (c < -1.0) c = -1.0;
        double theta_r = acos(c);

        if ( ((!invol_flag_1) && (!invol_flag_2)) && ( (theta_l > MY_PI2) || (theta_r > MY_PI2) ) ){
            // if both particles are outside and one of the angles is larger than PI_2, there is no crossing
          invol_fraction = 0.0;
        } else {
          // determine the left portion of distance inside the circle
          if (invol_flag_1) {
              dist_l = sqrt(l1sq - dist_from_com_sq);
          } else {
              dist_l = sqrt(radii_sq - dist_from_com_sq);
          }
          if (invol_flag_2) {
              dist_r = sqrt(l2sq - dist_from_com_sq);
          } else {
              dist_r = sqrt(radii_sq - dist_from_com_sq);
          }

          if (theta_l > MY_PI2) {
              dist_l = -dist_l; 
          }
          if (theta_r > MY_PI2) {
              dist_r = -dist_r; 
          }
          invol_fraction = (dist_l+dist_r)/sqrt(rsq);
          //printf("pressure/local debug: %f %f %f %f %f %f\n", x1[0],x1[1],x1[2],x2[0],x2[1],x2[2]);
          //printf("pressure/local debug: %f %f %f %d %d %f %f %f %f\n", l1sq, l2sq, radii_sq, invol_flag_1, invol_flag_2, r1*r1, dist_from_com_sq, theta_l, invol_fraction);
        }
      }
  }

  if ( (invol_fraction<0.0) || (invol_fraction>1.0)) error->all(FLERR,"invol_fraction in pressure/local should be within [0,1.0]!!!");
  return invol_fraction;
}

