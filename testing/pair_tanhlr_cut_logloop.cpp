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
   Contributing authors: Arben Jusufi, Axel Kohlmeyer (Temple U.)
------------------------------------------------------------------------- */

#include "math.h"
#include "stdio.h"
#include "stdlib.h"
#include "string.h"
#include "pair_tanhlr_cut_logloop.h"
#include "atom.h"
#include "comm.h"
#include "force.h"
#include "neighbor.h"
#include "neigh_list.h"
#include "update.h"
#include "integrate.h"
#include "memory.h"
#include "error.h"
#include "math_const.h"

using namespace LAMMPS_NS;
using namespace MathConst;

/* ---------------------------------------------------------------------- */

PairTanhlrCutLogLoop::PairTanhlrCutLogLoop(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
}

/* ---------------------------------------------------------------------- */

PairTanhlrCutLogLoop::~PairTanhlrCutLogLoop()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(cutsq);

    memory->destroy(cut);
    memory->destroy(htanh);
    memory->destroy(sigmah);
    memory->destroy(rmh);
    memory->destroy(rmh4);
    memory->destroy(ptanh);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::compute(int eflag, int vflag)
{
  int i,j,ii,jj,inum,jnum,itype,jtype;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,rexp,utanh,factor_lj, tanr;
  int *ilist,*jlist,*numneigh,**firstneigh;

  evdwl = 0.0;
  if (eflag || vflag) ev_setup(eflag,vflag);
  else evflag = vflag_fdotr = 0;

  double **x = atom->x;
  double **f = atom->f;
  int *type = atom->type;
  int nlocal = atom->nlocal;
  double *special_lj = force->special_lj;
  int newton_pair = force->newton_pair;

  inum = list->inum;
  ilist = list->ilist;
  numneigh = list->numneigh;
  firstneigh = list->firstneigh;

  // loop over neighbors of my atoms

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
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];

      int iglobal, jglobal, jmi; // jglobal is not necessarily larger than iglobal
      iglobal = atom->tag[i];
      jglobal = atom->tag[j];
      jmi = abs(jglobal-iglobal);

       //printf("rsq %f", rsq, cutsq[itype][jtype]);
      if ( (rsq < cutsq[itype][jtype]) && (jmi<=MaxLoop) ) {
        r = sqrt(rsq);

        double ptanh_gamma;
        //ptanh_gamma = -ptanh[itype][jtype] * log(pow(jmi, (g_a-1.0)) * exp(-jmi/g_b) / g_norm);
        ptanh_gamma = ptanh[itype][jtype] - log(pow(jmi, (g_a-1.0)) * exp(-jmi/g_b) / g_norm);

        if (r <= rmh[itype][jtype]) { 
            rexp = (rmh[itype][jtype]-r)*sigmah[itype][jtype];
            // for us, p is 0.5 * h
            tanr = tanh(rexp);
            utanh = ptanh_gamma*(1.0+ tanr);
            // the extra negative sign is taken care in f
            fpair = factor_lj/r*ptanh_gamma*sigmah[itype][jtype]*(1-tanr*tanr);
        } else {
            utanh = ptanh_gamma*rmh4[itype][jtype] / rsq / rsq;
            // there is an extra r here which is to normalize position vector
            fpair = factor_lj/r*ptanh_gamma*rmh4[itype][jtype]* (4.0) /rsq/rsq/r;
        }

        f[i][0] += delx*fpair;
        f[i][1] += dely*fpair;
        f[i][2] += delz*fpair;
        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = utanh - offset[itype][jtype]*ptanh_gamma;
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }
  }

  if (vflag_fdotr) virial_fdotr_compute();
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::allocate()
{
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(cutsq,n+1,n+1,"pair:cutsq");

  memory->create(cut,n+1,n+1,"pair:cut");
  memory->create(htanh,n+1,n+1,"pair:htanh");
  memory->create(sigmah,n+1,n+1,"pair:sigmah");
  memory->create(rmh,n+1,n+1,"pair:rmh");
  memory->create(rmh4,n+1,n+1,"pair:rmh4");
  memory->create(ptanh,n+1,n+1,"pair:ptanh");
  memory->create(offset,n+1,n+1,"pair:offset");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::settings(int narg, char **arg)
{
  if (narg != 4) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);
  g_a        = force->numeric(FLERR,arg[1]);
  g_b        = force->numeric(FLERR,arg[2]);
  g_norm     = force->numeric(FLERR,arg[3]);

  MaxLoop = 2000; // this is only correct in the unit of kb; be careful
  /*
  g_norm = 0;
  // 2000 is chosen to be 2Mb, which is the largest loop observed 
  for (int i=0; i <=MaxLoop; i++) {
      g_norm += pow(i, (g_a-1)) * exp(-i/g_b);
  }
  */

  // reset cutoffs that have been explicitly set

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::coeff(int narg, char **arg)
{
  if (narg < 5 || narg > 6) error->all(FLERR,"Incorrect args for pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(arg[0],atom->ntypes,ilo,ihi);
  force->bounds(arg[1],atom->ntypes,jlo,jhi);

  double htanh_one = force->numeric(FLERR,arg[2]);
  double rmh_one = force->numeric(FLERR,arg[3]);
  double sigmah_one = force->numeric(FLERR,arg[4]);

  double cut_one = cut_global;
  if (narg == 6) cut_one = force->numeric(FLERR,arg[5]);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {
      htanh[i][j] = htanh_one;
      sigmah[i][j] = sigmah_one;
      rmh[i][j] = rmh_one;
      cut[i][j] = cut_one;
      setflag[i][j] = 1;
      count++;
    }
  }

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */

double PairTanhlrCutLogLoop::init_one(int i, int j)
{
  if (setflag[i][j] == 0) error->all(FLERR,"All pair coeffs are not set");
  ptanh[i][j] = htanh[i][j] * 0.5;

  if (offset_flag) {
    double rexp = (rmh[i][j]-cut[i][j])*sigmah[i][j];
    //offset[i][j] = ptanh[i][j] * (1.0 + tanh(rexp));
    //offset[i][j] = (1.0 + tanh(rexp)); // we remove the prefactor here, and add it on the fly, since
    offset[i][j] = pow(rmh[i][j],4.0)/pow(cut[i][j],4.0); // note that in the offset here, no prefactor is calculated.
                                       // the prefactor depends on ij
  } else offset[i][j] = 0.0;

  htanh[j][i] = htanh[i][j];
  sigmah[j][i] = sigmah[i][j];
  rmh[j][i] = rmh[i][j];
  rmh4[i][j] = pow(rmh[i][j],4.0);
  rmh4[j][i] = rmh4[i][j];
  ptanh[j][i] = ptanh[i][j];
  offset[j][i] = offset[i][j];
  cut[j][i] = cut[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  if (tail_flag) {
    int *type = atom->type;
    int nlocal = atom->nlocal;

    double count[2],all[2];
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {
      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);
  }

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::write_restart(FILE *fp)
{
  write_restart_settings(fp);

  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&htanh[i][j],sizeof(double),1,fp);
        fwrite(&rmh[i][j],sizeof(double),1,fp);
        fwrite(&sigmah[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::read_restart(FILE *fp)
{
  read_restart_settings(fp);
  allocate();

  int i,j;
  int me = comm->me;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      if (me == 0) fread(&setflag[i][j],sizeof(int),1,fp);
      MPI_Bcast(&setflag[i][j],1,MPI_INT,0,world);
      if (setflag[i][j]) {
        if (me == 0) {
          fread(&htanh[i][j],sizeof(double),1,fp);
          fread(&rmh[i][j],sizeof(double),1,fp);
          fread(&sigmah[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&htanh[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&rmh[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&sigmah[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::write_restart_settings(FILE *fp)
{
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairTanhlrCutLogLoop::read_restart_settings(FILE *fp)
{
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairTanhlrCutLogLoop::single(int i, int j, int itype, int jtype, double rsq,
                         double factor_coul, double factor_lj,
                         double &fforce)
{
  double r, rexp,utanh,phitanh, tanr;

  r=sqrt(rsq);

  int iglobal, jglobal, jmi;
  iglobal = atom->tag[i];
  jglobal = atom->tag[j];
  jmi = abs(jglobal-iglobal);
  double ptanh_gamma;
  if (jmi<=MaxLoop) {
    //ptanh_gamma = -ptanh[itype][jtype] * log(pow(jmi, (g_a-1.0)) * exp(-jmi/g_b) / g_norm);
    ptanh_gamma = ptanh[itype][jtype] - log(pow(jmi, (g_a-1.0)) * exp(-jmi/g_b) / g_norm);
  } else { ptanh_gamma = 0.0; }

  if (r <= rmh[itype][jtype]) {
      rexp = (rmh[itype][jtype]-r)*sigmah[itype][jtype];
      tanr = tanh(rexp);
      utanh = ptanh_gamma*(1.0 + tanr);
      fforce = factor_lj/r*ptanh_gamma*(1.0-tanr*tanr)*sigmah[itype][jtype];
  } else {
      utanh = ptanh_gamma*rmh4[itype][jtype] / rsq / rsq;
      // there is an extra r here which is to normalize position vector
      fforce = factor_lj/r*ptanh_gamma*rmh4[itype][jtype]* (4.0) /rsq/rsq/r;
  }

  phitanh = utanh - offset[itype][jtype] * ptanh_gamma;
  return factor_lj*phitanh;
}

/* ---------------------------------------------------------------------- */
double PairTanhlrCutLogLoop::memory_usage()
{
  const int n=atom->ntypes;

  double bytes = Pair::memory_usage();

  bytes += 7*((n+1)*(n+1) * sizeof(double) + (n+1)*sizeof(double *));
  bytes += 1*((n+1)*(n+1) * sizeof(int) + (n+1)*sizeof(int *));

  return bytes;
}
