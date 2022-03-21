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
#include "pair_harmonic_cut.h"
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

PairHarmonicCut::PairHarmonicCut(LAMMPS *lmp) : Pair(lmp)
{
  respa_enable = 0;
  writedata = 1;
}

/* ---------------------------------------------------------------------- */

PairHarmonicCut::~PairHarmonicCut()
{
  if (allocated) {
    memory->destroy(setflag);
    memory->destroy(k_param);
    memory->destroy(r0_param);

    // memory->destroy(cutsq);
    memory->destroy(cut);
    memory->destroy(offset);
  }
}

/* ---------------------------------------------------------------------- */

void PairHarmonicCut::compute(int eflag, int vflag)
{
printf("start compute\n");
  int i,j,ii,jj,inum,jnum,itype,jtype,imol,jmol;
  double xtmp,ytmp,ztmp,delx,dely,delz,evdwl,fpair;
  double rsq,r,dr,rk,factor_lj;
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

  // loop over neighbors of atoms
  for (ii = 0; ii < inum; ii++) {
    i = ilist[ii];
    xtmp = x[i][0];
    ytmp = x[i][1];
    ztmp = x[i][2];
    itype = type[i];
    jlist = firstneigh[i];
    jnum = numneigh[i];
    // 
    // imol = atom->molecule[i];

    for (jj = 0; jj < jnum; jj++) {
      j = jlist[jj];
      factor_lj = special_lj[sbmask(j)];
      j &= NEIGHMASK;

      delx = xtmp - x[j][0];
      dely = ytmp - x[j][1];
      delz = ztmp - x[j][2];
      rsq = delx*delx + dely*dely + delz*delz;
      jtype = type[j];
      // 
      // jmol = atom->molecule[j];

      r = sqrt(rsq);
      dr = r-r0_param[itype][jtype];
      rk = k_param[itype][jtype]*dr;

      if (r>0.0) fpair = -2.0*rk/r;
      else fpair = 0.0;

      // i and j are index that start with zeor, so we need to use floor
      // i and j are only local index; use tag to map them back to global
//       int iglobal, jglobal, jmi;

//       iglobal = atom->tag[i];
//       jglobal = atom->tag[j];
//       jmi = abs(jglobal-iglobal);

//       // not the same molecule
//       if ( (same_mol_flag) && (imol != jmol) ) continue;

//       if ( (rsq < cutsq[itype][jtype]) && (jmi<nideal) ){
//         r = sqrt(rsq);

//         double ideal_alpha;
//         ideal_alpha = ideal_potential[jmi][active[iglobal-1]-1][active[jglobal-1]-1];

		// // debug //
		// printf("iglobal %d, jglobal %d \n", iglobal, jglobal);
		// printf("jmi %d\n", jmi);
		// printf("activei %d, activej %d \n",active[iglobal-1]-1,active[jglobal-1]-1);
		// printf("second ideal, %f \n\n", ideal_alpha);
		// // 

//         if (r <= rmh[itype][jtype]) { 
//             rexp = (rmh[itype][jtype]-r)*sigmah[itype][jtype];
//             tanr = tanh(rexp);
//             utanh = ideal_alpha*(1.0+ tanr);
//             // p is 0.5 * h, the extra negative sign is taken care in f
//             fpair = factor_lj/r*ideal_alpha*sigmah[itype][jtype]*(1-tanr*tanr);
//         } else {
//             utanh = ideal_alpha*rmh4[itype][jtype] / rsq / rsq;
//             // there is an extra r here which is to normalize position vector
//             fpair = factor_lj/r*ideal_alpha*rmh4[itype][jtype]* (4.0) /rsq/rsq/r;
//         }

		f[i][0] += delx*fpair;
		f[i][1] += dely*fpair;
		f[i][2] += delz*fpair;

        if (newton_pair || j < nlocal) {
          f[j][0] -= delx*fpair;
          f[j][1] -= dely*fpair;
          f[j][2] -= delz*fpair;
        }

        if (eflag) {
          evdwl = rk*dr - offset[itype][jtype];
          evdwl *= factor_lj;
        }

        if (evflag) ev_tally(i,j,nlocal,newton_pair,
                             evdwl,0.0,fpair,delx,dely,delz);
      }
    }

  if (vflag_fdotr) virial_fdotr_compute();
 printf("end compute\n");
}


/* ----------------------------------------------------------------------
   allocate all arrays
------------------------------------------------------------------------- */

void PairHarmonicCut::allocate()
{
printf("start allocate\n");
  allocated = 1;
  int n = atom->ntypes;

  memory->create(setflag,n+1,n+1,"pair:setflag");
  for (int i = 1; i <= n; i++)
    for (int j = i; j <= n; j++)
      setflag[i][j] = 0;

  memory->create(k_param,n+1,n+1,"pair:k_param");
  memory->create(r0_param,n+1,n+1,"pair:r0_param");
  memory->create(offset,n+1,n+1,"pair:offset");
  memory->create(cut,n+1,n+1,"pair:cut");

  // memory->create(cutsq,n+1,n+1,"pair:cutsq");
  // memory->create(htanh,n+1,n+1,"pair:htanh");
  // memory->create(sigmah,n+1,n+1,"pair:sigmah");
  // memory->create(rmh,n+1,n+1,"pair:rmh");
  // memory->create(rmh4,n+1,n+1,"pair:rmh4");
  // memory->create(ptanh,n+1,n+1,"pair:ptanh");
  
printf("end allocate\n");
}

/* ----------------------------------------------------------------------
   global settings
------------------------------------------------------------------------- */

void PairHarmonicCut::settings(int narg, char **arg)
{
printf("start settings\n");
// format for the ideal potential
// start with 0, 
// 0    0.1
// 1    0.1
// 2    0.1 

// the first argument is the cutoff

  if (narg != 1) error->all(FLERR,"Illegal pair_style command");

  cut_global = force->numeric(FLERR,arg[0]);

  if (allocated) {
    int i,j;
    for (i = 1; i <= atom->ntypes; i++)
      for (j = i+1; j <= atom->ntypes; j++)
        if (setflag[i][j]) cut[i][j] = cut_global;
  }
printf("end settings\n");
}

/* ----------------------------------------------------------------------
   set coeffs for one or more type pairs
------------------------------------------------------------------------- */

void PairHarmonicCut::coeff(int narg, char **arg)
{
  printf("start coeff\n");
  if (narg != 4) error->all(FLERR,"Incorrect args for harmonic pair coefficients");
  if (!allocated) allocate();

  int ilo,ihi,jlo,jhi;
  force->bounds(FLERR, arg[0],atom->ntypes,ilo,ihi);
  force->bounds(FLERR, arg[1],atom->ntypes,jlo,jhi);

  double k_one = force->numeric(FLERR,arg[2]);
  double r0_one = force->numeric(FLERR,arg[3]);
  printf("k_one is %f, r0_one is %f\n", k_one,r0_one);

  int count = 0;
  for (int i = ilo; i <= ihi; i++) {
    for (int j = MAX(jlo,i); j <= jhi; j++) {

      k_param[i][j] = k_one;
      r0_param[i][j] = r0_one;
      cut[i][j] = cut_global;
      setflag[i][j] = 1;
      count++;
    }
  }

  // printf("%d,%d, %d,%d,%d\n", ilo,ihi,jlo,jhi,count);

  if (count == 0) error->all(FLERR,"Incorrect args for pair coefficients");
  printf("end coeff\n");
}

/* ----------------------------------------------------------------------
   init for one type pair i,j and corresponding j,i
------------------------------------------------------------------------- */
// void PairHarmonicCut::init_style()
// {
//   neighbor->request(this,instance_me);
// }

double PairHarmonicCut::init_one(int i, int j)
{
printf("start init_one\n");
printf("i is %d, j is %d\n", i,j);
  if (setflag[i][j] == 0) {
    error->all(FLERR,"All pair coeffs are not set");
    cut[i][j] = mix_distance(cut[i][i],cut[j][j]);
  }

  offset[i][j] = 0.0;
  // cut[i][j] = 100.0;
  // printf("offset is %f\n", offset[i][j]);

  k_param[j][i] = k_param[i][j];
  r0_param[j][i] = r0_param[i][j];
  offset[j][i] = offset[i][j];
  cut[j][i] = cut[i][j];

  // compute I,J contribution to long-range tail correction
  // count total # of atoms of type I and J via Allreduce

  printf("reach here?\n");
  printf("tail_flag is %d\n", tail_flag);
  if (tail_flag) {
    printf("here1\n");
    int *type = atom->type;
    int nlocal = atom->nlocal;

    printf("here2\n");
    double count[2],all[2];
    printf("here3\n");
    count[0] = count[1] = 0.0;
    for (int k = 0; k < nlocal; k++) {

      if (type[k] == i) count[0] += 1.0;
      if (type[k] == j) count[1] += 1.0;
    }
    MPI_Allreduce(count,all,2,MPI_DOUBLE,MPI_SUM,world);
  }
  	printf("cut is %f\n",cut[i][j]);
    printf("%d\n", atom->ntypes);
    printf("end init_one\n");

  return cut[i][j];
}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHarmonicCut::write_restart(FILE *fp)
{
  printf("here111\n");
  write_restart_settings(fp);
  int i,j;
  for (i = 1; i <= atom->ntypes; i++)
    for (j = i; j <= atom->ntypes; j++) {
      fwrite(&setflag[i][j],sizeof(int),1,fp);
      if (setflag[i][j]) {
        fwrite(&k_param[i][j],sizeof(double),1,fp);
        fwrite(&r0_param[i][j],sizeof(double),1,fp);
        fwrite(&cut[i][j],sizeof(double),1,fp);
      }
    }
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHarmonicCut::read_restart(FILE *fp)
{
  printf("here222\n");
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
          fread(&k_param[i][j],sizeof(double),1,fp);
          fread(&r0_param[i][j],sizeof(double),1,fp);
          fread(&cut[i][j],sizeof(double),1,fp);
        }
        MPI_Bcast(&k_param[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&r0_param[i][j],1,MPI_DOUBLE,0,world);
        MPI_Bcast(&cut[i][j],1,MPI_DOUBLE,0,world);
      }
    }

}

/* ----------------------------------------------------------------------
   proc 0 writes to restart file
------------------------------------------------------------------------- */

void PairHarmonicCut::write_restart_settings(FILE *fp)
{
  printf("here333\n");
  fwrite(&cut_global,sizeof(double),1,fp);
  fwrite(&offset_flag,sizeof(int),1,fp);
  fwrite(&mix_flag,sizeof(int),1,fp);
  fwrite(&tail_flag,sizeof(int),1,fp);
}

/* ----------------------------------------------------------------------
   proc 0 reads from restart file, bcasts
------------------------------------------------------------------------- */

void PairHarmonicCut::read_restart_settings(FILE *fp)
{
  printf("here444\n");
  int me = comm->me;
  if (me == 0) {
    fread(&cut_global,sizeof(double),1,fp);
    fread(&offset_flag,sizeof(int),1,fp);
    fread(&mix_flag,sizeof(int),1,fp);
    fread(&tail_flag,sizeof(int),1,fp);
  }
  MPI_Bcast(&cut_global,1,MPI_DOUBLE,0,world);
  MPI_Bcast(&offset_flag,1,MPI_INT,0,world);
  MPI_Bcast(&mix_flag,1,MPI_INT,0,world);
  MPI_Bcast(&tail_flag,1,MPI_INT,0,world);
}

/* ---------------------------------------------------------------------- */

double PairHarmonicCut::single(int i, int j, int itype, int jtype, double rsq,
								double &fforce)
{
  printf("here555\n");
  double r=sqrt(rsq);
  double dr = r-r0_param[itype][jtype];
  double rk = k_param[itype][jtype] * dr;
  fforce = 0.0;
  if (r> 0.0) fforce = -2.0*rk/r;
  return rk*dr;
}

/* ---------------------------------------------------------------------- */
double PairHarmonicCut::memory_usage()
{
  printf("here666\n");
  const int n=atom->ntypes;

  double bytes = Pair::memory_usage();

  bytes += 7*((n+1)*(n+1) * sizeof(double) + (n+1)*sizeof(double *));
  bytes += 1*((n+1)*(n+1) * sizeof(int) + (n+1)*sizeof(int *));

  return bytes;
}
