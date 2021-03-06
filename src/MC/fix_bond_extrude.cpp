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

#include <math.h>
#include <mpi.h>
#include <string.h>
#include <stdlib.h>
#include "fix_bond_extrude.h"
#include "update.h"
#include "respa.h"
#include "atom.h"
#include "atom_vec.h"
#include "force.h"
#include "pair.h"
#include "comm.h"
#include "neighbor.h"
#include "domain.h"
#include "neigh_list.h"
#include "neigh_request.h"
#include "random_mars.h"
#include "memory.h"
#include "error.h"

using namespace LAMMPS_NS;
using namespace FixConst;

#define BIG 1.0e20
#define DELTA 16

/* ---------------------------------------------------------------------- */

FixBondExtrude::FixBondExtrude(LAMMPS *lmp, int narg, char **arg) :
  Fix(lmp, narg, arg),
  bondcount(NULL), partner(NULL), finalpartner(NULL), distsq(NULL),
  probability(NULL), created(NULL), copy(NULL), random(NULL), list(NULL)
{
  if (narg < 5) error->all(FLERR,"Illegal fix bond/extrude command");

  MPI_Comm_rank(world,&me);
  MPI_Comm_size(world,&nprocs);

  nevery = force->inumeric(FLERR,arg[3]);
  if (nevery <= 0) error->all(FLERR,"Illegal fix bond/extrude command");

  force_reneighbor = 1;
  next_reneighbor = -1;
  vector_flag = 1;
  size_vector = 2;
  global_freq = 1;
  extvector = 0;

  btype = force->inumeric(FLERR,arg[4]);

  if (btype < 1 || btype > atom->nbondtypes)
    error->all(FLERR,"Invalid bond type in fix bond/extrude command");

  // optional keywords

  int iarg = 6;

  // error check

  if (atom->molecular != 1)
    error->all(FLERR,"Cannot use fix bond/extrude with non-molecular systems");

  // initialize Marsaglia RNG with processor-unique seed

  random = new RanMars(lmp,seed + me);

  // perform initial allocation of atom-based arrays
  // register with Atom class
  // bondcount values will be initialized in setup()

  bondcount = NULL;
  grow_arrays(atom->nmax);
  atom->add_callback(0);
  countflag = 0;

  // set comm sizes needed by this fix
  // forward is big due to comm of broken bonds and 1-2 neighbors

  comm_forward = MAX(2,2+atom->maxspecial);
  comm_reverse = 2;

  // allocate arrays local to this fix

  nmax = 0;
  partner = finalpartner = NULL;
  distsq = NULL;

  maxbreak = 0;
  maxcreate = 0;
  created = NULL;

  // copy = special list for one atom
  // size = ms^2 + ms is sufficient
  // b/c in rebuild_special_one() neighs of all 1-2s are added,
  //   then a dedup(), then neighs of all 1-3s are added, then final dedup()
  // this means intermediate size cannot exceed ms^2 + ms

  int maxspecial = atom->maxspecial;
  copy = new tagint[maxspecial*maxspecial + maxspecial];

  // zero out stats

  breakcount = 0;
  breakcounttotal = 0;
  createcount = 0;
  createcounttotal = 0;
}

/* ---------------------------------------------------------------------- */

FixBondExtrude::~FixBondExtrude()
{
  // unregister callbacks to this fix from Atom class

  atom->delete_callback(id,0);

  delete random;

  // delete locally stored arrays

  memory->destroy(bondcount);
  memory->destroy(partner);
  memory->destroy(finalpartner);
  memory->destroy(distsq);
  memory->destroy(broken);
  memory->destroy(created);
  delete [] copy;
}

/* ---------------------------------------------------------------------- */

int FixBondExtrude::setmask()
{
  int mask = 0;
  mask |= POST_INTEGRATE;
  mask |= POST_INTEGRATE_RESPA;
  return mask;
}

/* ---------------------------------------------------------------------- */

void FixBondExtrude::init()
{
  if (strstr(update->integrate_style,"respa"))
    nlevels_respa = ((Respa *) update->integrate)->nlevels;



  // need a half neighbor list, built every Nevery steps

  int irequest = neighbor->request(this,instance_me);
  neighbor->requests[irequest]->pair = 0;
  neighbor->requests[irequest]->fix = 1;
  neighbor->requests[irequest]->occasional = 1;

  lastcheck = -1;

  // DEBUG
  //print_bb();
}

/* ---------------------------------------------------------------------- */

void FixBondExtrude::init_list(int id, NeighList *ptr)
{
  list = ptr;
}

/* ---------------------------------------------------------------------- */

void FixBondExtrude::setup(int vflag)
{
  int i,j,m;

  // compute initial bondcount if this is first run
  // can't do this earlier, in constructor or init, b/c need ghost info

  if (countflag) return;
  countflag = 1;

  // count bonds stored with each bond I own
  // if newton bond is not set, just increment count on atom I
  // if newton bond is set, also increment count on atom J even if ghost
  // bondcount is long enough to tally ghost atom counts

  int *num_bond = atom->num_bond;
  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int nlocal = atom->nlocal;
  int nghost = atom->nghost;
  int nall = nlocal + nghost;
  int newton_bond = force->newton_bond;

  for (i = 0; i < nall; i++) bondcount[i] = 0;

  for (i = 0; i < nlocal; i++)
    for (j = 0; j < num_bond[i]; j++) {
      if (bond_type[i][j] == btype) {
        bondcount[i]++;
        if (newton_bond) {
          m = atom->map(bond_atom[i][j]);
          if (m < 0)
            error->one(FLERR,"Fix bond/create needs ghost atoms "
                       "from further away");
          bondcount[m]++;
        }
      }
    }

  // if newton_bond is set, need to sum bondcount

  commflag = 1;
  if (newton_bond) comm->reverse_comm_fix(this,1);
}

/* ---------------------------------------------------------------------- */

void FixBondExtrude::post_integrate()
{
  int i,j,k,m,n,i1,i2,n1,n3,type;
  double delx,dely,delz,rsq;
  tagint *slist;

  if (update->ntimestep % nevery) return;

  // check that all procs have needed ghost atoms within ghost cutoff
  // only if neighbor list has changed since last check

  if (lastcheck < neighbor->lastcall) check_ghosts();

  // acquire updated ghost atom positions
  // necessary b/c are calling this after integrate, but before Verlet comm

  comm->forward_comm();

  // forward comm of bondcount, so ghosts have it; copied from create/bond

  commflag = 1;
  comm->forward_comm_fix(this,1);

  // resize bond partner list and initialize it
  // probability array overlays distsq array
  // needs to be atom->nmax in length

  if (atom->nmax > nmax) {
    memory->destroy(partner);
    memory->destroy(break_partner);
    memory->destroy(create_partner);
    memory->destroy(finalpartner);
    memory->destroy(distsq);
    nmax = atom->nmax;
    memory->create(break_partner,nmax,"bond/extrude:break_partner");
    memory->create(create_partner,nmax,"bond/extrude:create_partner");
    memory->create(partner,nmax,"bond/extrude:partner");
    memory->create(finalpartner,nmax,"bond/extrude:finalpartner");
    memory->create(distsq,nmax,"bond/extrude:distsq");
    probability = distsq;
  }

  int nlocal = atom->nlocal;
  int nall = atom->nlocal + atom->nghost;

  for (i = 0; i < nall; i++) {
    break_partner[i] = 0;
    create_partner[i] = 0;
    partner[i] = 0;
    finalpartner[i] = 0;
  }

  // loop over bond list
  // setup possible partner list of bonds to break

  double **x = atom->x;
  tagint *tag = atom->tag;
  int *mask = atom->mask;
  int **bondlist = neighbor->bondlist;
  int nbondlist = neighbor->nbondlist;

  for (n = 0; n < nbondlist; n++) {
    i1 = bondlist[n][0];
    i2 = bondlist[n][1];
    type = bondlist[n][2];
    if (!(mask[i1] & groupbit)) continue;
    if (!(mask[i2] & groupbit)) continue;
    if (type != btype) continue;

    if (tag[i2]>tag[i1]) {
        atomr = tag[i2];
        atoml = tag[i1];
        if (type[i2] !=2) && (type[atom->map(tag[i2]+1)] != 2) atomr = tag[i2]+1;
        if (type[i1] !=1) && (type[atom->map(tag[i1]-1)] != 1) atoml = tag[i1]-1; 

        if ( (atomr != tag[i2]) || (atoml != tag[i1]) ) {
            create_partner[atom->map(atoml)] = atomr;
            create_partner[atom->map(atomr)] = atoml;

            break_partner[i1] = tag[i2];
            break_partner[i2] = tag[i1];
        }
    } else {
        atomr = tag[i1];
        atoml = tag[i2];
        if (type[i1] !=2) && (type[atom->map(tag[i1]+1)] != 2) atomr = tag[i1]+1;
        if (type[i2] !=1) && (type[atom->map(tag[i2]-1)] != 1) atoml = tag[i2]-1; 

        if ( (atomr != tag[i1]) || (atoml != tag[i2]) ) {
            create_partner[atom->map(atoml)] = atomr;
            create_partner[atom->map(atomr)] = atoml;

            break_partner[i1] = tag[i2];
            break_partner[i2] = tag[i1];
        }
    }
  }

  // first do the bond breaking
  for (i = 0; i < nall; i++) {
    partner[i] = break_partner[i];
    finalpartner[i] = 0;
  }
  // reverse comm of partner info; the following two communication are from 
  // bond/break, and the flag needs to increase by 1

  commflag = 2;
  if (force->newton_bond) comm->reverse_comm_fix(this);

  commflag = 2;
  comm->forward_comm_fix(this,2);

  // break bonds
  // if both atoms list each other as winning bond partner
  // and probability constraint is satisfied

  int **bond_type = atom->bond_type;
  tagint **bond_atom = atom->bond_atom;
  int *num_bond = atom->num_bond;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  nbreak = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    // delete bond from atom I if I stores it
    // atom J will also do this

    for (m = 0; m < num_bond[i]; m++) {
      if (bond_atom[i][m] == partner[i]) {
        for (k = m; k < num_bond[i]-1; k++) {
          bond_atom[i][k] = bond_atom[i][k+1];
          bond_type[i][k] = bond_type[i][k+1];
        }
        num_bond[i]--;
        break;
      }
    }

    // remove J from special bond list for atom I
    // atom J will also do this, whatever proc it is on

    slist = special[i];
    n1 = nspecial[i][0];
    for (m = 0; m < n1; m++)
      if (slist[m] == partner[i]) break;
    n3 = nspecial[i][2];
    for (; m < n3-1; m++) slist[m] = slist[m+1];
    nspecial[i][0]--;
    nspecial[i][1]--;
    nspecial[i][2]--;

    // store final broken bond partners and count the broken bond once

    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j]) nbreak++;
  }

  // tally stats

  MPI_Allreduce(&nbreak,&breakcount,1,MPI_INT,MPI_SUM,world);
  breakcounttotal += breakcount;
  atom->nbonds -= breakcount;

  // trigger reneighboring if any bonds were broken
  // this insures neigh lists will immediately reflect the topology changes
  // done if no bonds broken

  if (breakcount) next_reneighbor = update->ntimestep;
  if (!breakcount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect broken bonds; 
  // comm from bond/break; flag needs to increaase by 1

  commflag = 3;
  comm->forward_comm_fix(this);

  // create list of broken bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner is now set for owned and ghost atoms so loop over nall
  // OK if duplicates in broken list due to ghosts duplicating owned atoms
  // check J < 0 to insure a broken bond to unknown atom is included
  //   i.e. bond partner outside of cutoff length

  nbreak = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (nbreak == maxbreak) {
        maxbreak += DELTA;
        memory->grow(broken,maxbreak,2,"bond/break:broken");
      }
      broken[nbreak][0] = tag[i];
      broken[nbreak][1] = finalpartner[i];
      nbreak++;
    }
  }

  // update special neigh lists of all atoms affected by any broken bond
  // also remove angles/dihedrals/impropers broken by broken bonds

  update_topology_break();

  //
  // now we create bonds
  // 

  for (i = 0; i < nall; i++) {
    partner[i] = create_partner[i];
    finalpartner[i] = 0;
  }

  commflag = 2;
  if (force->newton_pair) comm->reverse_comm_fix(this);

  commflag = 2;
  comm->forward_comm_fix(this,2);

  // create bonds for atoms I own
  // only if both atoms list each other as winning bond partner
  //   and probability constraint is satisfied
  // if other atom is owned by another proc, it should do same thing

  int newton_bond = force->newton_bond;

  ncreate = 0;
  for (i = 0; i < nlocal; i++) {
    if (partner[i] == 0) continue;
    j = atom->map(partner[i]);
    if (partner[j] != tag[i]) continue;

    // if newton_bond is set, only store with I or J
    // if not newton_bond, store bond with both I and J
    // atom J will also do this consistently, whatever proc it is on

    if (!newton_bond || tag[i] < tag[j]) {
      if (num_bond[i] == atom->bond_per_atom)
        error->one(FLERR,"New bond exceeded bonds per atom in fix bond/create");
      bond_type[i][num_bond[i]] = btype;
      bond_atom[i][num_bond[i]] = tag[j];
      num_bond[i]++;
    }

    // add a 1-2 neighbor to special bond list for atom I
    // atom J will also do this, whatever proc it is on
    // need to first remove tag[j] from later in list if it appears
    // prevents list from overflowing, will be rebuilt in rebuild_special_one()

    slist = special[i];
    n1 = nspecial[i][0];
    n2 = nspecial[i][1];
    n3 = nspecial[i][2];
    for (m = n1; m < n3; m++)
      if (slist[m] == tag[j]) break;
    if (m < n3) {
      for (n = m; n < n3-1; n++) slist[n] = slist[n+1];
      n3--;
      if (m < n2) n2--;
    }
    if (n3 == atom->maxspecial)
      error->one(FLERR,
                 "New bond exceeded special list size in fix bond/create");
    for (m = n3; m > n1; m--) slist[m] = slist[m-1];
    slist[n1] = tag[j];
    nspecial[i][0] = n1+1;
    nspecial[i][1] = n2+1;
    nspecial[i][2] = n3+1;

    // increment bondcount, convert atom to new type if limit reached
    // atom J will also do this, whatever proc it is on

    bondcount[i]++;
    if (type[i] == iatomtype) {
      if (bondcount[i] == imaxbond) type[i] = inewtype;
    } else {
      if (bondcount[i] == jmaxbond) type[i] = jnewtype;
    }

    // store final created bond partners and count the created bond once

    finalpartner[i] = tag[j];
    finalpartner[j] = tag[i];
    if (tag[i] < tag[j]) ncreate++;
  }

  // tally stats

  MPI_Allreduce(&ncreate,&createcount,1,MPI_INT,MPI_SUM,world);
  createcounttotal += createcount;
  atom->nbonds += createcount;

  // trigger reneighboring if any bonds were formed
  // this insures neigh lists will immediately reflect the topology changes
  // done if any bonds created

  if (createcount) next_reneighbor = update->ntimestep;
  if (!createcount) return;

  // communicate final partner and 1-2 special neighbors
  // 1-2 neighs already reflect created bonds

  commflag = 3;
  comm->forward_comm_fix(this);

  // create list of broken bonds that influence my owned atoms
  //   even if between owned-ghost or ghost-ghost atoms
  // finalpartner is now set for owned and ghost atoms so loop over nall
  // OK if duplicates in broken list due to ghosts duplicating owned atoms
  // check J < 0 to insure a broken bond to unknown atom is included
  //   i.e. a bond partner outside of cutoff length

  ncreate = 0;
  for (i = 0; i < nall; i++) {
    if (finalpartner[i] == 0) continue;
    j = atom->map(finalpartner[i]);
    if (j < 0 || tag[i] < tag[j]) {
      if (ncreate == maxcreate) {
        maxcreate += DELTA;
        memory->grow(created,maxcreate,2,"bond/create:created");
      }
      created[ncreate][0] = tag[i];
      created[ncreate][1] = finalpartner[i];
      ncreate++;
    }
  }

  // update special neigh lists of all atoms affected by any created bond
  // also add angles/dihedrals/impropers induced by created bonds

  update_topology_create();


  // DEBUG
  // print_bb();
}

/* ----------------------------------------------------------------------
   insure all atoms 2 hops away from owned atoms are in ghost list
   this allows dihedral 1-2-3-4 to be properly deleted
     and special list of 1 to be properly updated
   if I own atom 1, but not 2,3,4, and bond 3-4 is deleted
     then 2,3 will be ghosts and 3 will store 4 as its finalpartner

     The same as in create bond. no need to redefine.
------------------------------------------------------------------------- */

void FixBondExtrude::check_ghosts()
{
  int i,j,n;
  tagint *slist;

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  int flag = 0;
  for (i = 0; i < nlocal; i++) {
    slist = special[i];
    n = nspecial[i][1];
    for (j = 0; j < n; j++)
      if (atom->map(slist[j]) < 0) flag = 1;
  }

  int flagall;
  MPI_Allreduce(&flag,&flagall,1,MPI_INT,MPI_SUM,world);
  if (flagall)
    error->all(FLERR,"Fix bond/break needs ghost atoms from further away");
  lastcheck = update->ntimestep;
}

/* ----------------------------------------------------------------------
   double loop over my atoms and broken bonds
   influenced = 1 if atom's topology is affected by any broken bond
     yes if is one of 2 atoms in bond
     yes if both atom IDs appear in atom's special list
     else no
   if influenced:
     check for angles/dihedrals/impropers to break due to specific broken bonds
     rebuild the atom's special list of 1-2,1-3,1-4 neighs
------------------------------------------------------------------------- */

void FixBondExtrude::update_topology_break()
{
  int i,j,k,n,influence,influenced,found;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  nangles = 0;
  ndihedrals = 0;
  nimpropers = 0;

  //printf("NBREAK %d: ",nbreak);
  //for (i = 0; i < nbreak; i++)
  //  printf(" %d %d,",broken[i][0],broken[i][1]);
  //printf("\n");

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < nbreak; j++) {
      id1 = broken[j][0];
      id2 = broken[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2) influence = 1;
      else {
        n = nspecial[i][2];
        found = 0;
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) found++;
        if (found == 2) influence = 1;
      }
      if (!influence) continue;
      influenced = 1;

    }

    if (influenced) rebuild_special_one(i);
  }

  int newton_bond = force->newton_bond;

  int all;
}

void FixBondExtrude::update_topology_create()
{
  int i,j,k,n,influence,influenced;
  tagint id1,id2;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;
  int nlocal = atom->nlocal;

  nangles = 0;
  ndihedrals = 0;
  nimpropers = 0;
  overflow = 0;

  //printf("NCREATE %d: ",ncreate);
  //for (i = 0; i < ncreate; i++)
  //  printf(" %d %d,",created[i][0],created[i][1]);
  //printf("\n");

  for (i = 0; i < nlocal; i++) {
    influenced = 0;
    slist = special[i];

    for (j = 0; j < ncreate; j++) {
      id1 = created[j][0];
      id2 = created[j][1];

      influence = 0;
      if (tag[i] == id1 || tag[i] == id2) influence = 1;
      else {
        n = nspecial[i][1];
        for (k = 0; k < n; k++)
          if (slist[k] == id1 || slist[k] == id2) {
            influence = 1;
            break;
          }
      }
      if (!influence) continue;
      influenced = 1;
    }

    // rebuild_special_one() first, since used by create_angles, etc

    if (influenced) {
      rebuild_special_one(i);
    }
  }

  int overflowall;
  MPI_Allreduce(&overflow,&overflowall,1,MPI_INT,MPI_SUM,world);
  if (overflowall) error->all(FLERR,"Fix bond/create induced too many "
                              "angles/dihedrals/impropers per atom");

}

/* ----------------------------------------------------------------------
   re-build special list of atom M
   does not affect 1-2 neighs (already include effects of new bond)
   affects 1-3 and 1-4 neighs due to other atom's augmented 1-2 neighs

   Create and Break are essentially the same. Using the create version, which
   has a few extra error messages
------------------------------------------------------------------------- */

void FixBondExtrude::rebuild_special_one(int m)
{
  int i,j,n,n1,cn1,cn2,cn3;
  tagint *slist;

  tagint *tag = atom->tag;
  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  // existing 1-2 neighs of atom M

  slist = special[m];
  n1 = nspecial[m][0];
  cn1 = 0;
  for (i = 0; i < n1; i++)
    copy[cn1++] = slist[i];

  // new 1-3 neighs of atom M, based on 1-2 neighs of 1-2 neighs
  // exclude self
  // remove duplicates after adding all possible 1-3 neighs

  cn2 = cn1;
  for (i = 0; i < cn1; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn2++] = slist[j];
  }

  cn2 = dedup(cn1,cn2,copy);
  if (cn2 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/create");

  // new 1-4 neighs of atom M, based on 1-2 neighs of 1-3 neighs
  // exclude self
  // remove duplicates after adding all possible 1-4 neighs

  cn3 = cn2;
  for (i = cn1; i < cn2; i++) {
    n = atom->map(copy[i]);
    if (n < 0)
      error->one(FLERR,"Fix bond/create needs ghost atoms from further away");
    slist = special[n];
    n1 = nspecial[n][0];
    for (j = 0; j < n1; j++)
      if (slist[j] != tag[m]) copy[cn3++] = slist[j];
  }

  cn3 = dedup(cn2,cn3,copy);
  if (cn3 > atom->maxspecial)
    error->one(FLERR,"Special list size exceeded in fix bond/create");

  // store new special list with atom M

  nspecial[m][0] = cn1;
  nspecial[m][1] = cn2;
  nspecial[m][2] = cn3;
  memcpy(special[m],copy,cn3*sizeof(int));
}

/* ----------------------------------------------------------------------
   remove all ID duplicates in copy from Nstart:Nstop-1
   compare to all previous values in copy
   return N decremented by any discarded duplicates

   The same for both create and break
------------------------------------------------------------------------- */

int FixBondExtrude::dedup(int nstart, int nstop, tagint *copy)
{
  int i;

  int m = nstart;
  while (m < nstop) {
    for (i = 0; i < m; i++)
      if (copy[i] == copy[m]) {
        copy[m] = copy[nstop-1];
        nstop--;
        break;
      }
    if (i == m) m++;
  }

  return nstop;
}

/* ---------------------------------------------------------------------- 
 * The same for break and create
 * */

void FixBondExtrude::post_integrate_respa(int ilevel, int iloop)
{
  if (ilevel == nlevels_respa-1) post_integrate();
}

/* ---------------------------------------------------------------------- 
 * Use the create version which has more option.
 * Now commflag=2 for break should be commflag=3
 * */

int FixBondExtrude::pack_forward_comm(int n, int *list, double *buf,
                                     int pbc_flag, int *pbc)
{
  int i,j,k,m,ns;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(bondcount[j]).d;
    }
    return m;
  }

  if (commflag == 2) {
    for (i = 0; i < n; i++) {
      j = list[i];
      buf[m++] = ubuf(partner[j]).d;
      buf[m++] = probability[j];
    }
    return m;
  }

  int **nspecial = atom->nspecial;
  tagint **special = atom->special;

  m = 0;
  for (i = 0; i < n; i++) {
    j = list[i];
    buf[m++] = ubuf(finalpartner[j]).d;
    ns = nspecial[j][0];
    buf[m++] = ubuf(ns).d;
    for (k = 0; k < ns; k++)
      buf[m++] = ubuf(special[j][k]).d;
  }
  return m;
}

/* ---------------------------------------------------------------------- 
 * Using the create version. need to be careful with the flag
 * */

void FixBondExtrude::unpack_forward_comm(int n, int first, double *buf)
{
  int i,j,m,ns,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      bondcount[i] = (int) ubuf(buf[m++]).i;

  } else if (commflag == 2) {
    for (i = first; i < last; i++) {
      partner[i] = (tagint) ubuf(buf[m++]).i;
      probability[i] = buf[m++];
    }

  } else {
    int **nspecial = atom->nspecial;
    tagint **special = atom->special;

    m = 0;
    last = first + n;
    for (i = first; i < last; i++) {
      finalpartner[i] = (tagint) ubuf(buf[m++]).i;
      ns = (int) ubuf(buf[m++]).i;
      nspecial[i][0] = ns;
      for (j = 0; j < ns; j++)
        special[i][j] = (tagint) ubuf(buf[m++]).i;
    }
  }
}

/* ---------------------------------------------------------------------- 
 * Using create version. be careful with the flag
 * */

int FixBondExtrude::pack_reverse_comm(int n, int first, double *buf)
{
  int i,m,last;

  m = 0;
  last = first + n;

  if (commflag == 1) {
    for (i = first; i < last; i++)
      buf[m++] = ubuf(bondcount[i]).d;
    return m;
  }

  for (i = first; i < last; i++) {
    buf[m++] = ubuf(partner[i]).d;
    buf[m++] = distsq[i];
  }
  return m;
}

/* ---------------------------------------------------------------------- */

void FixBondExtrude::unpack_reverse_comm(int n, int *list, double *buf)
{
  int i,j,m;

  m = 0;

  if (commflag == 1) {
    for (i = 0; i < n; i++) {
      j = list[i];
      bondcount[j] += (int) ubuf(buf[m++]).i;
    }

  } else {
    for (i = 0; i < n; i++) {
      j = list[i];
      if (buf[m+1] < distsq[j]) {
        partner[j] = (tagint) ubuf(buf[m++]).i;
        distsq[j] = buf[m++];
      } else m += 2;
    }
  }
}

/* ----------------------------------------------------------------------
   allocate local atom-based arrays
------------------------------------------------------------------------- */

void FixBondExtrude::grow_arrays(int nmax)
{
  memory->grow(bondcount,nmax,"bond/create:bondcount");
}

/* ----------------------------------------------------------------------
   copy values within local atom-based arrays
------------------------------------------------------------------------- */

void FixBondExtrude::copy_arrays(int i, int j, int delflag)
{
  bondcount[j] = bondcount[i];
}

/* ----------------------------------------------------------------------
   pack values in local atom-based arrays for exchange with another proc
------------------------------------------------------------------------- */

int FixBondExtrude::pack_exchange(int i, double *buf)
{
  buf[0] = bondcount[i];
  return 1;
}

/* ----------------------------------------------------------------------
   unpack values in local atom-based arrays from exchange with another proc
------------------------------------------------------------------------- */

int FixBondExtrude::unpack_exchange(int nlocal, double *buf)
{
  bondcount[nlocal] = static_cast<int> (buf[0]);
  return 1;
}

/* ---------------------------------------------------------------------- 
 * Presumably this is not important; 
 * */

double FixBondExtrude::compute_vector(int n)
{
  if (n == 0) return (double) breakcount;
  return (double) breakcounttotal;
}

/* ----------------------------------------------------------------------
   memory usage of local atom-based arrays

   Assuming that this is not important
------------------------------------------------------------------------- */

double FixBondExtrude::memory_usage()
{
  int nmax = atom->nmax;
  double bytes = 2*nmax * sizeof(tagint);
  bytes += nmax * sizeof(double);
  return bytes;
}
