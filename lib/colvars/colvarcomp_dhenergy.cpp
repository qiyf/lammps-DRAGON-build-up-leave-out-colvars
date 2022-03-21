#include <cmath>

#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvarparse.h"
#include "colvar.h"
#include "colvarcomp.h"

/*
 * Implement the Debye Huckel Electrostatic function here 
 * -------          Limitations         ------
 * Currently assume the charges are 1.0 for all the atoms, 
 * could easily generalize this by reading charges from external file if necessary
 */


colvar::dhenergy::dhenergy (std::string const &conf)
    : cvc (conf)
{
    function_type = "dhenergy";
    group1 = parse_group (conf, "group1");
    atom_groups.push_back (group1);
    group2 = parse_group (conf, "group2");
    atom_groups.push_back (group2);

    group1->b_center = false;
    group2->b_center = false;
          
    n_group1 = group1->size();
    n_group2 = group2->size();
    // allocate variables
    allocated = false;
    allocate();

    // The 4th argument is just a default value? 
    get_keyval (conf, "I", I, 0.1);
    get_keyval (conf, "TEMP", T, 300);
    get_keyval (conf, "Epsilon", epsilon, 80);
    get_keyval (conf, "Cutoff", cutoff, 40);

    cvm::real energyUnits=4.184;
    cvm::real lengthUnits=0.1;
    cvm::real timeUnits=0.001;
    constant=138.935458111/energyUnits/lengthUnits; 
    k= sqrt(I/(epsilon*T))*502.903741125*lengthUnits; 
    cvm::log ("Debye screening length: " + 
                cvm::to_str (1.0/k)+"\n");

    // read in the charges
    std::string charge_file_name;
    if ( get_keyval (conf, "group1Charge", charge_file_name, std::string("")) ) {

        std::ifstream in_charge(charge_file_name.c_str());
	if (!in_charge.good()) cvm::fatal_error ("Error: charge File for group1 can't be read!!! \n");
	for (int i=0;i<n_group1;++i)
            in_charge >> charge1[i];
                
	in_charge.close();
    } else {
        cvm::fatal_error ("Error: Charge file for group1 not set!!! \n");
    }
    if ( get_keyval (conf, "group2Charge", charge_file_name, std::string("")) ) {

        std::ifstream in_charge(charge_file_name.c_str());
	if (!in_charge.good()) cvm::fatal_error ("Error: charge File for group1 can't be read!!! \n");
	for (int j=0;j<n_group2;++j)
            in_charge >> charge2[j];
                
	in_charge.close();
    } else {
        cvm::fatal_error ("Error: Charge file for group2 not set!!! \n");
    }

    x.type (colvarvalue::type_scalar);
}


colvar::dhenergy::dhenergy()
    : cvc ()
{
  function_type = "dhenergy";
  x.type (colvarvalue::type_scalar);
}


void colvar::dhenergy::calc_value()
{
    x.real_value = 0.0;
    cvm::real r, invr;
    int i, j;

    for (i=0;i<n_group1;++i) {
	for (j=0;j<n_group2;++j) {
	        	
            cvm::rvector const dx = (*group1)[i].pos - (*group2)[j].pos;
	    
	    r = dx.norm();
            if ( r >= cutoff ) continue;
            invr = 1.0/r;

	    x.real_value += exp(-k*r)*invr*constant*charge1[i]*charge2[j]/epsilon;
	}
    }

}


void colvar::dhenergy::calc_gradients()
{

    cvm::real a, r, invr, force1, tmp, dtmp;
    int i, j;
    
    for (i=0;i<n_group1;++i) {
	for (j=0;j<n_group2;++j) {
	        	
            cvm::rvector const dx = (*group1)[i].pos - (*group2)[j].pos;
	    
	    r = dx.norm();
            if ( r >= cutoff ) continue;
            invr = 1.0/r;

	    tmp  = exp(-k*r)*invr*constant*charge1[i]*charge2[j]/epsilon;
            dtmp = -(k+invr)*tmp;

	    force1 = dtmp * invr;

            (*group1)[i].grad +=   force1 * dx;
            (*group2)[j].grad += - force1 * dx;
	}
    }

}


void colvar::dhenergy::calc_force_invgrads()
{
    cvm::fatal_error ("Error: Inverse gradients for dhenergy is not implemented !!! \n");
}


void colvar::dhenergy::calc_Jacobian_derivative()
{
    cvm::fatal_error ("Error: Jacobian for dhenergy is not implemented !!! \n");
}


void colvar::dhenergy::apply_force (colvarvalue const &force)
{
  if (!group1->noforce)
    group1->apply_colvar_force (force.real_value);

  if (!group2->noforce)
    group2->apply_colvar_force (force.real_value);
}

void colvar::dhenergy::allocate()
{
    charge1 = new cvm::real[n_group1];
    charge2 = new cvm::real[n_group2];

allocated = true;
}

colvar::dhenergy::~dhenergy()
{
	if (allocated) {

	    delete [] charge1;
	    delete [] charge2;
	}
}
