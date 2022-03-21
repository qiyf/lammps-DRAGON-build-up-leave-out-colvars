#include <cmath>

#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvarparse.h"
#include "colvar.h"
#include "colvarcomp.h"

/*
 * Implement the qinterface function here 
 */


colvar::qinterface::qinterface (std::string const &conf)
    : cvc (conf)
{
    function_type = "qinterface";
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

    // No exponential flag for interface q
    qbias_flag      = 0;
    qbias_exp_flag  = 0;
    qobias_flag     = 0;
    qobias_exp_flag = 0;

    // get qbias flag
    get_keyval (conf, "qbias",      qbias_flag, false);
    if (qbias_flag) cvm::log("QBias flag on\n");

    get_keyval (conf, "qobias",     qobias_flag, false);
    if (qobias_flag) cvm::log("QOBias flag on\n");

    // The 4th argument is just a default value? 
    cutoff_flag = get_keyval (conf, "cutoff", cutoff, 10.0);
    if (!cutoff_flag && (qobias_flag || qobias_exp_flag) )
        cvm::fatal_error ("Error: Cutoff distance not found!!! \n");

    if (qbias_exp_flag || qobias_exp_flag) 
        get_keyval (conf, "sigmaExp", sigma_exp, 0.15);

    if (qbias_flag || qobias_flag) 
        get_keyval (conf, "sigma", sigma, 3.0);

    // read in native distance
    std::string native_file_name;
    if ( get_keyval (conf, "nativeFile", native_file_name, std::string("")) ) {

        std::ifstream in_rnative(native_file_name.c_str());
	if (!in_rnative.good()) cvm::fatal_error ("Error: File rnative.dat can't be read!!! \n");
        for (int i=0;i<n_group1;++i) {
	    for (int j=0;j<n_group2;++j) {
                in_rnative >> rN[i][j];
            }
        }
	in_rnative.close();
    } else {
        cvm::fatal_error ("Error: Native file not set!!! \n");
    }

    // Minimal sequence separation
    min_sep = 3;
    if (qobias_flag || qobias_exp_flag) min_sep=4;
    
    sigma_sq = sigma*sigma;

    x.type (colvarvalue::type_scalar);
}


colvar::qinterface::qinterface()
    : cvc ()
{
  function_type = "qinterface";
  x.type (colvarvalue::type_scalar);
}


void colvar::qinterface::calc_value()
{
    x.real_value = 0.0;
    cvm::real a, dr;
    int i, j;

    a = 0.0;
    for (i=0;i<n_group1;++i) {
	for (j=0;j<n_group2;++j) {
	    if ( (qobias_flag || qobias_exp_flag) && rN[i][j]>=cutoff ) continue;
	        	
            cvm::rvector const dx = (*group1)[i].pos - (*group2)[j].pos;
	    
	    r[i][j] = dx.norm();
	    dr = r[i][j] - rN[i][j];
	    q[i][j] = exp(-dr*dr/(2*sigma_sq));

	    x.real_value += q[i][j];
	    a +=1.0;
	}
    }
    x.real_value = x.real_value/a;

}


void colvar::qinterface::calc_gradients()
{

    cvm::real a, force1, dr;
    int i, j;

    a = 0.0;
    for (i=0;i<n_group1;++i) {
	for (j=0;j<n_group2;++j) {
	    if ( (qobias_flag || qobias_exp_flag) && rN[i][j]>=cutoff ) continue;
	    a +=1.0;
	}
    }
    
    for (i=0;i<n_group1;++i) {
	for (j=0;j<n_group2;++j) {
	        	
	    if ( (qobias_flag || qobias_exp_flag) && rN[i][j]>=cutoff ) continue;
	        	
            cvm::rvector const dx = (*group1)[i].pos - (*group2)[j].pos;
	    r[i][j] = dx.norm();
	    dr = r[i][j] - rN[i][j];
	    q[i][j] = exp(-dr*dr/(2*sigma_sq));

	    force1 = q[i][j]*dr/r[i][j]/sigma_sq/a;

            (*group1)[i].grad += - force1 * dx;
            (*group2)[j].grad +=   force1 * dx;
	}
    }

}


void colvar::qinterface::calc_force_invgrads()
{
    cvm::fatal_error ("Error: Inverse gradients for qinterface is not implemented !!! \n");
}


void colvar::qinterface::calc_Jacobian_derivative()
{
    cvm::fatal_error ("Error: Jacobian for qinterface is not implemented !!! \n");
}


void colvar::qinterface::apply_force (colvarvalue const &force)
{
  if (!group1->noforce)
    group1->apply_colvar_force (force.real_value);

  if (!group2->noforce)
    group2->apply_colvar_force (force.real_value);
}

colvar::qinterface::~qinterface()
{
	if (allocated) {
		for (int i=0;i<n_group1;i++) {
			delete [] r[i];
			delete [] rN[i];
			delete [] q[i];

		}

		delete [] r;
		delete [] rN;
		delete [] q;

	}
}

void colvar::qinterface::allocate()
{
    r = new double*[n_group1];
    rN = new double*[n_group1];
    q = new double*[n_group1];
    
    
    for (int i = 0; i < n_group1; ++i) {
    	r[i] = new double [n_group2];
    	rN[i] = new double [n_group2];
    	q[i] = new double [n_group2];
    }

allocated = true;
}



