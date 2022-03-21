#include <cmath>

#include "colvarmodule.h"
#include "colvarvalue.h"
#include "colvarparse.h"
#include "colvar.h"
#include "colvarcomp.h"

/*
 * Implement the probability distribution function of pair-wise distance here 
 */

colvar::pofr::pofr(std::string const &conf)
  : cvc(conf)
{
  function_type = "pofr";

  if (get_keyval(conf, "forceNoPBC", b_no_PBC, false)) {
    cvm::log("Computing distance using absolute positions (not minimal-image)");
  }

  group1 = parse_group(conf, "group1");
  group2 = parse_group(conf, "group2");

  /*
  x.type(colvarvalue::type_vector);
  x.vector1d_value.resize(group1->size() * group2->size());
  */
  x.type (colvarvalue::type_scalar);
}


colvar::pof::pof()
{
  function_type = "pof";
  //x.type(colvarvalue::type_vector);
  x.type (colvarvalue::type_scalar);
}


void colvar::pof::calc_value()
{
  cvm::real t_min, t_max;
  x.real_value = 0.0;
  //x.vector1d_value.resize(group1->size() * group2->size());

  if (b_no_PBC) {
    size_t i1, i2;
    for (i1 = 0; i1 < group1->size(); i1++) {
      for (i2 = 0; i2 < group2->size(); i2++) {
        cvm::rvector const dv = (*group2)[i2].pos - (*group1)[i1].pos;
        cvm::real const d = dv.norm();

        t_min = tanh( kappa*(d - well_r_min) );
        t_max = tanh( kappa*(well_r_max - d) );
        v_theta[i_well][i][j] = 0.25*(1.0 + t_min)*(1.0 + t_max);

        x.real_value = d;

        (*group1)[i1].grad = -1.0 * dv.unit();
        (*group2)[i2].grad =  dv.unit();
      }
    }
  } else {
    size_t i1, i2;
    for (i1 = 0; i1 < group1->size(); i1++) {
      for (i2 = 0; i2 < group2->size(); i2++) {
        cvm::rvector const dv = cvm::position_distance((*group1)[i1].pos, (*group2)[i2].pos);
        cvm::real const d = dv.norm();
        x.vector1d_value[i1*group2->size() + i2] = d;
        (*group1)[i1].grad = -1.0 * dv.unit();
        (*group2)[i2].grad =  dv.unit();
      }
    }
  }
}

void colvar::distance_pairs::calc_gradients()
{
  // will be calculated on the fly in apply_force()
}

void colvar::distance_pairs::apply_force(colvarvalue const &force)
{
  if (b_no_PBC) {
    size_t i1, i2;
    for (i1 = 0; i1 < group1->size(); i1++) {
      for (i2 = 0; i2 < group2->size(); i2++) {
        cvm::rvector const dv = (*group2)[i2].pos - (*group1)[i1].pos;
        (*group1)[i1].apply_force(force[i1*group2->size() + i2] * (-1.0) * dv.unit());
        (*group2)[i2].apply_force(force[i1*group2->size() + i2] * dv.unit());
      }
    }
  } else {
    size_t i1, i2;
    for (i1 = 0; i1 < group1->size(); i1++) {
      for (i2 = 0; i2 < group2->size(); i2++) {
        cvm::rvector const dv = cvm::position_distance((*group1)[i1].pos, (*group2)[i2].pos);
        (*group1)[i1].apply_force(force[i1*group2->size() + i2] * (-1.0) * dv.unit());
        (*group2)[i2].apply_force(force[i1*group2->size() + i2] * dv.unit());
      }
    }
  }
}


colvar::qbias::qbias (std::string const &conf)
    : cvc (conf)
{
    function_type = "qbias";
    atoms = parse_group (conf, "atoms");
    atom_groups.push_back (atoms);

    atoms->b_center = false;
          
    n = atoms->size();
    l = 2;      // only power of 2 is allowed
    // allocate variables
    allocated = false;
    allocate();

    qbias_flag      = 0;
    qbias_exp_flag  = 0;
    qobias_flag     = 0;
    qobias_exp_flag = 0;

    // get qbias flag
    get_keyval (conf, "qbias",      qbias_flag, false);
    if (qbias_flag) cvm::log("QBias flag on\n");

    get_keyval (conf, "qbias_exp",  qbias_exp_flag, false);
    if (qbias_exp_flag) cvm::log("QBias_Exp flag on\n");

    get_keyval (conf, "qobias",     qobias_flag, false);
    if (qobias_flag) cvm::log("QOBias flag on\n");

    get_keyval (conf, "qobias_exp", qobias_exp_flag, false);
    if (qobias_exp_flag) cvm::log("QOBias_Exp flag on\n");

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
	for (int i=0;i<n;++i)
	    for (int j=0;j<n;++j)
                in_rnative >> rN[i][j];
                
	in_rnative.close();
    } else {
        cvm::fatal_error ("Error: Native file not set!!! \n");
    }

    // Minimal sequence separation
    min_sep = 3;
    if (qobias_flag || qobias_exp_flag) min_sep=4;
    
    for (int i=0;i<n;i++) {
        sigma_sq[i] = Sigma(i)*Sigma(i);
        //fprintf(screen, "%d %12.6f %12.6f\n",i, Sigma(i), sigma_sq[i]);
    }

    x.type (colvarvalue::type_scalar);
}


colvar::qbias::qbias()
    : cvc ()
{
  function_type = "qbias";
  x.type (colvarvalue::type_scalar);
}


void colvar::qbias::calc_value()
{
    x.real_value = 0.0;
    cvm::real a, dr;
    int i, j;

    a = 0.0;
    for (i=0;i<n;++i) {
	for (j=i+min_sep;j<n;++j) {
	    if ( (qobias_flag || qobias_exp_flag) && rN[i][j]>=cutoff ) continue;
	        	
            cvm::rvector const dx = (*atoms)[i].pos - (*atoms)[j].pos;
	    
	    r[i][j] = dx.norm();
	    dr = r[i][j] - rN[i][j];
	    q[i][j] = exp(-dr*dr/(2*sigma_sq[j-i]));

	    x.real_value += q[i][j];
	    a +=1.0;
	}
    }
    x.real_value = x.real_value/a;

}


void colvar::qbias::calc_gradients()
{

    cvm::real a, dr, force1;
    int i, j;
    
    a = 0.0;
    for (i=0;i<n;++i) {
	for (j=i+min_sep;j<n;++j) {
	    if ( (qobias_flag || qobias_exp_flag) && rN[i][j]>=cutoff ) continue;
	    a +=1.0;
	}
    }

    for (i=0;i<n;++i) {
	for (j=i+min_sep;j<n;++j) {
	    if ( (qobias_flag || qobias_exp_flag) && rN[i][j]>=cutoff ) continue;
	        	
            cvm::rvector const dx = (*atoms)[i].pos - (*atoms)[j].pos;
	    r[i][j] = dx.norm();
	    dr = r[i][j] - rN[i][j];
	    q[i][j] = exp(-dr*dr/(2*sigma_sq[j-i]));

	    force1 = q[i][j]*dr/r[i][j]/sigma_sq[j-i]/a;

            (*atoms)[i].grad += - force1 * dx;
            (*atoms)[j].grad +=   force1 * dx;
	}
    }

}


void colvar::qbias::calc_force_invgrads()
{
    cvm::fatal_error ("Error: Inverse gradients for qbias is not implemented !!! \n");
}


void colvar::qbias::calc_Jacobian_derivative()
{
    cvm::fatal_error ("Error: Jacobian for qbias is not implemented !!! \n");
}


void colvar::qbias::apply_force (colvarvalue const &force)
{
  if (!atoms->noforce)
    atoms->apply_colvar_force (force.real_value);
}


colvar::qbias::~qbias()
{
	if (allocated) {
		for (int i=0;i<n;i++) {
			delete [] r[i];
			delete [] rN[i];
			delete [] q[i];

		}

		delete [] r;
		delete [] rN;
		delete [] q;

		delete [] sigma_sq;
	}
}

void colvar::qbias::allocate()
{
    r = new double*[n];
    rN = new double*[n];
    q = new double*[n];
    
    sigma_sq = new double[n];
    
    for (int i = 0; i < n; ++i) {
    	r[i] = new double [n];
    	rN[i] = new double [n];
    	q[i] = new double [n];
    }

allocated = true;
}

cvm::real colvar::qbias::Sigma(int sep)
{
    if (qbias_exp_flag || qobias_exp_flag)
    	return pow(1+sep, sigma_exp);
    
    return sigma;
}


