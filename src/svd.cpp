
/** Eigenvector computation using ARPACK.
*/

#define LOG1(mess)

#include "svd.h"
#include <iostream>
#include <cstdlib>
#include <cstring>
#include <cmath>

//----( arpack declarations )-------------------------------------------------

typedef const char * ArpackChars;
typedef int & ArpackInt;
typedef int * ArpackInts;
typedef float & ArpackFloat;
typedef float * ArpackVector<float>;
typedef int & ArpackBool;
typedef int * ArpackBools;

extern "C" void ssaupd_ (
    ArpackInt   ido,
    ArpackChars bmat,
    ArpackInt   n,
    ArpackChars which,
    ArpackInt   nev,
    ArpackFloat  tol,
    ArpackVector<float> resid,
    ArpackInt   ncv,
    ArpackVector<float> v,
    ArpackInt   ldv,
    ArpackInts  iparam,
    ArpackInts  ipntr,
    ArpackVector<float> workd,
    ArpackVector<float> workl,
    ArpackInt   lworkl,
    ArpackInt   info);

extern "C" void sseupd_ (
    ArpackBool  rvec,
    ArpackChars howmny,
    ArpackBools select,
    ArpackVector<float> d,
    ArpackVector<float> z,
    ArpackInt   ldz,
    ArpackFloat  sigma,
    ArpackChars bmat,
    ArpackInt   n,
    ArpackChars which,
    ArpackInt   nev,
    ArpackFloat  tol,
    ArpackVector<float> resid,
    ArpackInt   ncv,
    ArpackVector<float> v,
    ArpackInt   ldv,
    ArpackInts  iparam,
    ArpackInts  ipntr,
    ArpackVector<float> workd,
    ArpackVector<float> workl,
    ArpackInt   lworkl,
    ArpackInt   info);

//----( arpack wrapper )------------------------------------------------------

const bool random_initial_vector = true;
const size_t ncv_factor = 3;  // ARPACK docs recommend at least 2

void SymmetricLinearForm::compute_eigs (std::vector<Vector<float> *> & eigs)
{
  size_t num_eigs = eigs.size();
  LOG("computing " << num_eigs << " eigenvectors");

  // set up call to ssaupd

  int ido = 0; // start
  ArpackChars bmat = "I";
  int n = m_size;
  ArpackChars which = "LA";
  int nev = num_eigs;
  float tol = 1e-10;
  ArpackVector<float> resid = new float[n];
  int ncv = ncv_factor * nev;
  ArpackVector<float> v = new float[n * ncv];
  int ldv = n;

  int ishift = 1;
  int max_iters = 100;
  int nb = 1;
  int mode = 1;
  int iparam[11] = {ishift, 0, max_iters, nb, 0, 0, mode, 0, 0, 0, 0};
  int & num_iters = iparam[3-1];
  int & num_ops = iparam[9-1];

  int ipntr[11];

  ArpackVector<float> workd = new float[3 * n];
  int lworkl = ncv * (ncv + 8);
  ArpackVector<float> workl = new float[lworkl];

  int info = random_initial_vector ? 0 : 1;

  if (not random_initial_vector) {
    float value = pow(m_size, -0.5);
    for (size_t i = 0; i < m_size; ++i) {
      resid[i] = value;
    }
  }

  while (ido != 99)
  {
    LOG1("calling arpack's ssaupd routine (ido = " << ido << ")");
    ssaupd_(ido, bmat, n, which, nev, tol, resid, ncv, v, ldv,
            iparam, ipntr, workd, workl, lworkl, info);

    switch (info)
    {
      case 0: // Normal exit.
        break;

      case 1: LOG1("Maximum number of iterations taken.");
        break;

      case 3: ERROR("No shifts could be applied during a cycle of the\n"
                    "Implicitly restarted Arnoldi iteration. One possibility\n"
                    "is to increase the size of NCV relative to NEV.");
      case -1: ERROR("N must be positive, but was " << n);
      case -2: ERROR("NEV must be positive, but was " << nev);
      case -3: ERROR("NCV must be > NEV and <= N, but was " << ncv);
      case -4: ERROR("max_iters must be > 0, but was " << max_iters);
      case -5: ERROR("WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.");
      case -6: ERROR("BMAT must be one of 'I' or 'G'.");
      case -7: ERROR("Length of private work array WORKL is not sufficient.");
      case -8: ERROR("Error return from trid. eigenvalue calculation;\n"
                     "Informatinal error from LAPACK routine ssteqr.");
      case -9: ERROR("Starting vector is zero.");
      case -10: ERROR("IPARAM(7) must be 1,2,3,4,5.");
      case -11: ERROR("IPARAM(7) = 1 and BMAT = 'G' are incompatable.");
      case -12: ERROR("IPARAM(1) must be equal to 0 or 1.");
      case -13: ERROR("NEV and WHICH = 'BE' are incompatable.");
      case -9999: ERROR("Could not build an Arnoldi factorization.\n"
                        "Size of current Arnoldi factorization = "
                        << iparam[5] << ".\n"
                        "Check that enough workspace and\n"
                        "array storage has been allocated.");

      default: ERROR("unknown ssaupd info code: " << info);
    }

    switch (ido)
    {
      case -1: LOG1("initializing (ido = -1)"); {
        Vector<float> x(m_size, workd + ipntr[1-1] - 1);
        Vector<float> y(m_size, workd + ipntr[2-1] - 1);
        apply(x,y);
      } break;

      case 1: LOG1("applying matrix to vector (ido = 1)"); {
        Vector<float> x(m_size, workd + ipntr[1-1] - 1);
        Vector<float> y(m_size, workd + ipntr[2-1] - 1);
        apply(x,y);
      } break;

      case 99: LOG1("done (ido = 99)");
        break;

      default: ERROR("unknown ido value: " << ido);
    }
  }

  LOG(" ssaupd did " << num_ops << " matrix multiplications"
      " during " << num_iters << " iterations");

  // set up call to sseupd

  int rvec = true;
  ArpackChars howmny = "A";
  ArpackBools select = new int[ncv];
  ArpackVector<float> d = new float[nev];
  ArpackVector<float> z = new float[nev * n];
  memcpy(z, v, n * nev * sizeof(float));
  int ldz = n;
  float sigma;

  LOG("calling arpack's sseupd routine");
  sseupd_(rvec, howmny, select, d, z, ldz, sigma,
          bmat, n, which, nev, tol, resid, ncv, v, ldv,
          iparam, ipntr, workd, workl, lworkl, info);

  switch (info)
  {
    case 0: // Normal exit.
      break;

    case -1: ERROR("N must be positive, but was " << n);
    case -2: ERROR("NEV must be positive, but was " << nev);
    case -3: ERROR("NCV must be > NEV and <= N, but was " << ncv);
    case -5: ERROR("WHICH must be one of 'LM', 'SM', 'LA', 'SA' or 'BE'.");
    case -6: ERROR("BMAT must be one of 'I' or 'G'.");
    case -7: ERROR("Length of private work array WORKL is not sufficient.");
    case -8: ERROR("Error return from trid. eigenvalue calculation;\n"
                   "Informatinal error from LAPACK routine ssteqr.");
    case -9: ERROR("Starting vector is zero.");
    case -10: ERROR("IPARAM(7) must be 1,2,3,4,5.");
    case -11: ERROR("IPARAM(7) = 1 and BMAT = 'G' are incompatable.");
    case -12: ERROR("IPARAM(1) must be equal to 0 or 1.");
    case -14: ERROR("DSAUPD  did not find any eigenvalues to sufficient"
                    "accuracy.");
    case -15: ERROR("HOWMNY must be one of 'A' or 'S' if RVEC = true");
    case -16: ERROR("HOWMNY = 'S' not yet implemented");
    case -17: ERROR("DSEUPD  got a different count of the number of converged"
                    "Ritz values than DSAUPD  got.  This indicates the user"
                    "probably made an error in passing data from DSAUPD  to"
                    "DSEUPD  or that the data was modified before entering"
                    "DSEUPD .");

    default: ERROR("unknown ssaupd info code: " << info);
  }

  switch (ido)
  {
    case -1: LOG1("initializing (ido = -1)"); {
      Vector<float> x(m_size, workd + ipntr[1-1] - 1);
      Vector<float> y(m_size, workd + ipntr[2-1] - 1);
      apply(x,y);
    } break;

    case 1: LOG1("applying matrix to vector (ido = 1)"); {
      Vector<float> x(m_size, workd + ipntr[1-1] - 1);
      Vector<float> y(m_size, workd + ipntr[2-1] - 1);
      apply(x,y);
    } break;

    case 99: LOG1("done (ido = 99)");
      break;

    default: ERROR("unknown ido value: " << ido);
  }

  delete[] resid;
  delete[] v;
  delete[] workd;
  delete[] workl;
  delete[] select;

  LOG("extracting eigenvalue result");
  for (size_t e = 0; e < num_eigs; ++e) {
    for (size_t i = 0; i < m_size; ++i) {
      TODO("rearrange eigenvectors");
      //size_t f = num_eigs - 1 - e;  // put eigenvalues in descending order
      //eigs[i * num_eigs + f] = d[e] * z[e * m_size + i];
    }
  }

  delete[] d;
  delete[] z;
}

