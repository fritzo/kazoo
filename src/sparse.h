#ifndef KAZOO_SPARSE_H
#define KAZOO_SPARSE_H

#include "common.h"
#include "vectors.h"

//----( sparse solvers )------------------------------------------------------

namespace LinAlg
{

void sparse_symmetric_solve (
    const size_t dim,
    const size_t degree,
    const Vector<uint16_t> & nbhd,
    const Vector<float> & Pxx,
    const Vector<float> & Pxy,
    Vector<float> & Fxy,
    double tol = 1e-4,
    bool debug = true);

} // namespace LinAlg

#endif // KAZOO_SPARSE_H
