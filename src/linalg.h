#ifndef KAZOO_LINALG_H
#define KAZOO_LINALG_H

/** Linear algebra structures and algorithms.

  Symmetric matrices are stored as full square matrices.

  TODO sketch online svd algorithm.
  TODO copy linalg stuff from johann/cpp/linalg.[Ch]
  TODO wrap necessary lapack routines
*/

#include "common.h"
#include "vectors.h"

#define PRINT_VECT(vector) \
  LOG(#vector " = " << LinAlg::print_vector(vector))
#define PRINT_MAT(m,n,matrix) \
  LOG(#matrix " = " << LinAlg::print_matrix(m,n,matrix))

namespace LinAlg
{

string print_vector (const Vector<float> & A);
string print_matrix (int m, int n, const Vector<float> & A);

//----( fortran wrappers )----------------------------------------------------

void transpose (
    int m,
    int n,
    const Vector<float> & A,
    Vector<float> & At);

// AB = alpha A B + beta AB
void matrix_multiply (
    int I,
    int J,
    int K,
    const Vector<float> & A,  // rect(I,J)
    bool trans_A,
    const Vector<float> & B,  // rect(J,K)
    bool trans_B,
    Vector<float> & AB,       // rect(I,K)
    float alpha = 1,
    float beta = 0);

void outer_prod (
    int m,
    int n,
    const Vector<float> & A,  // rect(m,n)
    bool trans_A,
    Vector<float> & AAt);     // sym(m,m)

// returns <x|A|x>
float symmetric_norm (
    const Vector<float> & x,  // vect(n)
    const Vector<float> & A); // sym(n)

// in-place, chol(A) chol(A)' = A
void cholesky (
    int n,
    Vector<float> & A, // sym(n,trans)
    bool trans = false);

// in-place
void symmetric_invert (
    int n,
    Vector<float> & A); // sym(n)

void symmetric_solve (
    int m,
    int n,
    const Vector<float> & A,  // sym(m)
    const Vector<float> & BA, // rect(n,m)
    bool trans_B,
    Vector<float> & B);       // rect(n,m)

void least_squares (
    const Vector<float> & A,  // rect(m,n)
    bool trans_A,
    const Vector<float> & Ax, // vect(n), defines n
    Vector<float> & x);       // vect(m), defines m

//----( rotation )------------------------------------------------------------

void set_identity (size_t i, float * restrict matrix);
void orthonormalize (size_t size, size_t count, float * restrict basis);

template<size_t dim>
class Orientation
{
  Vector<float> m_basis;

protected:

  Vector<float> coord (size_t n) { return m_basis.block(dim, n); }

public:

  Orientation () : m_basis(sqr(dim)) { set_identity(dim, m_basis); }
  Orientation (const Orientation & o) : m_basis(sqr(dim)) { operator=(o); }

  void orthonormalize () { LinAlg::orthonormalize(dim, dim, m_basis); }

  float coord_dot (size_t n, float * restrict x) const
  {
    ASSERT_LT(n, dim);
    const float * restrict basis_n = m_basis + dim * n;
    float ip = 0;
    for (size_t i = 0; i < dim; ++i) {
      ip += basis_n[i] * x[i];
    }
    return ip;
  }
};

class Orientation3D : public Orientation<3>
{
public:

  void drag (float dx, float dy);
};

class Orientation4D : public Orientation<4>
{
public:

  void drag1 (float dx, float dy);
  void drag2 (float dx, float dy);
};

} // namespace LinAlg

#endif // KAZOO_LINALG_H

