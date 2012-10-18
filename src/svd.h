#ifndef KAZOO_SVD_H
#define KAZOO_SVD_H

#include "common.h"
#include "vectors.h"
#include <vector>

class SymmetricLinearForm
{
protected:

  size_t m_size;

public:

  SymmetricLinearForm () : m_size(0) {}
  SymmetricLinearForm (size_t size) : m_size(size) {}
  virtual ~SymmetricLinearForm () {}

  size_t size () const { return m_size; }

  virtual void apply (const Vector<float> & x, Vector<float> & y) const = 0;

  void compute_eigs (std::vector<Vector<float> *> & eigs);
};

#endif // KAZOO_SVD_H
