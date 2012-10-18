
#ifndef KAZOO_PARTICLES_H
#define KAZOO_PARTICLES_H

#include "common.h"
#include "vectors.h"
#include <utility>

namespace Particle
{

//----( manager )-------------------------------------------------------------

class Manager
{
  const size_t m_capacity;
  size_t m_size;
  const size_t m_default_reserve_size;

  uint32_t m_first_free;
  Vector<uint32_t> m_next_free;

  typedef std::pair<float, uint32_t> Rank;
  const Vector<float> & m_rank_fun;
  Vector<Rank> m_ranks;

public:

  Manager (size_t capacity, const Vector<float> & rank_fun);
  ~Manager () {}

  size_t capacity () const { return m_capacity; }
  size_t size () const { return m_size; }
  size_t available () const { return m_capacity - m_size; }
  bool full () const { return m_size == m_capacity; }

  size_t alloc ();
  void free (size_t i);
  void reserve (size_t count);
};

//----( particle matrix )-----------------------------------------------------

class Matrix
{
public:

  const size_t num_particles;
  const size_t num_fields;

  Vector<float> data_pf;
  Vector<float> data_fp;

  Matrix (size_t p, size_t f)
    : num_particles(p),
      num_fields(f),
      data_pf(p * f),
      data_fp(f * p)
  {
    ASSERT_DIVIDES(4, p);
    ASSERT_DIVIDES(4, f);
  }

  float * particle (size_t p) { return data_pf + num_fields * p; }
  const float * particle (size_t p) const { return data_pf + num_fields * p; }

  float * field (size_t f) { return data_fp + num_particles * f; }
  const float * field (size_t f) const { return data_fp + num_particles * f; }

  void pf_to_fp ();
  void fp_to_pf ();
};

} // namespace Particle

#endif // KAZOO_PARTICLES_H

