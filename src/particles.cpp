
#include "particles.h"
#include "images.h"
#include <algorithm>

namespace Particle
{

//----( manager )-------------------------------------------------------------

Manager::Manager (size_t capacity, const Vector<float> & rank_fun)
  : m_capacity(capacity),
    m_size(0),
    m_default_reserve_size(sqrt(capacity)),

    m_first_free(0),
    m_next_free(capacity),

    m_rank_fun(const_cast<Vector<float> &>(rank_fun)),
    m_ranks(capacity)
{
  ASSERT_LT(0, m_default_reserve_size);

  for (size_t i = 0; i < m_capacity; ++i) {
    free(m_capacity - i - 1);
  }
}

size_t Manager::alloc ()
{
  if (full()) reserve(m_default_reserve_size);

  size_t result = m_first_free;
  m_first_free = m_next_free[m_first_free];
  ++m_size;

  return result;
}

void Manager::free (size_t i)
{
  m_next_free[i] = m_first_free;
  m_first_free = i;
  --m_size;
}

void Manager::reserve (size_t count)
{
  if (count < available()) return;
  count -= available();

  const size_t I = capacity();
  const float * restrict rank_fun = m_rank_fun;
  Rank * restrict ranks = m_ranks;

  for (size_t i = 0; i < I; ++i) {
    ranks[i] = Rank(rank_fun[i], i);
  }

  std::nth_element(ranks, ranks + count, ranks + I);

  for (size_t i = 0; i < count; ++i) {
    free(ranks[i].second);
  }
}

//----( particle matrix )-----------------------------------------------------

void Matrix::pf_to_fp ()
{
  Image::transpose(num_particles, num_fields, data_pf, data_fp);
}

void Matrix::fp_to_pf ()
{
  Image::transpose(num_fields, num_particles, data_fp, data_pf);
}

} // namespace Particle

