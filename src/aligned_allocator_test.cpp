
#include "aligned_allocator.h"
#include "common.h"
#include <vector>

enum { alignment = 32 };

#ifdef ASSERT_ALIGNED
#undef ASSERT_ALIGNED
#endif
#define ASSERT_ALIGNED(ptr) \
  ASSERT((reinterpret_cast<size_t>(ptr) % alignment) == 0, \
         "pointer " # ptr " is not aligned, offset = " \
      << (reinterpret_cast<size_t>(ptr) % alignment));

typedef std::vector<float, nonstd::aligned_allocator<float, alignment>> V;

void test_alignment (int max_size = 2000)
{
  std::vector<V *, nonstd::aligned_allocator<V *>> vectors(0);

  for (int size = 0; size < max_size; ++size) {
    vectors.push_back(new V(size));
  }

  for (int size = 0; size < max_size; ++size) {
    V * p = vectors[size];
    ASSERT_ALIGNED(& (* p)[0]);
    delete p;
  }

  ASSERT_ALIGNED(& vectors[0]);

  vectors.clear();
}

int main ()
{
  test_alignment();

  return 0;
}

