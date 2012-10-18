
#include "common.h"
#include "random.h"

void random_multinomials (
    const Vector<float> & weights_in,
    Vector<uint32_t> & indices_out)
{
  float total = sum(weights_in);
  ASSERT_LT(0, total);

  const size_t I = weights_in.size;
  const size_t N = indices_out.size;
  const float * restrict weights = weights_in;

  // this uses a comb-in-bin particle resampling method
  float dw = N / total;
  size_t i = 0;
  float offset = weights[i] * dw - random_01();
  typedef Vector<uint32_t>::iterator Auto;
  for (Auto index = indices_out.begin(); index != indices_out.end(); ++index) {

    while (offset < 0) {
      i = (i + 1) % I;
      offset += weights[i] * dw;
    }
    offset -= 1;

    *index = i;
  }
}

