
#include "common.h"
#include "window.h"

void test_wrap (int I = 24)
{
  for (int i = 0; i < I; ++i) {
    float t = 2.0f * (0.5f + i) / I - 1.0f;
    cout << "wrap(" << t << ",0.5) = " << wrap(t,0.5) << endl;
  }
}

void test_narrow (size_t max_exponent = 6)
{
  for (size_t i = 0; i <= max_exponent; ++i) {
    LOG(i << ", " << NarrowWindow(1.0 / (1 << i)).power);
  }
}

int main ()
{
  test_wrap();
  test_narrow();

  return 0;
}

