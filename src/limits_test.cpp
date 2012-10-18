
#include "common.h"

int main ()
{
  float a = 0.0f;
  float b = 1.0f;
  complex c(a,b);

  PRINT2(a, safe_isfinite(a));
  PRINT2(b, safe_isfinite(b));
  PRINT2(c, safe_isfinite(c));

  float x = 1.0f / 0.0f;
  float y = x / (1 - x);
  complex z(x,y);

  PRINT2(x, safe_isfinite(x));
  PRINT2(y, safe_isfinite(y));
  PRINT2(z, safe_isfinite(z));

  return 0;
}

