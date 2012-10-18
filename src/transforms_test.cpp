
#include "common.h"
#include "transforms.h"

void test_splitter (size_t size = 1024, size_t size_lowpass = 16)
{
  HiLoSplitter(size, size_lowpass).test();
}

int main (void)
{
  test_splitter();

  return 0;
}

