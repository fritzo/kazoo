
#include "common.h"
#include "fft.h"

void test_fft (size_t exponent)
{
  size_t size = 1 << exponent;

  FFT_C2C fft(exponent);

  for (size_t i = 0; i < size; ++i) {
    float t = 2 * M_PI * (i + 0.5) / size;
    fft.time_in[i] = complex(cos(t), sin(t));
  }

  cout << "time input:" << endl;
  print_complex(fft.time_in, size);

  fft.transform_fwd();

  cout << "freq output:" << endl;
  print_complex(fft.freq_out, size);

  multiply(1.0f/size, fft.freq_out, fft.freq_in);

  fft.transform_bwd();

  cout << "time output:" << endl;
  print_complex(fft.time_out, size);
}

int main (void)
{
  size_t exponent = 4;
  test_fft(exponent);
}

