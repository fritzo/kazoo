
#include "common.h"
#include "audio.h"

namespace
{

class Wire : public AudioThread
{

public:

  Wire (
      size_t frames_per_buffer = DEFAULT_FRAMES_PER_BUFFER,
      size_t sample_rate_hz = DEFAULT_SAMPLE_RATE)
    : AudioThread(frames_per_buffer, sample_rate_hz, true, true, true)
  {}

protected:

  virtual void process (
      const complex * restrict samples_in,
      complex * restrict samples_out,
      size_t size)
  {
    copy_complex(samples_in, samples_out, size);
  }
};

class Gong : public AudioThread
{
  const complex m_rate;
  const float m_scale;
  complex m_state;

public:

  Gong (
      size_t frames_per_buffer = DEFAULT_FRAMES_PER_BUFFER,
      size_t sample_rate_hz = DEFAULT_SAMPLE_RATE,
      float frequency = 110.0,
      float timescale = 1.0)
    : AudioThread(frames_per_buffer, sample_rate_hz, true, true, true, true),

      m_rate(exp( complex(-1 / timescale, 2 * M_PI * frequency)
                / (float) sample_rate_hz)),
      m_scale(frequency / sample_rate_hz),
      m_state(0)
  {
    PRINT(m_rate);
  }

protected:

  virtual void process (
      const complex * restrict samples_in,
      complex * restrict samples_out,
      size_t size)
  {
    for (size_t i = 0; i < size; ++i) {
      m_state += samples_in[i];
      m_state *= m_rate;
      samples_out[i] = m_scale * m_state;
    }
    //cout << '.' << flush;
  }
};

} // anonymous namespace

void test_full_duplex (size_t frames_per_buffer)
{
  Gong duplex(frames_per_buffer);

  LOG("starting full duplex - press any key to exit");

  duplex.start();
  getchar();
  duplex.stop();
}

int main (int argc, char ** argv)
{
  size_t frames_per_buffer = DEFAULT_FRAMES_PER_BUFFER;

  if (argc > 1) frames_per_buffer = atoi(argv[1]);

  test_full_duplex(frames_per_buffer);

  return 0;
}

