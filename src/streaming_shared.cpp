
#include "streaming_shared.h"

namespace Streaming
{

SharedMaxLowpass::SharedMaxLowpass (
    float timescale,
    float initial_value)

  : m_timescale(timescale),
    m_time(Seconds::now()),
    m_lowpass(initial_value)
{}

void SharedMaxLowpass::push (Seconds time, const float & value)
{
  float dt = max(0.0f, time - m_time);
  m_time = time;
  float decay = exp(-dt / m_timescale);

  m_mutex.lock();

  m_lowpass = max(decay * m_lowpass, value);

  m_mutex.unlock();
}

void SharedMaxLowpass::pull (Seconds time, float & value)
{
  m_mutex.lock();

  value = m_lowpass;

  m_mutex.unlock();
}

} // namespace Streaming

