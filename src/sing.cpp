
#include "sing.h"

namespace Streaming
{

RationalSinger::RationalSinger (float acuity)
  : m_harmony(
      Rational::HARMONY_MAX_RADIUS,
      Rational::HARMONY_ACUITY,
      acuity)
{
}

void RationalSinger::push (Seconds time, const StereoAudioFrame & sound_in)
{
  m_harmony.analyze(sound_in);
}

void RationalSinger::pull (Seconds time, StereoAudioFrame & sound_out)
{
  sound_out.zero();
  m_harmony.sample(sound_out);
}

} // namespace Streaming

