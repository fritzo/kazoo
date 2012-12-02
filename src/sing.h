
#ifndef KAZOO_SING_H
#define KAZOO_SING_H

#include "common.h"
#include "streaming_audio.h"
#include "rational.h"

namespace Streaming
{

class RationalSinger
  : public Pushed<StereoAudioFrame>,
    public Pulled<StereoAudioFrame>
{
  Rational::Harmony m_harmony;
  StereoAudioFrame m_sound_in;

public:

  RationalSinger (float acuity = Rational::HARMONY_ACUITY);

  virtual void push (Seconds time, const StereoAudioFrame & sound_in);
  virtual void pull (Seconds time, StereoAudioFrame & sound_out);
};

} // namespace Streaming

#endif // KAZOO_SING_H

