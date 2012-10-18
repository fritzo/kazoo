
#ifndef KAZOO_AUDIO_TYPES_H
#define KAZOO_AUDIO_TYPES_H

#include "common.h"
#include "vectors.h"

struct MonoAudioFrame : public Vector<float>
{
  MonoAudioFrame (
      size_t size = DEFAULT_FRAMES_PER_BUFFER,
      float * data = NULL)
    : Vector<float>(size, data)
  {}
};

struct StereoAudioFrame : public Vector<complex>
{
  StereoAudioFrame (
      size_t size = DEFAULT_FRAMES_PER_BUFFER,
      complex * data = NULL)
    : Vector<complex>(size, data)
  {}
};

#endif // KAZOO_AUDIO_TYPES_H

