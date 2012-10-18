
#include "synthesis.h"

using namespace Synthesis;

void profile_chorus (
    float duration_seconds = 10.0f,
    size_t num_voices = 8)
{
  size_t T = roundi(duration_seconds * DEFAULT_SAMPLE_RATE);
  size_t dT = DEFAULT_FRAMES_PER_BUFFER;

  Chorus synth;

  Ids ids(num_voices);
  Vector<float> energies(num_voices);
  Vector<float4> positions(num_voices);

  Vector<complex> sound(dT);

  for (size_t t = 0; t < T; t += dT) {

    for (size_t i = 0; i < num_voices; ++i) {
      ids[i] = i;
      energies[i] = 1;
      positions[i] = float4(i,0,1,0);
    }

    sound.zero();
    synth.sample(ids, energies, positions, sound);
  }
}

int main ()
{
  profile_chorus ();

  return 0;
}

