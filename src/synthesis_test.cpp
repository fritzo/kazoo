
#include "synthesis.h"
#include "audio.h"
#include "filters.h"

using namespace Synthesis;

void test_simple_synth ()
{
  Audio audio(256, DEFAULT_SAMPLE_RATE, false, true);

  StereoAudioFrame sound(audio.size());
  complex phase = 1;

  audio.start();
  while (true) {
    for (size_t i = 0; i < 256; ++i) {
      phase *= exp_2_pi_i(.01f * random_std());
      sound[i] = phase;
    }
    phase /= norm(phase);
    audio.write(sound);
  }
  audio.stop();
}

void test_voice_ramp (size_t steps = 100)
{
  typedef Buzzer::Timbre Timbre;

  Timbre t0(1.0, -24, -10, 0.1);
  Timbre t1(1.0,  24,  10, 0.1);
  Timbre dt = (t1 - t0) / steps;

  Buzzer buzzer(t0);
  Audio audio(1024, DEFAULT_SAMPLE_RATE, false, true);
  Reverberator reverb;
  StereoAudioFrame sound(audio.size());

  audio.start();
  Timbre t = t0;
  for (size_t i = 0; i < steps; ++i) {
    t += dt;

    sound.zero();
    buzzer.sample(t, sound);
    reverb.transform(sound);
    audio.write(sound);
    cout << '.' << flush;
  }
  audio.stop();
}

void test_layout (size_t steps = 1000)
{
  Buzzer buzzer;
  Audio audio(1024, DEFAULT_SAMPLE_RATE, false, true);
  Reverberator reverb;

  StereoAudioFrame sound(audio.size());

  audio.start();
  for (size_t i = 0; i < steps; ++i) {

    Finger finger;
    finger.set_energy(1);
    finger.set_x(40 * sin(3.0 * 2 * M_PI * i / steps));
    finger.set_y(20 * sin(7.0 * 2 * M_PI * i / steps));
    finger.set_z(sqr(sin(11.0 * 2 * M_PI * i / steps)));

    Buzzer::Timbre t = Buzzer::layout(finger);

    sound.zero();
    buzzer.sample(t, sound);
    reverb.transform(sound);
    sound *= 1e-1f; // HACK
    soft_clip(sound);
    audio.write(sound);
    cout << '.' << flush;
  }
  audio.stop();
}

void test_shepard (size_t steps = 1<<8)
{
  float frames_per_second = DEFAULT_AUDIO_FRAMERATE;

  Audio audio(1024, DEFAULT_SAMPLE_RATE, false, true);
  ShepardVibe voice;
  Filters::MaxGain output_gain(120.0 * frames_per_second);
  StereoAudioFrame sound_out(audio.size());

  audio.start();
  for (size_t i = 0; i < steps; ++i) {

    float t = 2 * (0.5 + i) / steps - 1; // in [-1,1]

    Finger finger;
    finger.clear();
    finger.set_energy(1);
    finger.set_x(8 * t);
    finger.set_y(-2 * t);
    finger.set_z(1);

    ShepardVibe::Timbre timbre = ShepardVibe::layout(finger);

    sound_out.zero();
    voice.sample(timbre, sound_out);

    float energy_out = sqrtf(norm_squared(sound_out));
    sound_out *= output_gain.update(energy_out);

    audio.write(sound_out);
    cout << '.' << flush;
  }
  audio.stop();
}

const char * help_message =
"Usage: synthesis_test COMMAND"
"\nCommands:"
"\n  shepard      Play shepard tones"
;

int main (int argc, char ** argv)
{
  if (argc < 2) {
    LOG(help_message);
    return 1;
  }

  string command(argv[1]);

  if (command == "shepard") test_shepard(); else
  ERROR("unknown command: " << command << "\n" << help_message);

  return 0;
}

