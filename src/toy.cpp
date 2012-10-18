
#include "common.h"
#include "streaming.h"
#include "streaming_shared.h"
#include "streaming_audio.h"
#include "streaming_video.h"
#include "streaming_devices.h"
#include "streaming_synthesis.h"
#include "flock.h"
#include "events.h"
#include "args.h"

#define TOY_PSYCHO_HARMONY_SIZE            (800)

Rectangle g_screen_shape = Rectangle(400,800);
bool g_deaf = true;

namespace Streaming
{

//----( commands )------------------------------------------------------------

void run_flock_pitch (Args & args)
{
  float acuity = args.pop(PSYCHO_PITCH_ACUITY);
  size_t size = args.pop(TOY_PSYCHO_HARMONY_SIZE);
  float min_freq = args.pop(MIN_CHROMATIC_FREQ_HZ);
  float max_freq = args.pop(MAX_CHROMATIC_FREQ_HZ);

  StereoAudioThread audio(not g_deaf);
  SpeakerGain speaker_gain;
  PitchFlockViewer flock(g_screen_shape, size, acuity, min_freq, max_freq);
  ShowRgb screen(flock);

  audio.in - speaker_gain;
  speaker_gain.in - flock;
  screen.in - flock;

  if (g_deaf) {

    run();

  } else {

    MicGain mic_gain;

    audio.out - mic_gain;
    mic_gain.out - flock;

    run();
  }
}

void run_flock_tempo (Args & args)
{
  size_t size = args.pop(PSYCHO_RHYTHM_SIZE);
  float min_freq = args.pop(PSYCHO_MIN_TEMPO_HZ);
  float max_freq = args.pop(PSYCHO_MAX_TEMPO_HZ);

  StereoAudioThread audio(not g_deaf);
  SpeakerGain speaker_gain;
  TempoFlockViewer flock(g_screen_shape, size, min_freq, max_freq);
  ShowRgb screen(flock);

  audio.in - speaker_gain;
  speaker_gain.in - flock;
  screen.in - flock;

  if (g_deaf) {

    run();

  } else {

    PowerMeter power;

    audio.out - power;
    power.out - flock;

    run();
  }
}

} // namespace Streaming

using namespace Streaming;

//----( options )-------------------------------------------------------------

void option_zoom (Args & args)
{
  ASSERT(args.size() > 2, "no zoom values specified");

  size_t width = atoi(args.pop());
  size_t height = atoi(args.pop());
  g_screen_shape = Rectangle(width, height);
}
void option_listen (Args & args) { g_deaf = false; }

//----( harness )-------------------------------------------------------------

const char * help_message =
"Usage: toy [OPTIONS] COMMAND [ARGS]"
"\nOptions:"
"\n  zoom WIDTH HEIGHT"
"\n  listen"
"\nCommands:"
"\n  pitch [ACUITY = 7] [SIZE = 256] [MIN_FREQ] [MAX_FREQ]"
"\n  tempo [SIZE] [MIN_FREQ] [MAX_FREQ]"
;

int main (int argc, char ** argv)
{
  LOG(kazoo_logo);

  Args args(argc, argv, help_message);

  args
    .case_("zoom", option_zoom)
    .case_("listen", option_listen)
    .default_break_else_repeat();

  args
    .case_("pitch", run_flock_pitch)
    .case_("tempo", run_flock_tempo)
    .default_(run_flock_pitch);

  return 0;
}

