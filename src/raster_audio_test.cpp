
#include "raster_audio.h"
#include "streaming_devices.h"
#include "streaming_vectors.h"
#include "filters.h"
#include "scripting.h"
#include "events.h"
#include "args.h"

using namespace RasterAudio;

namespace Streaming
{

//----( global parameters )---------------------------------------------------

size_t g_spectrum_size = 512;
float g_spectrum_min_freq_hz = SPECTRUM_MIN_FREQ_HZ;
float g_spectrum_max_freq_hz = SPECTRUM_MAX_FREQ_HZ;
float g_spectrum_max_timescale_sec = SPECTRUM_MAX_TIMESCALE_SEC;

//----( commands )------------------------------------------------------------

void run_pitch (Args & args)
{
  args
    .case_("-s", g_spectrum_size)
    .case_("-f", g_spectrum_min_freq_hz)
    .case_("-F", g_spectrum_max_freq_hz)
    .case_("-T", g_spectrum_max_timescale_sec)
    .default_break_else_repeat();

  string filename = args.pop("data/mono.raw");
  DecompressAudio temp(filename, false);

  size_t duration = 800;
  bool transposed = true;

  SpectrumParam spectrum_param(
      g_spectrum_size,
      g_spectrum_min_freq_hz,
      g_spectrum_max_freq_hz,
      g_spectrum_max_timescale_sec);

  PitchAnalyzer analyzer(spectrum_param);
  VectorMaxGain gain(analyzer.energy_out.size(), duration);
  SweepVector sweep(analyzer.energy_out.size(), duration, transposed);
  Screen::set_title(
      "kazoo - spacebar to continue, any other key to exit",
      "kazoo");

  analyzer.energy_out - gain;
  gain.out - sweep;

  AudioFile file(filename, false);
  MonoAudioFrame frame;

  while (not file.done()) {

    for (size_t i = 0; i < duration; ++i) {
      file.read_frame(frame);
      analyzer.push(Seconds::now(), frame);
    }

    if (wait_for_keypress() != SDLK_SPACE) break;
  }
}

void run_play (Args & args)
{
  args
    .case_("-s", g_spectrum_size)
    .case_("-f", g_spectrum_min_freq_hz)
    .case_("-F", g_spectrum_max_freq_hz)
    .case_("-T", g_spectrum_max_timescale_sec)
    .default_break_else_repeat();

  string filename = args.pop("data/mono.raw");
  DecompressAudio temp(filename, false);

  size_t size = g_spectrum_size;
  size_t duration = 800;
  bool transposed = true;

  SpectrumParam spectrum_param(
      g_spectrum_size,
      g_spectrum_min_freq_hz,
      g_spectrum_max_freq_hz,
      g_spectrum_max_timescale_sec);

  MonoAudioFile file(filename);
  PullSplitter<MonoAudioFrame> splitter;
  MonoAudioThread audio(false, true);
  PitchAnalyzer analyzer(spectrum_param);
  VectorOperations::Sqrtf op;
  VectorMap<VectorOperations::Sqrtf> sqrt(op, size);
  VectorMaxGain gain(size, duration);
  SweepVector sweep(size, duration, transposed);

  audio.in - splitter;
  splitter.in - file;
  splitter.out - analyzer;
  analyzer.energy_out - sqrt;
  sqrt.out - gain;
  gain.out - sweep;

  run();
}

void run_resynth (Args & args)
{
  args
    .case_("-s", g_spectrum_size)
    .case_("-f", g_spectrum_min_freq_hz)
    .case_("-F", g_spectrum_max_freq_hz)
    .case_("-T", g_spectrum_max_timescale_sec)
    .default_break_else_repeat();

  string filename = args.pop("data/mono.raw");
  DecompressAudio temp(filename, false);

  size_t size = g_spectrum_size;
  size_t duration = 800;
  bool transposed = true;

  SpectrumParam spectrum_param(
      g_spectrum_size,
      g_spectrum_min_freq_hz,
      g_spectrum_max_freq_hz,
      g_spectrum_max_timescale_sec);

  MonoAudioFile file(filename);
  PitchAnalyzer analyzer(spectrum_param);
  PitchSynthesizer synthesizer(spectrum_param);
  PullSplitter<Vector<float> > splitter;
  StereoAudioThread audio(false, true);
  VectorOperations::Sqrtf op;
  VectorMap<VectorOperations::Sqrtf> sqrt(op, size);
  VectorMaxGain gain(size, duration);
  SweepVector sweep(size, duration, transposed);

  analyzer.signal_in - file;
  audio.in - synthesizer;
  synthesizer.amplitude_in - splitter;
  splitter.in - sqrt;
  sqrt.in - analyzer;
  splitter.out - gain;
  gain.out - sweep;

  run();
}

void run_reassign (Args & args)
{
  args
    .case_("-s", g_spectrum_size)
    .case_("-f", g_spectrum_min_freq_hz)
    .case_("-F", g_spectrum_max_freq_hz)
    .case_("-T", g_spectrum_max_timescale_sec)
    .default_break_else_repeat();

  string filename = args.pop("data/mono.raw");
  DecompressAudio temp(filename, false);

  size_t size = g_spectrum_size;
  size_t duration = 800;
  bool transposed = true;

  SpectrumParam spectrum_param(
      g_spectrum_size,
      g_spectrum_min_freq_hz,
      g_spectrum_max_freq_hz,
      g_spectrum_max_timescale_sec);

  PitchReassigner reassign(spectrum_param);

  // pass 1
  {
    MonoAudioFile file(filename);
    PitchAnalyzer analyzer(spectrum_param);

    file.out - analyzer;
    analyzer.energy_out - reassign;

    file.run(DEFAULT_AUDIO_FRAMERATE);
  }

  ASSERT_LT(0, reassign.size());
  reassign.process();

  // pass 2
  {
    PitchSynthesizer synthesizer(spectrum_param);
    PullSplitter<Vector<float> > splitter;
    StereoAudioThread audio(false, true);
    VectorOperations::Sqrtf op;
    VectorMap<VectorOperations::Sqrtf> sqrt(op, size);
    VectorMaxGain gain(size, duration);
    SweepVector sweep(size, duration, transposed);

    audio.in - synthesizer;
    synthesizer.amplitude_in - sqrt;
    sqrt.in - splitter;
    splitter.in - reassign;
    splitter.out - gain;
    gain.out - sweep;

    run();
  }
}

} // namespace Streaming

using namespace Streaming;

//----( main )----------------------------------------------------------------

const char * help_message =
"Usage: raster_audio_test COMMAND_SEQUENCE [ARGS]"
"\nCommands:"
"\n  pitch [-s SIZE -f MIN_FREQ MAX_FREQ -t TIMESCALE] [FILENAME]"
"\n  play [-s SIZE -f MIN_FREQ MAX_FREQ -t TIMESCALE] [FILENAME]"
"\n  resynth [-s SIZE -f MIN_FREQ MAX_FREQ -t TIMESCALE] [FILENAME]"
"\n  reassign [-s SIZE -f MIN_FREQ MAX_FREQ -t TIMESCALE] [FILENAME]"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  args
    .case_("pitch", run_pitch)
    .case_("play", run_play)
    .case_("resynth", run_resynth)
    .case_("reassign", run_reassign)
    .default_error();

  return 0;
}

