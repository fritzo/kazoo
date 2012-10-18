
#include "raster_audio.h"
#include "streaming_devices.h"
#include "streaming_vectors.h"
#include "streaming_video.h"
#include "filters.h"
#include "scripting.h"
#include "events.h"
#include "args.h"

using namespace RasterAudio;
using namespace Streaming;

//----( main )----------------------------------------------------------------

const char * help_message =
"Usage: player [OPTIONS] [AUDIOFILES]"
"\nOptions:"
"\n  -s SIZE"
"\n  -f MIN_FREQ"
"\n  -F MAX_FREQ"
"\n  -t TIMESCALE"
"\n  -d DURATION"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);


  size_t size = 512;
  float spectrum_min_freq_hz = SPECTRUM_MIN_FREQ_HZ;
  float spectrum_max_freq_hz = SPECTRUM_MAX_FREQ_HZ;
  float spectrum_max_timescale_sec = SPECTRUM_MAX_TIMESCALE_SEC;
  size_t duration = 1000;

  args
    .case_("-s", size)
    .case_("-d", duration)
    .case_("-f", spectrum_min_freq_hz)
    .case_("-F", spectrum_max_freq_hz)
    .case_("-T", spectrum_max_timescale_sec)
    .default_break_else_repeat();

  if (args.size() == 0) {
    LOG(help_message);
    exit(0);
  }

  SpectrumParam spectrum_param(
      size,
      spectrum_min_freq_hz,
      spectrum_max_freq_hz,
      spectrum_max_timescale_sec);

  // TODO switch to stereo output
  PullSplitter<MonoAudioFrame> splitter;
  MonoAudioThread audio(false, true);
  PitchAnalyzer analyzer(spectrum_param);
  VectorOperations::Powf op(1.0f / 3);
  VectorMap<VectorOperations::Powf> pow(op, size);
  VectorMaxGain gain(size, duration);

  History history(size, duration);
  ShowMono screen(history);

  audio.in - splitter;
  splitter.out - analyzer;
  analyzer.energy_out - pow;
  pow.out - gain;
  gain.out - history;
  screen.in - history;

  while (args.size()) {

    string filename = args.pop();
    Screen::set_title(filename.c_str(), "kazoo");

    DecompressAudio temp(filename, false);
    MonoAudioFile file(filename);

    splitter.in - file;

    run();

    // TODO get audio file to exit here
  }

  return 0;
}

