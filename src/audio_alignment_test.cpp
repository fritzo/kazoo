
#include "common.h"
#include "audio_alignment.h"
#include "voice.h"
#include "streaming_devices.h"
#include "scripting.h"
#include "images.h"
#include "args.h"

using namespace AudioAlignment;

namespace Streaming
{

const char * g_voice_config = "config/alignment.voice.conf";
string g_audio_infilename = "data/mono.raw";
size_t g_max_duration = 8192;

//----( common patches )------------------------------------------------------

AlignmentModel * g_model = NULL;
FeatureBuffer * g_buffer = NULL;

void init_features ()
{
  DecompressAudio temp(g_audio_infilename, false);
  MonoAudioFile file(g_audio_infilename);
  VoiceAnalyzer analyzer(g_voice_config);

  g_model = new AlignmentModel(analyzer.features_out.size());
  AlignmentModel & model = * g_model;

  g_buffer = new FeatureBuffer(model, g_max_duration);
  FeatureBuffer & buffer = * g_buffer;

  file.out - analyzer;
  analyzer.features_out - buffer;

  file.run(DEFAULT_AUDIO_FRAMERATE);

  PRINT2(buffer.duration(), buffer.feature_size());
}

void clear_features ()
{
  if (g_model) delete g_model;
  if (g_buffer) delete g_buffer;
}

//----( commands )------------------------------------------------------------

void run_features (Args & args)
{
  string outfilename = "data/audio_features.png";

  args
    .case_("-d", g_max_duration)
    .case_("-i", g_audio_infilename)
    .case_("-o", outfilename)
    .default_break_else_repeat();

  init_features();
  FeatureBuffer & buffer = * g_buffer;

  Image::save_png(
      outfilename.c_str(),
      buffer.duration(),
      buffer.feature_size(),
      buffer.image().data);
}

void run_cost (Args & args)
{
  string outfilename = "data/cost_matrix.png";

  args
    .case_("-d", g_max_duration)
    .case_("-i", g_audio_infilename)
    .case_("-o", outfilename)
    .default_break_else_repeat();

  init_features();
  AlignmentModel & model = * g_model;
  FeatureBuffer & buffer = * g_buffer;

  CostMatrix cost_matrix(model, buffer, false);

  Vector<float> & image = cost_matrix.cost();

  // shade: similar as black, different as white
  image /= max(image);

  Image::save_png(
      outfilename.c_str(),
      cost_matrix.size(),
      cost_matrix.size(),
      image);
}

void run_marginal (Args & args)
{
  string outfilename = "data/alignment.png";

  args
    .case_("-d", g_max_duration)
    .case_("-i", g_audio_infilename)
    .case_("-o", outfilename)
    .default_break_else_repeat();

  init_features();
  AlignmentModel & model = * g_model;
  FeatureBuffer & buffer = * g_buffer;

  CostMatrix cost_matrix(model, buffer, false);
  AlignmentMatrix alignment(model, buffer.duration());
  alignment.init_marginal(cost_matrix);

  Vector<float> & image = alignment.posterior();

  // shade: similar as black, different as white
  image /= max(image);

  Image::save_png(
      outfilename.c_str(),
      cost_matrix.size(),
      cost_matrix.size(),
      image);
}

void run_maxlike (Args & args)
{
  string outfilename = "data/alignment.png";

  args
    .case_("-d", g_max_duration)
    .case_("-i", g_audio_infilename)
    .case_("-o", outfilename)
    .default_break_else_repeat();

  init_features();
  AlignmentModel & model = * g_model;
  FeatureBuffer & buffer = * g_buffer;

  CostMatrix cost_matrix(model, buffer, false);
  AlignmentMatrix alignment(model, buffer.duration());
  alignment.init_maxlike(cost_matrix);

  Vector<float> & image = alignment.posterior();

  // shade: similar as black, different as white
  float radius = 1e3f;
  imax(image, -radius);
  imin(image, radius);
  image /= -2 * radius;
  image += 0.5f;

  Image::save_png(
      outfilename.c_str(),
      alignment.size(),
      alignment.size(),
      image);
}

void run_path (Args & args)
{
  string outfilename = "data/path.png";

  args
    .case_("-d", g_max_duration)
    .case_("-i", g_audio_infilename)
    .case_("-o", outfilename)
    .default_break_else_repeat();

  init_features();
  AlignmentModel & model = * g_model;
  FeatureBuffer & buffer = * g_buffer;

  CostMatrix cost_matrix(model, buffer, true);
  AlignmentPath path(model, cost_matrix);

  Vector<float> & image = path.posterior();
  image -= min(image);
  image /= max(image);

  Image::save_png(
      outfilename.c_str(),
      path.size(),
      path.size(),
      image);
}

} // namespace Streaming

using namespace Streaming;

//----( main )----------------------------------------------------------------

const char * help_message =
"Usage: audio_alignment_test [COMMAND]"
"\nCommands:"
"\n  features [-d MAX_DURATION -i INFILE -o OUTFILE]"
"\n  cost [-d MAX_DURATION -i INFILE -o OUTFILE]"
"\n  marginal [-d MAX_DURATION -i INFILE -o OUTFILE]"
"\n  maxlike [-d MAX_DURATION -i INFILE -o OUTFILE]"
"\n  path [-d MAX_DURATION -i INFILE -o OUTFILE]"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  args
    .case_("features", run_features)
    .case_("cost", run_cost)
    .case_("marginal", run_marginal)
    .case_("maxlike", run_maxlike)
    .case_("path", run_path)
    .default_error();

  clear_features();

  return 0;
}

