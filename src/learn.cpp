
#include "common.h"
#include "streaming.h"
#include "streaming_shared.h"
#include "streaming_audio.h"
#include "streaming_video.h"
#include "streaming_camera.h"
#include "streaming_devices.h"
#include "streaming_synthesis.h"
#include "streaming_clouds.h"
#include "compress.h"
#include "gloves.h"
#include "voice.h"
#include "scripting.h"
#include "events.h"
#include "args.h"
#include <algorithm>
#include <unistd.h>
#include <omp.h>

//----( options )-------------------------------------------------------------

const string g_date = get_date();

bool g_use_existing = false;

Rectangle g_screen_shape = Rectangle(0,0);
float g_framerate = 25.0f;

float g_tol = 1e-3f;
size_t g_min_bits = 4;
size_t g_max_bits = 10;
size_t g_points_factor = 8;
size_t g_fit_passes = 40;
size_t g_map_iters = 20;
size_t g_denoise_iters = 1;

string g_video_infile = "data/gloves/filtered.avi";
string g_audio_infile = "data/mono.raw";
string g_cloud_infile = "data/gloves/default.cloud";
string g_cloud_outfile = "data/gloves/default.cloud";
string g_dom_infile = "data/voice/default.cloud";
string g_cod_infile = "data/gloves/default.cloud";
string g_map_infile = "data/gloves/default.map";
string g_map_outfile = "data/gloves/default.map";
string g_prior_file = "data/gloves/prior.py";
string g_histogram_file = "data/gloves/histogram.py";
string g_raster_audio_subdir = "";
string g_voice_config = "config/default.voice.conf";

Cloud::VideoFormat g_video_format = Cloud::YUV_SINGLE;
bool g_recording_video = false; // TODO change to true when code is stable

namespace Gpu { void set_using_cuda (bool); }
void option_nogpu (Args & args) { Gpu::set_using_cuda(false); }

void option_zoom (Args & args)
{
  ASSERT(args.size() > 2, "no zoom values specified");

  size_t width = atoi(args.pop());
  size_t height = atoi(args.pop());
  g_screen_shape = Rectangle(width, height);
}

void option_cloud_file (Args & args)
{
  g_cloud_infile = g_cloud_outfile = args.pop();
}
void option_map_file (Args & args)
{
  g_map_infile = g_map_outfile = args.pop();
}

void option_video_format (Args & args)
{
  string name = args.pop();

  if (name == "yuv-single") g_video_format = Cloud::YUV_SINGLE; else
  if (name == "mono-single") g_video_format = Cloud::YUV_SINGLE; else
  if (name == "mono-batch") g_video_format = Cloud::MONO_BATCH; else

  ERROR("unrecognized format: " << name);
}

void option_recording_video (Args & args)
{
  g_recording_video = not g_recording_video;
}

//----( scripting tools )-----------------------------------------------------

namespace Streaming
{

size_t get_dim_from_video (Rectangle shape)
{
  switch (g_video_format) {
    case Cloud::YUV_SINGLE: return shape.size() * 3/2;
    case Cloud::MONO_SINGLE: return shape.size();
    case Cloud::MONO_BATCH: return shape.height();
    default: ERROR("unknown video format");
  }
}

//----( common patches )------------------------------------------------------

const Rectangle g_gloves_shape(24,32);

void run_from_filtered_gloves (Pushed<Gloves8Image> & out)
{
  Camera::set_config("config/gloves.camera.conf");

  FifthMono8CameraThread camera;
  GlovesFilter filter(camera.out);

  camera.out - filter;
  filter.out - out;

  if (g_recording_video) {

    string outfilename = "data/gloves/gloves-" + g_date + ".avi";

    LOG("recording gloves video to " << outfilename);

    Splitter<Mono8Image> camera_splitter;
    VideoEncoder encoder(outfilename, camera.out);

    camera.out - camera_splitter;
    camera_splitter.out1 - encoder;
    camera_splitter.out2 - filter;

    run();

  } else {

    run();
  }
}

//----( clouds )--------------------------------------------------------------

void run_cloud_new_points (Args & args)
{
  args
    .case_("-b", g_max_bits)
    .case_("-p", g_fit_passes)
    .case_("-t", g_tol)
    .case_("-o", g_cloud_outfile)
    .case_("-v", option_video_format)
    .default_break_else_repeat();

  VideoSequence seq;
  seq.add_files(args.pop_all(g_video_infile));
  size_t dim = get_dim_from_video(seq.shape());

  PRINT3(dim, g_max_bits, seq.shape());
  Cloud::Cloud cloud(dim, g_min_bits, seq.shape());

  cloud.init_points(seq, g_fit_passes);
  cloud.save(g_cloud_outfile);

  for (size_t bits = g_min_bits; bits < g_max_bits; ++bits) {
    cloud.grow_points(seq, g_fit_passes);
    cloud.save(g_cloud_outfile);
  }

  cloud.init_flow(seq, g_tol);
  cloud.save(g_cloud_outfile);
}

void run_cloud_grow_points (Args & args)
{
  size_t grow_steps = 1;

  args
    .case_("-g", grow_steps)
    .case_("-p", g_fit_passes)
    .case_("-t", g_tol)
    .case_("-i", g_cloud_infile)
    .case_("-o", g_cloud_outfile)
    .case_("-f", option_cloud_file)
    .case_("-v", option_video_format)
    .default_break_else_repeat();

  VideoSequence seq;
  seq.add_files(args.pop_all(g_video_infile));
  size_t video_dim = get_dim_from_video(seq.shape());

  Cloud::Cloud cloud(g_cloud_infile);
  ASSERT_EQ(video_dim, cloud.points().dim);

  for (size_t step = 0; step < grow_steps; ++step) {
    cloud.grow_points(seq, g_fit_passes);
    cloud.save(g_cloud_outfile);
  }
}

void run_cloud_fit (Args & args)
{
  args
    .case_("-p", g_fit_passes)
    .case_("-i", g_cloud_infile)
    .case_("-o", g_cloud_outfile)
    .case_("-f", option_cloud_file)
    .default_break_else_repeat();

  VideoSequence seq;
  seq.add_files(args.pop_all(g_video_infile));

  Cloud::Cloud cloud(g_cloud_infile);

  cloud.fit_points(seq, g_fit_passes);
  cloud.save(g_cloud_outfile);
}

void run_cloud_prior (Args & args)
{
  args
    .case_("-i", g_cloud_infile)
    .case_("-o", g_prior_file)
    .default_break_else_repeat();

  Cloud::Cloud cloud(g_cloud_infile);

  cloud.save_priors(g_prior_file);
}

void run_cloud_histogram (Args & args)
{
  args
    .case_("-i", g_cloud_infile)
    .case_("-o", g_histogram_file)
    .default_break_else_repeat();

  VideoSequence seq;
  seq.add_files(args.pop_all(g_video_infile));

  Cloud::Cloud cloud(g_cloud_infile);

  cloud.save_histograms(g_histogram_file, seq);
}

void run_cloud_flow (Args & args)
{
  args
    .case_("-t", g_tol)
    .case_("-i", g_cloud_infile)
    .case_("-o", g_cloud_outfile)
    .case_("-f", option_cloud_file)
    .case_("--use-existing", g_use_existing)
    .default_break_else_repeat();

  Cloud::Cloud cloud(g_cloud_infile);

  if (g_use_existing and not cloud.flow().empty()) {
    args.pop_all();
    return;
  }

  VideoSequence seq;
  seq.add_files(args.pop_all(g_video_infile));

  cloud.init_flow(seq, g_tol);
  cloud.save(g_cloud_outfile);
}

void run_cloud_new_map (Args & args)
{
  args
    .case_("-t", g_tol)
    .case_("-s", g_map_iters)
    .case_("-d", g_dom_infile)
    .case_("-c", g_cod_infile)
    .case_("-o", g_map_outfile)
    .default_break_else_repeat();

  Cloud::Controller map(g_dom_infile, g_cod_infile);

  map.optimize(g_tol, g_map_iters);
  map.save(g_map_outfile);
}

void run_cloud_map (Args & args)
{
  args
    .case_("-t", g_tol)
    .case_("-s", g_map_iters)
    .case_("-i", g_map_infile)
    .case_("-o", g_map_outfile)
    .case_("-f", option_map_file)
    .default_break_else_repeat();

  Cloud::Controller map(g_map_infile);

  map.optimize(g_tol, g_map_iters);
  map.save(g_map_outfile);
}

void run_cloud_crop_map (Args & args)
{
  args
    .case_("-i", g_map_infile)
    .case_("-o", g_map_outfile)
    .default_break_else_repeat();

  Cloud::Controller map(g_map_infile);
  map.crop();
  map.save(g_map_outfile);
}

//----( gloves commands )-----------------------------------------------------

void run_gloves_record (Args & args)
{
  string outfilename = "data/gloves/gloves-" + g_date + ".avi";

  LOG("recording gloves video to " << outfilename);

  Camera::set_config("config/gloves.camera.conf");

  FifthMono8CameraThread camera;
  Splitter<Mono8Image> splitter;
  VideoEncoder encoder(outfilename, camera.out);
  Shared<Mono8Image, size_t> image(camera.out.size());
  image.unsafe_access().set(0);
  ShowMono8Zoom screen(camera.out, g_screen_shape);

  camera.out - splitter;
  splitter.out1 - encoder;
  splitter.out2 - image;
  screen.in - image;

  run();
}

void run_gloves_play (Args & args)
{
  string infilename = args.pop();
  float speed = args.pop(1.0f);

  LOG("playing gloves video from " << infilename
      << " at " << speed << "x speed");

  VideoFile infile(infilename);
  VideoPlayer decoder(infile, speed);
  PRINT2(decoder.out.width(), decoder.out.height());

  if (infilename.find("filtered") == infilename.npos) {

    Shared<Mono8Image, size_t> image(decoder.mono_out.size());
    image.unsafe_access().set(0);
    ShowMono8Zoom screen(decoder.mono_out, g_screen_shape);

    decoder.mono_out - image;
    screen.in - image;

    run();

  } else {

    Shared<Gloves8Image, size_t> image(decoder.out.size());
    image.unsafe_access().set(0);
    Gloves8ToColor colorize(decoder.out);
    ShowRgb8Zoom screen(colorize.out, g_screen_shape);

    decoder.out - image;
    colorize.in - image;
    screen.in - colorize;

    run();
  }
}

void  run_gloves_recode (Args & args)
{
  size_t framerate = args.pop(DEFAULT_VIDEO_FRAMERATE);
  std::vector<string> infiles = args.pop_all();

  for (size_t f = 0; f < infiles.size(); ++f) {

    string infilename = infiles[f];
    std::ostringstream framerate_fps;
    framerate_fps << framerate << "fps-";
    string outfilename = prepend_to_filename(infilename, framerate_fps.str());
    LOG("recoding gloves video : " << infilename << " --> " << outfilename);

    VideoFile infile(infilename);
    VideoEncoder outfile(outfilename, infile.shape(), framerate);

    LOG("recoding " << infile.frames().size() << " images");

    typedef VideoFile::iterator Auto;
    for (Auto i = infile.begin(); i != infile.end(); ++i) {

      outfile.push(i->time, *(i->image));
    }
  }
}

void run_gloves_filter (Args & args)
{
  if (args.size()) { // show filtered video from file

    string filename = args.pop();
    float speed = args.pop(1.0f);
    LOG("filtering gloves video from " << filename
        << " at " << speed << "x speed");

    VideoFile infile(filename);
    VideoPlayer decoder(infile, speed);
    GlovesFilter filter(decoder.mono_out);
    Splitter<Gloves8Image> filter_splitter;
    PushedCast<Gloves8Image, Cloud::Point> cast;
    DistanceLogger logger(filter.out.size() * Gloves8Image::num_channels);
    Shared<Gloves8Image, size_t> image(filter.out.size());
    image.unsafe_access().set(0);
    Gloves8ToColor colorize(filter.out);
    ShowRgb8Zoom screen(colorize.out, g_screen_shape);

    decoder.mono_out - filter;
    filter.out - filter_splitter;
    filter_splitter.out1 - cast;
    cast.out - logger;
    filter_splitter.out2 - image;
    colorize.in - image;
    screen.in - colorize;

    run();

  } else { // show filtered live video & record gloves video to file

    string outfilename = "data/gloves/gloves-" + g_date + ".avi";

    LOG("recording gloves video to " << outfilename);

    Camera::set_config("config/gloves.camera.conf");

    FifthMono8CameraThread camera;
    Splitter<Mono8Image> camera_splitter;
    VideoEncoder encoder(outfilename, camera.out);
    GlovesFilter filter(camera.out);
    Splitter<Gloves8Image> filter_splitter;
    PushedCast<Gloves8Image, Cloud::Point> cast;
    DistanceLogger logger(filter.out.size() * Gloves8Image::num_channels);
    Shared<Gloves8Image, size_t> image(filter.out.size());
    image.unsafe_access().set(0);
    Gloves8ToColor colorize(filter.out);
    ShowRgb8Zoom screen(colorize.out, g_screen_shape);

    camera.out - camera_splitter;
    camera_splitter.out1 - encoder;
    camera_splitter.out2 - filter;
    filter.out - filter_splitter;
    filter_splitter.out1 - cast;
    cast.out - logger;
    filter_splitter.out2 - image;
    colorize.in - image;
    screen.in - colorize;

    run();
  }
}

void run_gloves_stats (Args & args)
{
  std::vector<string> infiles = args.pop_all("data/gloves/gloves.avi");

  Rectangle shape(2*24, 2*32);

  GlovesFilter filter(shape);
  PushedCast<Gloves8Image, Cloud::Point> cast;
  DistanceLogger logger(filter.out.size() * Gloves8Image::num_channels);

  filter.out - cast;
  cast.out - logger;

  for (size_t f = 0; f < infiles.size(); ++f) {
    string infilename = infiles[f];

    VideoFile infile(infilename);

    typedef VideoFile::iterator Auto;
    for (Auto i = infile.begin(); i != infile.end(); ++i) {

      filter.push(i->time, i->image->y);
    }

    logger.done();
  }
}

void run_gloves_convert (Args & args)
{
  std::vector<string> infiles = args.pop_all("data/gloves/gloves.avi");

  for (size_t f = 0; f < infiles.size(); ++f) {

    string infilename = infiles[f];
    string outfilename = prepend_to_filename(infilename, "filtered-");
    LOG("extracting gloves features : "
        << infilename << " --> " << outfilename);

    VideoFile infile(infilename);
    GlovesFilter filter(infile.shape());
    Splitter<Gloves8Image> splitter;
    VideoEncoder outfile(outfilename, filter.out);
    PushedCast<Gloves8Image, Cloud::Point> cast;
    DistanceLogger logger(filter.out.size() * Gloves8Image::num_channels);

    filter.out - splitter;
    splitter.out1 - outfile;
    splitter.out2 - cast;
    cast.out - logger;

    LOG("filtering " << infile.frames().size() << " images");

    typedef VideoFile::iterator Auto;
    for (Auto i = infile.begin(); i != infile.end(); ++i) {

      filter.push(i->time, i->image->y);
    }
  }
}

void run_gloves_show (Args & args)
{
  const char * filename = args.pop();

  exit(execlp(
      "mplayer", "mplayer",
      "-really-quiet", "-nolirc",
      "-nofs", "-fps" , "5", "-xy", "5",
      filename,
      NULL));
}

void run_gloves_denoise (Args & args)
{
  args
    .case_("-i", g_cloud_infile)
    .case_("-n", g_denoise_iters)
    .default_break_else_repeat();

  g_screen_shape = Rectangle(2 * g_screen_shape.width(),
                             g_screen_shape.height());

  Camera::set_config("config/gloves.camera.conf");

  Cloud::Cloud cloud(g_cloud_infile);
  const Rectangle shape = cloud.points().shape;

  FifthMono8CameraThread camera;
  GlovesFilter filter(camera.out);
  Splitter<Gloves8Image> splitter;
  PushedCast<Gloves8Image, Cloud::Point> image_to_point;
  Streaming::CloudDenoiser denoiser(cloud.points(), g_denoise_iters);
  VectorAsImage<Gloves8Image> point_to_image(shape);
  Yuv420p8Mosaic mosaic(shape, 2);
  Gloves8ToColor colorize(mosaic.shape_out);
  ShowRgb8Zoom screen(colorize.out, g_screen_shape);

  camera.out - filter;
  filter.out - splitter;
  splitter.out1 - image_to_point;
  image_to_point.out - denoiser;
  denoiser.out - point_to_image;
  point_to_image.out - mosaic.in(1);
  splitter.out2 - mosaic.in(0);
  colorize.in - mosaic;
  screen.in - colorize;

  run();
}

template<class Simulator>
void run_gloves_ (Args & args)
{
  args
    .case_("-i", g_cloud_infile)
    .case_("-f", g_framerate)
    .default_break_else_repeat();

  Cloud::Cloud cloud(g_cloud_infile);
  Cloud::JointPrior & flow = cloud.flow();
  ASSERT(not flow.empty(), "flow was empty");
  const Rectangle shape = cloud.points().shape;

  Simulator simulator(flow, g_framerate);
  VectorAsImage<Gloves8Image> cast(shape);
  Shared<Gloves8Image, size_t> image(shape.size());
  image.unsafe_access().set(0);
  Gloves8ToColor colorize(shape);
  ShowRgb8Zoom screen(shape, g_screen_shape);

  simulator.out - cast;
  cast.out - image;
  colorize.in - image;
  screen.in - colorize;

  run();
}

void run_gloves (Args & args)
{
  if (not g_screen_shape.size()) {
    g_screen_shape = Rectangle(240,320);
  }

  g_video_format = Cloud::YUV_SINGLE;
  g_cloud_infile = "data/gloves/default.cloud";
  g_cloud_outfile = "data/gloves/default.cloud";
  g_histogram_file = "data/gloves/histogram.py";

  args
    .case_("record", run_gloves_record)
    .case_("play", run_gloves_play)
    .case_("recode", run_gloves_recode)
    .case_("filter", run_gloves_filter)
    .case_("stats", run_gloves_stats)
    .case_("convert", run_gloves_convert)
    .case_("new", run_cloud_new_points)
    .case_("grow", run_cloud_grow_points)
    .case_("fit", run_cloud_fit)
    .case_("show", run_gloves_show)
    .case_("prior", run_cloud_prior)
    .case_("hist", run_cloud_histogram)
    .case_("flow", run_cloud_flow)
    .case_("denoise", run_gloves_denoise)
    .case_("sim", run_gloves_<CloudSimulator>)
    .case_("diffuse", run_gloves_<CloudDiffuser>)
    .default_error();
}

//----( voice commands )------------------------------------------------------

void run_voice_analyze (Args & args)
{
  args
    .case_("-i", g_audio_infile)
    .default_break_else_repeat();

  string infilename = g_audio_infile;
  DecompressAudio temp(infilename, false);

  MonoAudioFile infile(infilename);
  VoiceAnalyzer analyzer;

  SizedPort<Pushed<Vector<float> > > * monitor = NULL;
  SizedPort<Pushed<Vector<uint8_t> > > * monitor2 = NULL;

  if (args.size()) {
    string type = args.pop();
    if (type == "large") monitor = & analyzer.large_monitor; else
    if (type == "medium") monitor = & analyzer.medium_monitor; else
    if (type == "small") monitor = & analyzer.small_monitor; else
    if (type == "features") monitor2 = & analyzer.features_out; else
    if (type == "debug") monitor2 = & analyzer.debug_out; else
    ERROR("unknown monitor type: " << type);
  }

  if (monitor) {

    size_t duration = 800;
    bool transposed = true;

    MonoAudioFrame frame;
    Relay<MonoAudioFrame> relay(DEFAULT_AUDIO_FRAMERATE, frame);
    SweepVector sweep(monitor->size(), duration, transposed);

    relay.in - infile;
    relay.out - analyzer;
    * monitor - sweep;

    run();

    wait_for_keypress();

  } else if (monitor2) {

    size_t duration = 800;
    bool transposed = true;

    MonoAudioFrame frame;
    Relay<MonoAudioFrame> relay(DEFAULT_AUDIO_FRAMERATE, frame);
    SweepVector sweep(monitor2->size(), duration, transposed);

    relay.in - infile;
    relay.out - analyzer;
    * monitor2 - sweep;

    run();

    wait_for_keypress();

  } else {

    infile.out - analyzer;

    infile.run(DEFAULT_AUDIO_FRAMERATE);
  }
}

void run_voice_play (Args & args)
{
  string infilename = args.pop(g_audio_infile.c_str());
  size_t duration = 800;
  bool transposed = true;

  DecompressAudio temp(infilename, false);
  MonoAudioFile infile(infilename);
  PullSplitter<MonoAudioFrame> splitter;
  MonoAudioThread audio(false, true);
  VoiceAnalyzer analyzer;
  SweepVector sweep(analyzer.debug_out.size(), duration, transposed);

  audio.in - splitter;
  splitter.in - infile;
  splitter.out - analyzer;
  analyzer.debug_out - sweep;

  run();

  wait_for_keypress();
}

void run_voice_resynth (Args & args)
{
  string infilename = args.pop(g_audio_infile.c_str());
  size_t duration = 800;
  bool transposed = true;

  DecompressAudio temp(infilename, false);
  MonoAudioFile infile(infilename);
  VoiceAnalyzer analyzer;
  VoiceSynthesizer synthesizer;
  MonoAudioFrame frame;
  Relay<MonoAudioFrame> relay(DEFAULT_AUDIO_FRAMERATE, frame);
  Splitter<Vector<uint8_t> > splitter;
  SweepVector sweep(analyzer.debug_out.size(), duration, transposed);
  StereoAudioThread audio(false, true);

  relay.in - infile;
  relay.out - analyzer;
  analyzer.debug_out - splitter;
  splitter.out1 - sweep;
  splitter.out2 - synthesizer;
  audio.in - synthesizer;

  run();

  wait_for_keypress();
}

void voice_convert_one (string infilename)
{
  // TODO recover more gracefully from errors like the following:
  //   [avi @ 0x7f5c70005db0]Could not find codec parameters
  //     (Video: ffv1, 768x128)
  //   ERROR failed to find stream information
	//   compress.cpp : 414
	//   Streaming::VideoFile::VideoFile(std::string)
  DecompressAudio temp(infilename, false);

  string outfilename
    = "data/raster_audio/"
    + g_raster_audio_subdir + "/"
    + strip_to_stem(infilename) + ".avi";
  LOG("extracting voice features : "
      << infilename << " --> " << outfilename);

  MonoAudioFile infile(infilename);
  VoiceAnalyzer analyzer;
  VoiceFeatureBuffer buffer(analyzer);
  VideoEncoder outfile(outfilename, buffer.out);

  infile.out - analyzer;
  analyzer.features_out - buffer;
  buffer.out - outfile;

  infile.run(DEFAULT_AUDIO_FRAMERATE);
}

void run_voice_convert (Args & args)
{
  args
    .case_("-d", g_raster_audio_subdir)
    .default_break_else_repeat();

  std::vector<string> infiles = args.pop_all(g_audio_infile);
  const size_t num_infiles = infiles.size();

  Timer timer;

  #pragma omp parallel for schedule(dynamic, 1)
  for (size_t i = 0; i < num_infiles; ++i) {
    voice_convert_one(infiles[i]);
  }

  LOG("converting " << infiles.size() << " files took "
      << timer.elapsed() << " sec");
}

void run_voice_show (Args & args)
{
  const char * filename = args.pop();

  exit(execlp(
      "mplayer", "mplayer",
      "-really-quiet", "-nolirc",
      "-nofs", "-fps" , "1", "-xy", "2",
      filename,
      NULL));
}

void run_voice_sim (Args & args)
{
  args
    .case_("-i", g_cloud_infile)
    .case_("-f", g_framerate)
    .default_break_else_repeat();

  size_t duration = 800;
  bool transposed = true;

  Cloud::Cloud cloud(g_cloud_infile);
  Cloud::JointPrior & flow = cloud.flow();
  ASSERT(not flow.empty(), "flow was empty");
  const Rectangle shape = cloud.points().shape;

  // TODO factor CloudSimulator into GlovesSimulator and VoiceSimulator,
  //   and use Voice::FeatureProcessor::update_history(...),
  //   as is done in Streaming::GlovesToVoice
  CloudSimulator simulator(flow, g_framerate);

  Splitter<Vector<uint8_t> > splitter;
  SweepVector sweep(simulator.out.size(), duration, transposed);
  VoiceSynthesizer synthesizer;
  StereoAudioThread audio(false, true);

  simulator.out - splitter;
  splitter.out1 - sweep;
  splitter.out2 - synthesizer;
  audio.in - synthesizer;

  run();
}

void run_voice (Args & args)
{
  g_video_format = Cloud::MONO_BATCH;
  g_video_infile = "data/raster_audio/test.avi";
  g_cloud_infile = "data/voice/default.cloud";
  g_cloud_outfile = "data/voice/default.cloud";
  g_histogram_file = "data/voice/histogram.py";

  args
    .case_("analyze", run_voice_analyze)
    .case_("play", run_voice_play)
    .case_("resynth", run_voice_resynth)
    .case_("convert", run_voice_convert)
    .case_("new", run_cloud_new_points)
    .case_("grow", run_cloud_grow_points)
    .case_("fit", run_cloud_fit)
    .case_("show", run_voice_show)
    .case_("prior", run_cloud_prior)
    .case_("hist", run_cloud_histogram)
    .case_("flow", run_cloud_flow)
    .case_("sim", run_voice_sim)
    .default_error();
}

//----( gloves <- gloves )----------------------------------------------------

void run_gg_play (Args & args)
{
  args
    .case_("-r", option_recording_video)
    .case_("-i", g_map_infile)
    .default_break_else_repeat();

  if (not g_screen_shape.size()) {
    g_screen_shape = Rectangle(2 * 240,320);
  }

  Cloud::Controller controller(g_map_infile);

  Splitter<Gloves8Image> splitter;
  GlovesToGloves map(controller);
  ASSERT_EQ(g_gloves_shape, map.out);
  Yuv420p8Mosaic mosaic(g_gloves_shape, 2);
  Gloves8ToColor colorize(mosaic.shape_out);
  ShowRgb8Zoom screen(colorize.out, g_screen_shape);

  splitter.out1 - map;
  splitter.out2 - mosaic.in(0);
  map.out - mosaic.in(1);
  colorize.in - mosaic;
  screen.in - colorize;

  run_from_filtered_gloves(splitter);
}

void run_gg (Args & args)
{
  g_cod_infile = "data/gloves/default.cloud";
  g_dom_infile = "data/gloves/default.cloud";
  g_map_outfile = "data/gg/default.map";
  g_map_infile = "data/gg/default.map";

  args
    .case_("new", run_cloud_new_map)
    .case_("map", run_cloud_map)
    .case_("crop", run_cloud_crop_map)
    .case_("play", run_gg_play)
    .default_error();
}

//----( gloves <- voice )-----------------------------------------------------

void run_gv_show (Args & args)
{
  args
    .case_("-r", option_recording_video)
    .case_("-i", g_map_infile)
    .default_break_else_repeat();

  size_t duration = 800;
  bool transposed = true;

  Cloud::Controller controller(g_map_infile);
  GlovesToVoice map(controller);
  SweepVector sweep(map.out.size(), duration, transposed);

  map.out - sweep;

  run_from_filtered_gloves(map);

  wait_for_keypress();
}

void run_gv_play (Args & args)
{
  args
    .case_("-r", option_recording_video)
    .case_("-i", g_map_infile)
    .case_("-c", g_voice_config)
    .default_break_else_repeat();

  size_t duration = 800;
  bool transposed = true;

  Cloud::Controller controller(g_map_infile);
  GlovesToVoice map(controller);
  Splitter<Cloud::Point> splitter;
  VoiceSynthesizer synthesizer(g_voice_config.c_str());
  StereoAudioThread audio(false, true);
  SweepVector sweep(map.out.size(), duration, transposed);

  map.out - splitter;
  splitter.out1 - synthesizer;
  splitter.out2 - sweep;
  audio.in - synthesizer;

  run_from_filtered_gloves(map);

  wait_for_keypress();
}

void run_gv (Args & args)
{
  g_cod_infile = "data/gloves/default.cloud";
  g_dom_infile = "data/voice/default.cloud";
  g_map_outfile = "data/gv/default.map";
  g_map_infile = "data/gv/default.map";

  args
    .case_("new", run_cloud_new_map)
    .case_("map", run_cloud_map)
    .case_("show", run_gv_show)
    .case_("crop", run_cloud_crop_map)
    .case_("play", run_gv_play)
    .default_error();
}

} // namespace Streaming

using namespace Streaming;

//----( harness )-------------------------------------------------------------

const char * help_message =
"Usage: learn [OPTIONS] COMMAND_SEQUENCE [ARGS]"
"\nOptions:"
"\n  nogpu"
"\n  zoom MIN_WIDTH MIN_HEIGHT"
"\nCommand Sequences:"
"\n  help"
"\n  gloves"
"\n    record"
"\n    play VIDEO"
"\n    recode FRAMERATE [VIDEOS]"
"\n    filter [VIDEO = live]"
"\n    stats [VIDEO = data/gloves/gloves.avi]"
"\n    convert [VIDEOS = data/gloves/gloves.avi]"
"\n    stats [VIDEOS = data/gloves/gloves.avi]"
"\n    new [-b MAX_BITS -p PASSES -t TOL -o OUTFILE] [VIDEOS]"
"\n    grow [-g STEPS -p PASSES -i INFILE -o OUTFILE -f IOFILE] [VIDEOS]"
"\n    fit [-p PASSES -i INFILE -o OUTFILE -f IOFILE] [VIDEOS]"
"\n    show FILENAME"
"\n    prior [-i INFILE -o OUTFILE]"
"\n    hist [-i INFILE -o OUTFILE] [VIDEOS]"
"\n    flow [-t TOL -i INFILE -o OUTFILE -f IOFILE --use-existing] [VIDEOS]"
"\n    denoise [-i INFILE -n ITERS]"
"\n    sim [-i INFILE -f FRAMERATE]"
"\n    diffuse [-i INFILE -f FRAMERATE]"
"\n  voice"
"\n    analyze [-i AUDIOFILE] [large | medium | small | features]"
"\n    play [AUDIOFILE]"
"\n    resynth [AUDIOFILE]"
"\n    convert [-d SUBDIR] [AUDIOFILES...]"
"\n    new [-b MAX_BITS -p PASSES -t TOL -o OUTFILE] [VIDEOS]"
"\n    grow [-g STEPS -p PASSES -i INFILE -o OUTFILE -f IOFILE] [VIDEOS]"
"\n    fit [-p PASSES -i INFILE -o OUTFILE -f IOFILE] [VIDEOS]"
"\n    show FILENAME"
"\n    prior [-i INFILE -o OUTFILE]"
"\n    hist [-i INFILE -o OUTFILE] [VIDEOS]"
"\n    flow [-t TOL -i INFILE -o OUTFILE -f IOFILE --use-existing] [VIDEOS]"
"\n    sim [-i INFILE -f FRAMERATE]"
"\n  gg"
"\n    new [-t TOL -s STEPS -d DOMAIN -c CODOMAIN -o OUTFILE]"
"\n    map [-t TOL -s STEPS -i INFILE -o OUTFILE -f IOFILE]"
"\n    crop [-i INFILE -o OUTFILE]"
"\n    play [-r -i INFILE]"
"\n  gv"
"\n    new [-t TOL -s STEPS -d DOMAIN -c CODOMAIN -o OUTFILE]"
"\n    map [-t TOL -s STEPS -i INFILE -o OUTFILE -f IOFILE]"
"\n    show [-r -i INFILE]"
"\n    crop [-i INFILE] -o OUTFILE]"
"\n    play [-r -i INFILE -c VOICE_CONFIG]"
;

void run_help (Args & args) { LOG(help_message); }

int main (int argc, char ** argv)
{
  omp_set_num_threads(1 + omp_get_num_procs());

  LOG(kazoo_logo);
  chdir_kazoo();

  Args args(argc, argv, help_message);

  args
    .case_("zoom", option_zoom)
    .case_("nogpu", option_nogpu)
    .default_break_else_repeat();

  args
    .case_("help", run_help)
    .case_("gloves", run_gloves)
    .case_("voice", run_voice)
    .case_("gg", run_gg)
    .case_("gv", run_gv)
    .default_(run_help);

  return 0;
}

