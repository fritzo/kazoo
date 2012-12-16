
#include "common.h"
#include "streaming.h"
#include "streaming_shared.h"
#include "streaming_audio.h"
#include "streaming_video.h"
#include "streaming_camera.h"
#include "streaming_devices.h"
#include "streaming_synthesis.h"
#include "flock.h"
#include "tracker.h"
#include "table.h"
#include "bucket.h"
#include "ball.h"
#include "sing.h"
#include "movement.h"
#include "dirt.h"
#include "playback.h"
#include "events.h"
#include "args.h"

Rectangle g_screen_shape = Rectangle(0,0);
bool g_showing = false;
bool g_hearing = false;
bool g_audio_stereo = true;
bool g_deaf = false;
bool g_spectrum = false;
bool g_sweep = false;
bool g_power = false;
const char * g_peaks_filename = "data/peaks.seq";

void g_audio_mono (Args & args) { g_audio_stereo = false; }

namespace Streaming
{

void run_spectrum (
    Args & args,
    Port<Pushed<StereoAudioFrame> > & port,
    Pushed<StereoAudioFrame> & pushed)
{
  if (g_sweep) {

    size_t size = args.pop(512);
    size_t duration = args.pop(1024);

    Splitter<StereoAudioFrame> splitter;
    Psychogram psychogram(size);
    SweepVector screen(size, duration);

    port - splitter;
    splitter.out1 - pushed;
    splitter.out2 - psychogram;
    psychogram.out - screen;

    run();

  } else if (g_spectrum) {

    size_t size = args.pop(512);
    size_t duration = args.pop(512);

    Splitter<StereoAudioFrame> splitter;
    Psychogram psychogram(size);
    History history(size, duration);
    ShowMonoZoom screen(history, g_screen_shape);

    port - splitter;
    splitter.out1 - pushed,
    splitter.out2 - psychogram;
    psychogram.out - history;
    screen.in - history;

    run();

  } else if (g_power) {

    size_t width = args.pop(640);
    size_t height = args.pop(480);
    Rectangle shape(width, height);
    const float timescale(2.0f / DEFAULT_SCREEN_FRAMERATE);

    Splitter<StereoAudioFrame> splitter;
    PowerMeter power;
    Denoiser denoiser;
    SharedMaxLowpass beat(timescale);
    GraphValue graph(shape);

    port - splitter;
    splitter.out1 - pushed;
    splitter.out2 - power;
    power.out - denoiser;
    denoiser.out - beat;
    graph.in - beat;

    run();

  } else {

    port - pushed;

    run();
  }
}

void run_spectrum (
    Args & args,
    Port<Pulled<StereoAudioFrame> > & port,
    Pulled<StereoAudioFrame> & pulled)
{
  if (g_sweep) {

    size_t size = args.pop(512);
    size_t duration = args.pop(1024);

    PullSplitter<StereoAudioFrame> splitter;
    Psychogram psychogram(size);
    SweepVector screen(size, duration);

    splitter.in - pulled,
    port - splitter;
    splitter.out - psychogram;
    psychogram.out - screen;

    run();

  } else if (g_spectrum) {

    size_t size = args.pop(512);
    size_t duration = args.pop(512);

    PullSplitter<StereoAudioFrame> splitter;
    Psychogram psychogram(size);
    History history(size, duration);
    ShowMonoZoom screen(history, g_screen_shape);

    splitter.in - pulled,
    port - splitter;
    splitter.out - psychogram;
    psychogram.out - history;
    screen.in - history;

    run();

  } else if (g_power) {

    size_t width = args.pop(640);
    size_t height = args.pop(480);
    Rectangle shape(width, height);
    const float timescale(2.0f / DEFAULT_SCREEN_FRAMERATE);

    PullSplitter<StereoAudioFrame> splitter;
    PowerMeter power;
    Denoiser denoiser;
    SharedMaxLowpass beat(timescale);
    GraphValue graph(shape);

    splitter.in - pulled;
    port - splitter;
    splitter.out - power;
    power.out - denoiser;
    denoiser.out - beat;
    graph.in - beat;

    run();

  } else {

    port - pulled;

    run();
  }
}

void run_vocoder (
    Args & args,
    Vocoder & vocoder,
    Port<Pulled<StereoAudioFrame> > & sound_out)
{
  if (g_showing) {

    size_t duration = args.pop(512);
    float rate_scale = args.pop(0.5f);

    float rate = rate_scale * DEFAULT_AUDIO_FRAMERATE;

    Relay<RgbImage, size_t> relay(rate, SYNTHESIS_VOCODER_SIZE);
    RgbHistory history(SYNTHESIS_VOCODER_SIZE, duration);
    ShowRgb screen(history);

    relay.in - vocoder;
    relay.out - history;
    screen.in - history;
    sound_out - vocoder;

    run();

  } else {

    run_spectrum(args, sound_out, vocoder);
  }
}

//----( video commands )------------------------------------------------------

//----( simulation )----

template<class System>
void run_sim_system (Args & args)
{
  float timescale = args.pop(2.0f);
  size_t width = args.pop(160);
  size_t height = args.pop(120);
  size_t framerate = args.pop(DEFAULT_VIDEO_FRAMERATE);

  Random01Video random(Rectangle(width, height), framerate);
  System system(random.out, timescale * framerate);
  Shared<MonoImage, size_t> image(system.out.size());
  ShowMonoZoom screen(random.out, g_screen_shape);

  random.out - system;
  system.out - image;
  screen.in - image;

  run();
}

void run_sim_show_bend (Rectangle shape, Vector<float> & bend)
{
  const size_t size = shape.size();
  bend /= sqrtf(max_norm_squared(bend));

  Vector<float> red(size);
  Vector<float> green(size);
  Vector<float> blue(size);

  for (size_t i = 0; i < size; ++i) {
    float b = bend[i] / sqrtf(fabsf(bend[i]) + 1e-8f);
    red[i] = max(b, 0.0f);
    blue[i] = max(-b, 0.0f);
    green[i] = sqr(b);
  }

  Screen screen(shape);
  screen.draw(red, green, blue, true);
  screen.update();

  wait_for_keypress();
}

template<class Oscillator>
void run_sim_tongues_ (Args & args)
{
  float acuity = args.pop(7.0f);
  float max_strength = args.pop(2.0f);
  size_t width = args.pop(200);
  size_t height = args.pop(50);
  Rectangle shape(width, height);

  Synchronized::ArnoldTongues<Oscillator> arnold(shape);
  arnold.tongues(acuity, max_strength);

  run_sim_show_bend(shape, arnold.bend);
}

void run_sim_tongues (Args & args)
{
  args
    .case_("phasor", run_sim_tongues_<Synchronized::Phasor>)
    .case_("syncopator", run_sim_tongues_<Synchronized::Syncopator>)
    .case_("shepard4", run_sim_tongues_<Synchronized::Shepard4>)
    .case_("shepard7", run_sim_tongues_<Synchronized::Shepard7>)
    .case_("geometric", run_sim_tongues_<Synchronized::Geometric>)
    .case_("boltz", run_sim_tongues_<Synchronized::Boltz>)
    .case_("phasor2", run_sim_tongues_<Synchronized::Phasor2>)
    .default_error();
}

template<class Oscillator>
void run_sim_keys_ (Args & args)
{
  float min_acuity = args.pop(1.0f);
  float max_acuity = args.pop(12.0f);
  size_t width = args.pop(200);
  size_t height = args.pop(50);
  Rectangle shape(width, height);

  Synchronized::ArnoldTongues<Oscillator> arnold(shape);
  arnold.keys(min_acuity, max_acuity);

  run_sim_show_bend(shape, arnold.bend);
}

void run_sim_keys (Args & args)
{
  args
    .case_("phasor", run_sim_keys_<Synchronized::Phasor>)
    .case_("syncopator", run_sim_keys_<Synchronized::Syncopator>)
    .case_("shepard4", run_sim_keys_<Synchronized::Shepard4>)
    .case_("shepard7", run_sim_keys_<Synchronized::Shepard7>)
    .case_("geometric", run_sim_keys_<Synchronized::Geometric>)
    .case_("boltz", run_sim_keys_<Synchronized::Boltz>)
    .case_("phasor2", run_sim_keys_<Synchronized::Phasor2>)
    .default_error();
}

template<class Oscillator>
void run_sim_islands_ (Args & args)
{
  float acuity = args.pop(7.0f);
  float strength_scale = args.pop(1.0f);
  size_t width = args.pop(128);
  size_t height = args.pop(128);
  Rectangle shape(width, height);

  Synchronized::ArnoldTongues<Oscillator> arnold(shape);
  arnold.islands(acuity, strength_scale);

  Vector<float> & bend = arnold.bend;
  bend *= -1.0f;
  affine_to_01(bend);

  Screen screen(shape);
  screen.draw(bend, true);
  screen.update();

  wait_for_keypress();
}

void run_sim_islands (Args & args)
{
  args
    .case_("phasor", run_sim_islands_<Synchronized::Phasor>)
    .case_("syncopator", run_sim_islands_<Synchronized::Syncopator>)
    .case_("shepard4", run_sim_islands_<Synchronized::Shepard4>)
    .case_("shepard7", run_sim_islands_<Synchronized::Shepard7>)
    .case_("geometric", run_sim_islands_<Synchronized::Geometric>)
    .case_("boltz", run_sim_islands_<Synchronized::Boltz>)
    .case_("phasor2", run_sim_islands_<Synchronized::Phasor2>)
    .default_error();
}

void run_sim_mouse (Args & args)
{
  size_t width = args.pop(640);
  size_t height = args.pop(480);

  Rectangle shape(width, height);

  MouseTest test(shape);
  ShowMono screen(shape);

  screen.in - test;

  run();
}

void run_sim (Args & args)
{
  args
    .case_("reassign", run_sim_system<ReassignAccum>)
    .case_("attract", run_sim_system<AttractRepelAccum>)
    .case_("tongues", run_sim_tongues)
    .case_("keys", run_sim_keys)
    .case_("islands", run_sim_islands)
    .case_("mouse", run_sim_mouse)
    .default_error();
}

//----( video processing )----

template<class Source>
void run_video_source_show (Args & args)
{
  Source source;
  Shared<MonoImage, size_t> image(source.out.size());
  ShowMonoZoom screen(source.out, g_screen_shape);

  source.out - image;
  screen.in - image;

  run();
}

template<class Source, class Filter>
void run_video_source_filter (Args & args)
{
  Source source;
  Filter filter(source.out);
  Shared<MonoImage, size_t> image(filter.out.size());
  ShowMonoZoom screen(filter.out, g_screen_shape);

  source.out - filter;
  filter.out - image;
  screen.in - image;

  run();
}

template<class Source, class Blur>
void run_video_source_blur (Args & args)
{
  size_t radius = args.pop(2);

  Source source;
  Blur blur(source.out, radius);
  Shared<MonoImage, size_t> image(blur.out.size());
  ShowMonoZoom screen(blur.out, g_screen_shape);

  source.out - blur;
  blur.out - image;
  screen.in - image;

  run();
}

template<class Source>
void run_video_source_hands (Args & args)
{
  Source source;
  EnhanceHands hands(source.out);
  Shared<HandImage, size_t> image(hands.out.size());
  HandsToColor colorize(hands.out);
  ShowRgbZoom screen(colorize.out, g_screen_shape);

  source.out - hands;
  hands.out - image;
  colorize.in - image;
  screen.in - colorize;

  run();
}

template<class Source>
void run_video_source_flow (Args & args)
{
  size_t highpass_radius = args.pop(OPTICAL_FLOW_HIGHPASS_RADIUS);

  Source source;
  OpticalFlow flow(source.out, highpass_radius);
  Shared<FlowImage, size_t> image(flow.out.size());
  FlowToColor colorize(flow.out);
  ShowRgbZoom screen(colorize.out, g_screen_shape);

  source.out - flow;
  flow.out - image;
  colorize.in - image;
  screen.in - colorize;

  run();
}

template<class Source>
void run_video_source_krig_flow (Args & args)
{
  size_t spacescale = args.pop(OPTICAL_FLOW_SPACESCALE);
  size_t highpass_radius = args.pop(OPTICAL_FLOW_HIGHPASS_RADIUS);

  Source source;
  KrigOpticalFlow flow(source.out, spacescale, highpass_radius);
  FlowToColor colorize(flow.out);
  Shared<FlowImage, size_t> image(flow.out.size());
  ShowRgbZoom screen(colorize.out, g_screen_shape);

  source.out - flow;
  flow.out - image;
  colorize.in - image;
  screen.in - colorize;

  run();
}

template<class Source>
void run_video_source_gloves_flow (Args & args)
{
  size_t spacescale = args.pop(OPTICAL_FLOW_SPACESCALE);
  size_t highpass_radius = args.pop(OPTICAL_FLOW_HIGHPASS_RADIUS);

  Source source;
  GlovesFlow gloves(source.out, spacescale, highpass_radius);
  GlovesToColor colorize(gloves.out);
  Shared<GlovesImage, size_t> image(gloves.out.size());
  ShowRgbZoom screen(colorize.out, g_screen_shape);

  source.out - gloves;
  gloves.out - image;
  colorize.in - image;
  screen.in - colorize;

  run();
}

template<class Source>
void run_video_source_filter_flow (Args & args)
{
  float process_noise = args.pop(OPTICAL_FLOW_PROCESS_NOISE);
  size_t highpass_radius = args.pop(OPTICAL_FLOW_HIGHPASS_RADIUS);
  float prior = args.pop(OPTICAL_FLOW_PRIOR_PER_PIX);

  Source source;
  FilterOpticalFlow flow(
      source.out,
      process_noise,
      highpass_radius,
      prior);
  FlowToColor colorize(flow.out);
  Shared<FlowImage, size_t> image(colorize.out.size());
  ShowRgbZoom screen(flow.out, g_screen_shape);

  source.out - flow;
  flow.out - image;
  colorize.in - image;
  screen.in - colorize;

  run();
}

template<class Source>
void run_video_source (Args & args)
{
  args
    .case_("transpose", run_video_source_filter<Source, Transpose>)
    .case_("square", run_video_source_blur<Source, SquareBlur>)
    .case_("quadratic", run_video_source_blur<Source, QuadraticBlur>)
    .case_("highpass", run_video_source_blur<Source, ImageHighpass>)
    .case_("points", run_video_source_filter<Source, EnhancePoints>)
    .case_("hands", run_video_source_hands<Source>)
    .case_("flow", run_video_source_flow<Source>)
    .case_("krig_flow", run_video_source_krig_flow<Source>)
    .case_("gloves", run_video_source_gloves_flow<Source>)
    .case_("filter_flow", run_video_source_filter_flow<Source>)
    .default_(run_video_source_show<Source>);
}

void run_video_mosaic (Args & args)
{
  size_t num_tiles = args.pop(1);
  ASSERT_LT(0, num_tiles);

  std::vector<CameraThread *> cameras;
  for (size_t i = 0; i < num_tiles; ++i) {
    cameras.push_back(new CameraThread());
  }

  Mosaic mosaic(cameras[0]->out, num_tiles);
  for (size_t i = 0; i < num_tiles; ++i) {
    cameras[i]->out - mosaic.in(i);
  }

  ShowMono screen(mosaic.shape_out);
  NormalizeTo01 normalize(mosaic.shape_out);;
  screen.in - normalize;
  normalize.in - mosaic;

  run();

  for (size_t i = 0; i < num_tiles; ++i) {
    delete cameras[i];
  }
}

void run_video (Args & args)
{
  args
    .case_("camera", run_video_source<CameraThread>)
    .case_("region", run_video_source<RegionThread>)
    .case_("crop", run_video_source<RegionCropThread>)
    .case_("mask", run_video_source<RegionMaskThread>)
    .case_("sub", run_video_source<RegionMaskSubThread>)
    .case_("ceil", run_video_source<RegionMaskCeilThread>)
    .case_("disk", run_video_source<DiskThread>)
    .case_("fifth", run_video_source<FifthCameraThread>)
    .case_("change", run_video_source<ChangeThread>)
    .case_("mosaic", run_video_mosaic)
    .default_error();
}

//----( audio commands )------------------------------------------------------

void run_audio_wire (Args & args)
{
  StereoAudioThread audio;
  AudioWire wire;

  audio.out - wire;

  run_spectrum(args, audio.in, wire);
}

void run_audio_gain (Args & args)
{
  StereoAudioThread audio;
  MicGain mic_gain;
  AudioWire wire;

  audio.out - mic_gain;
  mic_gain.out - wire;

  run_spectrum(args, audio.in, wire);
}

void run_audio_play (Args & args)
{
  args
    .case_("--mono", g_audio_mono)
    .default_break_else_repeat();

  const char * filename = args.pop("data/test.raw");

  if (g_audio_stereo) {

    StereoAudioFile file(filename);
    StereoAudioThread speaker(false, true);

    run_spectrum(args, speaker.in, file);

  } else {

    MonoAudioFile file(filename);
    MonoAudioThread speaker(false, true);

    speaker.in - file;

    run();
  }
}

template<class S>
void run_audio_spectrogram (Args & args)
{
  size_t size = args.pop(512);
  size_t duration = args.pop(512);

  StereoAudioThread audio(true, false);
  S spectrogram(size);
  History history(size, duration);
  ShowMonoZoom screen(history, g_screen_shape);

  audio.out - spectrogram;
  spectrogram.out - history;
  screen.in - history;

  run();
}

void run_audio_beater (Args & args)
{
  bool coalesce = args.pop(true);
  bool blur_factor = args.pop(1.0f);

  StereoAudioThread audio(not g_deaf);
  Beater beater(coalesce, blur_factor);
  ShowMonoZoom screen(beater.shape, g_screen_shape);
  PulledCast<Beater::Timbre, Vector<float> > cast;
  HearVector hear(beater.size);
  SpeakerGain gain;

  screen.in - beater;
  cast.in - beater;
  hear.in - cast;
  gain.in - hear;
  audio.in - gain;

  if (g_deaf) {

    run();

  } else {

    PowerMeter power;
    Denoiser denoiser;
    Shared<float> impact(0);

    audio.out - power;
    power.out - denoiser;
    denoiser.out - impact;
    beater.power_in - impact;

    run();
  }
}

void run_audio_beat (Args & args)
{
  StereoAudioThread audio;
  PowerMeter power;
  EnergyToBeat energy_to_beat;

  audio.out - power;
  power.out - energy_to_beat;

  if (g_showing) {

    float radius = args.pop(2.0f);
    size_t diameter = args.pop(512);
    Rectangle shape(diameter, diameter);

    Oscilloscope oscilloscope(shape, radius);
    ShowMonoZoom screen(shape, g_screen_shape);

    energy_to_beat.beat_monitor - oscilloscope;
    screen.in - oscilloscope;
    audio.in - energy_to_beat;

    run();

  } else {

    run_spectrum(args, audio.in, energy_to_beat);
  }
}

template<class Generator>
void run_audio_vocoder_ (Args & args)
{
  size_t num_tones = args.pop(1);
  float period = args.pop(4.0f);

  Generator generator(period, num_tones);
  Vocoder vocoder;
  StereoAudioThread audio(false, true);

  vocoder.in - generator;

  run_vocoder(args, vocoder, audio.in);
}

void run_audio_vocoder (Args & args)
{
  args
    .case_("chirp", run_audio_vocoder_<SimVocoderChirp>)
    .case_("chord", run_audio_vocoder_<SimVocoderChord>)
    .case_("drone", run_audio_vocoder_<SimVocoderDrone>)
    .case_("band", run_audio_vocoder_<SimVocoderNoiseBand>)
    .default_(run_audio_vocoder_<SimVocoderChirp>);
}

void run_audio_tempo (Args & args)
{
  const char * filename = args.pop("data/test.raw");

  StereoAudioFile file(filename);
  PowerMeter power;
  Denoiser denoiser;
  TempoFlockTest flock;

  file.out - power;
  power.out - denoiser;
  denoiser.out - flock;

  file.run();
}

void run_audio (Args & args)
{
  args
    .case_("wire", run_audio_wire)
    .case_("gain", run_audio_gain)
    .case_("play", run_audio_play)
    .case_("fourier", run_audio_spectrogram<Spectrogram>)
    .case_("fourier2", run_audio_spectrogram<Spectrogram2>)
    .case_("maskogram", run_audio_spectrogram<Maskogram>)
    .case_("psychogram", run_audio_spectrogram<Psychogram>)
    .case_("vocoder", run_audio_vocoder)
    .case_("beater", run_audio_beater)
    .case_("beat", run_audio_beat)
    .case_("tempo", run_audio_tempo)
    .default_error();
}

//----( tracker commands )----------------------------------------------------

void run_track_table (Args & args)
{
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);

  RegionThread camera;
  EnhancePoints points(camera.out);
  Splitter<MonoImage> point_splitter;
  PeakDetector peaks(points.out, capacity);
  Splitter<Image::Peaks> peak_splitter;
  Tracking::Tracker tracker(capacity);
  TrackVisualizer vis(points.out, capacity);
  ShowRgbZoom screen(points.out, g_screen_shape);

  camera.out - points;
  points.out - point_splitter;
  point_splitter.out1 - peaks;
  peaks.out - peak_splitter;
  peak_splitter.out1 - tracker;

  point_splitter.out2 - vis.image_in;
  peak_splitter.out2 - vis.detections_in;
  vis.tracks_in - tracker;
  screen.in - vis;

  run();
}

template<class Thing>
void run_track_thing (Args & args)
{
  size_t capacity = args.pop(BUCKET_FINGER_CAPACITY);

  Thing thing;
  Splitter<MonoImage> point_splitter;
  PeakDetector peaks(thing.out, capacity);
  Splitter<Image::Peaks> peak_splitter;
  Tracking::Tracker tracker(capacity);
  TrackVisualizer vis(thing.out, capacity);
  ShowRgbZoom screen(thing.out, g_screen_shape);

  thing.out - point_splitter;
  point_splitter.out1 - peaks;
  peaks.out - peak_splitter;
  peak_splitter.out1 - tracker;

  point_splitter.out2 - vis.image_in;
  peak_splitter.out2 - vis.detections_in;
  vis.tracks_in - tracker;
  screen.in - vis;

  run();
}

void run_track_table_age (Args & args)
{
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);
  float age_scale = args.pop(DEFAULT_TRACKER_VIS_AGE_SCALE);

  RegionThread camera;
  EnhancePoints points(camera.out);
  PeakDetector peaks(points.out, capacity);
  Tracking::Tracker tracker(capacity);
  TrackAgeVisualizer vis(points.out, capacity, age_scale);
  ShowRgbZoom screen(points.out, g_screen_shape);

  camera.out - points;
  points.out - peaks;
  peaks.out - tracker;

  vis.tracks_in - tracker;
  screen.in - vis;

  run();
}

template<class Thing>
void run_track_thing_age (Args & args)
{
  size_t capacity = args.pop(BUCKET_FINGER_CAPACITY);
  float age_scale = args.pop(DEFAULT_TRACKER_VIS_AGE_SCALE);

  Thing thing;
  PeakDetector peaks(thing.out, capacity);
  Tracking::Tracker tracker(capacity);
  TrackAgeVisualizer vis(thing.out, capacity, age_scale);
  ShowRgbZoom screen(thing.out, g_screen_shape);

  thing.out - peaks;
  peaks.out - tracker;

  vis.tracks_in - tracker;
  screen.in - vis;

  run();
}

void run_track_calibrate (Args & args)
{
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);

  RegionThread camera;
  Rectangle shape = camera.out.transposed();
  Splitter<MonoImage> image_splitter;
  Transpose transpose(camera.out);
  Shared<MonoImage, size_t> image(shape.size());
  NormalizeTo01 normalize(shape);
  EnhancePoints enhance_points(camera.out);
  Splitter<MonoImage> points_splitter;
  Shared<MonoImage, size_t> points(shape.size());
  PeakDetector peaks(shape, capacity);
  Splitter<Image::Peaks> peak_splitter;
  Tracking::Tracker tracker(capacity);
  Calibration::Calibrate calibrate(shape);
  CalibrationVisualizer vis(shape, capacity);
  CombineRgb rgb(shape);
  ShowRgb screen(shape);

  calibrate.fit_grid(camera.background(), camera.mask(), true);

  camera.out - image_splitter;
  image_splitter.out1 - image;
  image_splitter.out2 - enhance_points;
  enhance_points.out - points_splitter;
  points_splitter.out1 - points;
  points_splitter.out2 - peaks;
  peaks.out - tracker;

  calibrate.fingers_in - tracker;
  vis.fingers_in - calibrate;
  rgb.red_in - vis;
  rgb.green_in - points;
  rgb.blue_in - normalize;
  normalize.in - transpose;
  transpose.in - image;
  screen.in - rgb;

  run();
}

void run_track_fingers (Args & args)
{
  size_t finger_capacity = args.pop(TABLE_FINGER_CAPACITY);
  size_t width = args.pop(512);
  size_t height = args.pop(256);
  Rectangle shape(width, height);

  Silence silence;
  FingersTable table(finger_capacity);
  CalibrationVisualizer fingers(shape, finger_capacity);
  ShowMono screen(shape);

  table.sound_in - silence;
  fingers.fingers_in - table.fingers_out;
  screen.in - fingers;

  run();
}

void run_track_hands (Args & args)
{
  size_t hand_capacity = args.pop(TABLE_HAND_CAPACITY);
  size_t width = args.pop(512);
  size_t height = args.pop(256);
  Rectangle shape(width, height);

  Silence silence;
  HandsTable table(hand_capacity);
  CalibrationVisualizer hands(shape, hand_capacity);
  CalibrationVisualizer fingers(shape, hand_capacity * FINGERS_PER_HAND);
  CombineRgb rgb(shape);
  ShowRgb screen(shape);

  table.sound_in - silence;
  hands.fingers_in - table.hands_out;
  fingers.fingers_in - table.fingers_out;
  rgb.red_in - hands;
  rgb.green_in - fingers;
  screen.in - rgb;

  run();
}

void run_track_sitar (Args & args)
{
  size_t hand_capacity = args.pop(TABLE_HAND_CAPACITY);
  size_t width = args.pop(512);
  size_t height = args.pop(256);

  Rectangle shape(width, height);
  size_t capacity = hand_capacity * (1 + FINGERS_PER_HAND);

  Silence silence;
  HandsTable table(hand_capacity);
  Sitar sitar(hand_capacity);
  CalibrationVisualizer fingers(shape, hand_capacity * FINGERS_PER_HAND);
  CalibrationVisualizer hands(shape, hand_capacity);
  CalibrationVisualizer both(shape, capacity);
  CombineRgb rgb(shape);
  ShowRgb screen(shape);

  table.sound_in - silence;
  sitar.hands_in - table.hands_out;
  sitar.fingers_in - table.fingers_out;
  fingers.fingers_in - table.fingers_out;
  hands.fingers_in - table.hands_out;
  both.fingers_in - sitar;
  rgb.blue_in - hands;
  rgb.red_in - fingers;
  rgb.green_in - both;
  screen.in - rgb;

  run();
}

void run_track_save (Args & args)
{
  g_peaks_filename = args.pop(g_peaks_filename);
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);

  RegionThread camera;
  EnhancePoints points(camera.out);
  PeakDetector peaks(points.out, capacity);
  PeaksSequence file;

  camera.out - points;
  points.out - peaks;
  peaks.out - file;

  run();

  file.save(g_peaks_filename);
}

void run_track_sim (Args & args)
{
  size_t num_frames = args.pop(100);
  float speed = args.pop(125.0f);
  size_t capacity = args.pop(1);

  PeaksSequence file;

  Image::Peaks peaks;
  Seconds time = Seconds::now();
  float dt = 1.0f / DEFAULT_VIDEO_FRAMERATE;
  float radius = 0.5 * speed * num_frames * dt;
  float intensity = 0.8f;

  for (size_t i = 0; i < num_frames; ++i) {
    float t = i * dt;
    for (size_t j = 0; j < capacity; ++j) {
      Image::Peak peak(
            t * speed - radius,
            0,
            intensity);
      PRINT(peak);
      peaks.push_back(peak);
    }
    time += dt;

    file.push(time, peaks);
    peaks.clear();
  }

  file.save(g_peaks_filename);
}

void run_track_stats (Args & args)
{
  g_peaks_filename = args.pop(g_peaks_filename);
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);

  PeaksSequence file;
  file.load(g_peaks_filename);

  Tracking::Tracker tracker(capacity);

  file.out - tracker;

  file.playback();
}

void run_track_play (Args & args)
{
  g_peaks_filename = args.pop(g_peaks_filename);
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);
  float age_scale = args.pop(DEFAULT_TRACKER_VIS_AGE_SCALE);

  PeaksSequence file;
  file.load(g_peaks_filename);
  Image::Peak extent = file.extent();
  Rectangle shape = Rectangle(roundu(extent.x), roundu(extent.y));

  Tracking::Tracker tracker(capacity);
  TrackAgeVisualizer vis(shape, capacity, age_scale);
  ShowRgbZoom screen(shape, g_screen_shape);

  file.out - tracker;
  vis.tracks_in - tracker;
  screen.in - vis;

  run();
}

void run_track (Args & args)
{
  args
    .case_("table", run_track_table)
    .case_("bucket", run_track_thing<Bucket>)
    .case_("ball", run_track_thing<ShadowBall>)
    .case_("table-age", run_track_table_age)
    .case_("bucket-age", run_track_thing_age<Bucket>)
    .case_("ball-age", run_track_thing_age<ShadowBall>)
    .case_("calibrate", run_track_calibrate)
    .case_("fingers", run_track_fingers)
    .case_("hands", run_track_hands)
    .case_("sitar", run_track_sitar)
    .case_("save", run_track_save)
    .case_("sim", run_track_sim)
    .case_("stats", run_track_stats)
    .case_("play", run_track_play)
    .default_error();
}

//----( flock )---------------------------------------------------------------

void run_flock_pitch (Args & args)
{
  size_t size = args.pop(PSYCHO_HARMONY_SIZE);
  float acuity = args.pop(PSYCHO_PITCH_ACUITY);
  float min_freq = args.pop(MIN_CHROMATIC_FREQ_HZ);
  float max_freq = args.pop(MAX_CHROMATIC_FREQ_HZ);

  Rectangle shape = g_screen_shape.size() ? g_screen_shape : Rectangle(400,800);

  StereoAudioThread audio(not g_deaf);
  SpeakerGain speaker_gain;
  PitchFlockViewer flock(shape, size, acuity, min_freq, max_freq);
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

  Rectangle shape = g_screen_shape.size() ? g_screen_shape : Rectangle(400,800);

  StereoAudioThread audio(not g_deaf);
  SpeakerGain speaker_gain;
  TempoFlockViewer flock(shape, size, min_freq, max_freq);
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

void run_flock_pt (Args & args)
{
  size_t voice_count = args.pop(PSYCHO_POLYRHYTHM_COUNT);
  size_t pitch_size = args.pop(PSYCHO_HARMONY_SIZE);
  size_t tempo_size = args.pop(PSYCHO_RHYTHM_SIZE);
  float pitch_acuity = args.pop(PSYCHO_PITCH_ACUITY);
  float min_pitch_hz = args.pop(MIN_CHROMATIC_FREQ_HZ);
  float max_pitch_hz = args.pop(MAX_CHROMATIC_FREQ_HZ);
  float min_tempo_hz = args.pop(PSYCHO_MIN_TEMPO_HZ);
  float max_tempo_hz = args.pop(PSYCHO_MAX_TEMPO_HZ);

  Rectangle shape = g_screen_shape.size() ? g_screen_shape : Rectangle(640,640);

  StereoAudioThread audio(not g_deaf);
  SpeakerGain speaker_gain;
  FlockViewer flock(
      shape,
      voice_count,
      pitch_size,
      tempo_size,
      pitch_acuity,
      min_pitch_hz,
      max_pitch_hz,
      min_tempo_hz,
      max_tempo_hz);
  ShowRgb screen(flock);

  audio.in - speaker_gain;
  speaker_gain.in - flock;
  screen.in - flock;

  if (g_deaf) {

    run();

  } else {

    audio.out - flock;

    run();
  }
}

void run_flock (Args & args)
{
  args
    .case_("pitch", run_flock_pitch)
    .case_("tempo", run_flock_tempo)
    .case_("pt", run_flock_pt)
    .default_error();
}

//----( bucket playing commands )---------------------------------------------

void run_buckets (
    Args & args,
    Pushed<StereoAudioFrame> & mic,
    Pulled<StereoAudioFrame> & speaker)
{
  StereoAudioThread audio(not g_deaf);
  SpeakerGain gain;

  if (not g_deaf) {
    audio.out - mic;
  }

  gain.in - speaker;

  run_spectrum(args, audio.in, gain);
}

void run_bucket_formant (Args & args)
{
  float pitch_shift = args.pop(0.0f);

  FormantBucket bucket(pitch_shift);
  PowerMeter power;

  power.out - bucket.power_in;

  run_buckets(args, power, bucket);
}

void run_bucket_formant2 (Args & args)
{
  float pitch_shift1 = args.pop(0.0f);
  float pitch_shift2 = args.pop(0.0f);

  FormantBucket bucket1(pitch_shift1);
  FormantBucket bucket2(pitch_shift2);

  PowerSplitter power;
  AudioMixer mixer;

  mixer.in1 - bucket1;
  mixer.in2 - bucket2;

  power.out1 - bucket1.power_in;
  power.out2 - bucket2.power_in;

  run_buckets(args, power, mixer);
}

void run_bucket_loop (Args & args)
{
  const char * filename = args.pop();
  float duration_sec = args.pop(4.0f);
  float begin_sec = args.pop(0.0f);

  StereoAudioFrame loop = LoopSynth::load_loop(
      filename,
      duration_sec,
      begin_sec);

  LoopBucket bucket(loop);
  PowerMeter power;

  power.out - bucket.power_in;

  run_buckets(args, power, bucket);
}

void run_bucket_loop2 (Args & args)
{
  const char * filename1 = args.pop();
  const char * filename2 = args.pop();
  float duration1 = args.pop(4.0f);
  float duration2 = args.pop(4.0f);
  float begin1 = args.pop(0.0f);
  float begin2 = args.pop(0.0f);

  StereoAudioFrame loop1 = LoopSynth::load_loop(filename1, duration1, begin1);
  StereoAudioFrame loop2 = LoopSynth::load_loop(filename2, duration2, begin2);

  LoopBucket bucket1(loop1);
  LoopBucket bucket2(loop2);

  PowerSplitter power;
  AudioMixer mixer;

  mixer.in1 - bucket1;
  mixer.in2 - bucket2;

  power.out1 - bucket1.power_in;
  power.out2 - bucket2.power_in;

  run_buckets(args, power, mixer);
}

void run_bucket_formant_loop (Args & args)
{
  float duration = args.pop(4.0f);

  FormantLoopBuckets buckets(duration);

  PowerSplitter power;

  power.out1 - buckets.formant_power_in;
  power.out2 - buckets.loop_power_in;

  run_buckets(args, power, buckets);
}

template<class UI>
void run_ui_pipes (Args & args)
{
  size_t capacity = args.pop(BUCKET_FINGER_CAPACITY);
  float pitch_shift = args.pop(0.0f);

  CoupledBand<StereoAudioFrame> band;
  StereoAudioThread audio(false);

  UI ui(g_deaf, capacity);
  SyncoPipes member(capacity, pitch_shift);

  member.in - ui.fingers_out;
  band.add(member);

  if (g_showing) {

    size_t diameter = args.pop(512);
    Rectangle shape(diameter, diameter);

    Oscilloscope oscilloscope(shape);
    ShowMonoZoom screen(shape, g_screen_shape);

    member.phases_monitor - oscilloscope;
    screen.in - oscilloscope;
    audio.in - band;

    run();

  } else {

    run_spectrum(args, audio.in, band);
  }
}

template<class UI>
void run_ui_blobs (Args & args)
{
  size_t capacity = args.pop(BUCKET_FINGER_CAPACITY);

  CoupledBand<Vocoder::Timbre> band;
  Vocoder vocoder;
  StereoAudioThread audio(not g_deaf);
  PowerMeter power;

  UI ui(g_deaf, capacity);
  SyncoBlobs member(capacity);

  if (not g_deaf) {
    audio.out - power;
    power.out - ui.power_in;
  }

  member.in - ui.fingers_out;
  band.add(member);
  vocoder.in - band;

  run_spectrum(args, audio.in, vocoder);
}

template<class UI, class Member>
void run_ui_member (Args & args)
{
  size_t capacity = args.pop(BUCKET_FINGER_CAPACITY);

  CoupledBand<StereoAudioFrame> band;
  StereoAudioThread audio(false);

  UI ui(g_deaf, capacity);
  Member member(capacity);
  member.in - ui.fingers_out;
  band.add(member);

  run_spectrum(args, audio.in, band);
}

void run_bucket_band (Args & args)
{
  size_t num_members = args.pop(4);
  size_t capacity = args.pop(5);
  ASSERT_LE(1, num_members);
  ASSERT_LE(num_members, 4);

  CoupledBand<StereoAudioFrame> band;
  StereoAudioThread audio(false);
  SpeakerGain gain;
  audio.in - gain;

  typedef SyncoPipes    Member1;
  typedef SyncoStrings  Member2;
  typedef Wobbler       Member3;
  typedef CoupledHats   Member4;

  FingersBucket bucket1(g_deaf, capacity);
  Member1 member1(capacity);
  member1.in - bucket1.fingers_out;
  band.add(member1);

  if (num_members == 1) { run_spectrum(args, gain.in, band); return; }

  FingersBucket bucket2(g_deaf, capacity);
  Member2 member2(capacity);
  member2.in - bucket2.fingers_out;
  band.add(member2);

  if (num_members == 2) { run_spectrum(args, gain.in, band); return; }

  FingersBucket bucket3(g_deaf, capacity);
  Member3 member3(capacity);
  member3.in - bucket3.fingers_out;
  band.add(member3);

  if (num_members == 3) { run_spectrum(args, gain.in, band); return; }

  FingersBucket bucket4(g_deaf, capacity);
  Member4 member4(capacity);
  member4.in - bucket4.fingers_out;
  band.add(member4);

  if (num_members == 4) { run_spectrum(args, gain.in, band); return; }
}

void run_bucket (Args & args)
{
  args
    .case_("formant", run_bucket_formant)
    .case_("loop", run_bucket_loop)
    .case_("formant2", run_bucket_formant2)
    .case_("loop2", run_bucket_loop2)
    .case_("formant_loop", run_bucket_formant_loop)
    .case_("pipes", run_ui_pipes<FingersBucket>)
    .case_("blobs", run_ui_blobs<FingersBucket>)
    .case_("wobbler", run_ui_member<FingersBucket, Wobbler>)
    .case_("hats", run_ui_member<FingersBucket, CoupledHats>)
    .case_("strings", run_ui_member<FingersBucket, SyncoStrings>)
    .case_("band", run_bucket_band)
    .default_error();
}

//----( table playing commands )----------------------------------------------

template<class Voice>
void run_table_chorus (Args & args)
{
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);
  ASSERT_LT(0, capacity);

  if (capacity == 1) {
    FingerTable table(g_deaf);
    Soloist<Voice> soloist;

    soloist.in - table.finger_out;

    run_spectrum(args, table.sound_in, soloist);

  } else {

    FingersTable table(g_deaf, capacity);
    Chorus<Finger, Voice> chorus(capacity);

    chorus.in - table.fingers_out;
    table.sound_in - chorus;

    run_spectrum(args, table.sound_in, chorus);
  }
}

void run_table_vocosliders (Args & args)
{
  ShadowTable table(true);
  SliderColumns sliders(table.shadow_out, SYNTHESIS_VOCODER_SIZE);
  VocoSliders control;
  Vocoder vocoder;

  table.shadow_out - sliders;
  sliders.out - control;
  vocoder.in - control;

  run_vocoder(args, vocoder, table.sound_in);
}

template<class Voice>
void run_table_vocochorus (Args & args)
{
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);
  ASSERT_LT(0, capacity);

  if (capacity == 1) {
    FingerTable table(g_deaf);
    Soloist<Voice, Vocoder::Timbre> soloist;
    Vocoder vocoder;

    soloist.in - table.finger_out;
    vocoder.in - soloist;

    run_vocoder(args, vocoder, table.sound_in);

  } else {

    FingersTable table(g_deaf, capacity);
    Chorus<Finger, Voice, Vocoder::Timbre> chorus(capacity);
    Vocoder vocoder;

    chorus.in - table.fingers_out;
    vocoder.in - chorus;

    run_vocoder(args, vocoder, table.sound_in);
  }
}

template<class Instrument>
void run_table_fingers (Args & args)
{
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);

  FingersTable table(g_deaf, capacity);
  Instrument instrument(capacity);

  instrument.in - table.fingers_out;

  run_spectrum(args, table.sound_in, instrument);
}

template<class Instrument>
void run_table_hands (Args & args)
{
  size_t capacity = args.pop(TABLE_FINGER_CAPACITY);

  HandsTable table(g_deaf, capacity);
  Instrument instrument(capacity);

  instrument.fingers_in - table.fingers_out;
  instrument.hands_in - table.hands_out;

  run_spectrum(args, table.sound_in, instrument);
}

void run_table_dirt (Args & args)
{
  bool coalesce = args.pop(true);
  float blur_factor = args.pop(1.0f);

  if (g_showing) {

    ShowDirt system(g_screen_shape, g_deaf, coalesce, blur_factor);

    run();

  } else {

    Dirt system(g_deaf, coalesce, blur_factor);

    run();
  }
}

void run_table_dirt_pitch (Args & args)
{
  float freq0 = args.pop(4e2f);
  float freq1 = args.pop(4e3f);

  DirtPitchTest system(freq0, freq1, g_deaf, g_screen_shape);

  run();
}

void run_table_syncopoints (Args & args)
{
  size_t capacity = args.pop(BUCKET_FINGER_CAPACITY);

  FingersTable table(g_deaf, capacity);
  SyncoPoints member(capacity);
  CoupledBand<Vocoder::Timbre> band;
  Vocoder vocoder;

  member.in - table.fingers_out;
  band.add(member);
  vocoder.in - band;

  run_spectrum(args, table.sound_in, vocoder);
}

void run_table_sitar (Args & args)
{
  size_t hand_capacity = args.pop(TABLE_HAND_CAPACITY);

  HandsTable table(g_deaf, hand_capacity);
  Sitar sitar(hand_capacity);
  SitarStrings strings(hand_capacity * (1 + FINGERS_PER_HAND));

  sitar.fingers_in - table.fingers_out;
  sitar.hands_in - table.hands_out;
  strings.in - sitar;

  run_spectrum(args, table.sound_in, strings);
}

void run_table_sitar2 (Args & args)
{
  bool coalesce = args.pop(true);
  float blur_factor = args.pop(4.0f);
  size_t hand_capacity = args.pop(TABLE_HAND_CAPACITY);

  HandsTable table(g_deaf, hand_capacity);
  Beater beater(coalesce, blur_factor);
  Sitar sitar(hand_capacity);
  SitarStrings strings(hand_capacity * (1 + FINGERS_PER_HAND));

  sitar.fingers_in - table.fingers_out;
  sitar.hands_in - beater;
  beater.fingers_in - table.hands_out;
  beater.power_in - table.impact_out;
  strings.in - sitar;

  if (g_showing) {

    ShowMonoZoom screen(beater.shape, g_screen_shape);

    table.sound_in - strings;
    screen.in - beater;

    run();

  } else {

    run_spectrum(args, table.sound_in, strings);

  }
}

void run_table (Args & args)
{
  args
    .case_("glotta", run_table_chorus<Synthesis::Glottis>)
    .case_("buzzers", run_table_chorus<Synthesis::Buzzer>)
    .case_("bells", run_table_chorus<Synthesis::Bell>)
    .case_("sines", run_table_chorus<Synthesis::Sine>)
    .case_("esines", run_table_chorus<Synthesis::ExpSine>)
    .case_("pipes", run_table_chorus<Synthesis::Pipe>)
    .case_("vibes", run_table_chorus<Synthesis::Vibe>)
    .case_("gongs", run_table_chorus<Synthesis::Gong>)
    .case_("shepards", run_table_chorus<Synthesis::Shepard>)
    .case_("shepvibes", run_table_chorus<Synthesis::ShepardVibe>)
    .case_("strings", run_table_chorus<Synthesis::String>)
    .case_("plucked", run_table_chorus<Synthesis::Plucked>)
    .case_("vocosliders", run_table_vocosliders)
    .case_("vocopoints", run_table_vocochorus<Synthesis::VocoPoint>)
    .case_("vocoblobs", run_table_vocochorus<Synthesis::VocoBlob>)
    .case_("csines", run_table_fingers<CoupledSines>)
    .case_("sitarstrings", run_table_fingers<SitarStrings>)
    .case_("shepard4s", run_table_fingers<Shepard4s>)
    .case_("shepard7s", run_table_fingers<Shepard7s>)
    .case_("sitar", run_table_sitar)
    .case_("sitar2", run_table_sitar2)
    .case_("wideband", run_table_fingers<Wideband>)
    .case_("split1", run_table_fingers<Splitband1>)
    .case_("split2", run_table_fingers<Splitband2>)
    .case_("split3", run_table_fingers<Splitband3>)
    .case_("split4", run_table_fingers<Splitband4>)
    .case_("dirt", run_table_dirt)
    .case_("dirt_pitch", run_table_dirt_pitch)
    .case_("syncopoints", run_table_syncopoints)
    .default_error();
}

//----( ball playing commands )-----------------------------------------------

void run_ball (Args & args)
{
  args
    .case_("pipes", run_ui_pipes<FingersBall>)
    .case_("blobs", run_ui_blobs<FingersBall>)
    .case_("wobbler", run_ui_member<FingersBall, Wobbler>)
    .case_("hats", run_ui_member<FingersBall, CoupledHats>)
    .case_("strings", run_ui_member<FingersBall, SyncoStrings>)
    .default_error();
}

//----( singing commands )----------------------------------------------------

void run_sing (Args & args)
{
  float acuity = args.pop(Rational::HARMONY_ACUITY);
  float randomize_rate = args.pop(Rational::HARMONY_RANDOMIZE_RATE);

  StereoAudioThread audio(not g_deaf);
  SpeakerGain speaker_gain;
  RationalSinger singer(acuity, randomize_rate);

  audio.in - speaker_gain;
  speaker_gain.in - singer;

  if (g_deaf) {

    run();

  } else {

    MicGain mic_gain;

    audio.out - mic_gain;
    mic_gain.out - singer;

    run();
  }
}

} // namespace Streaming

using namespace Streaming;

//----( options )-------------------------------------------------------------

void option_brightness (Args & args)
{
  Camera::set_brightness(atoi(args.pop()));
  Camera::set_auto_white_balance(false);
}
void option_contrast (Args & args)
{
  Camera::set_contrast(atoi(args.pop()));
  Camera::set_auto_white_balance(false);
}
void option_gain (Args & args)
{
  Camera::set_gain(atoi(args.pop()));
  Camera::set_auto_gain(false);
}
void option_exposure (Args & args)
{
  Camera::set_exposure(atoi(args.pop()));
  Camera::set_auto_exposure(false);
}
void option_sharpness (Args & args)
{
  Camera::set_sharpness(atoi(args.pop()));
  Camera::set_auto_white_balance(false);
}

void option_config (Args & args)
{
  // TODO examine extension to determine what to configure
  Camera::set_config(args.pop());
}

void option_zoom (Args & args)
{
  ASSERT(args.size() > 2, "no zoom values specified");

  size_t width = atoi(args.pop());
  size_t height = atoi(args.pop());
  g_screen_shape = Rectangle(width, height);
}

void option_show (Args & args) { g_showing = true; }
void option_hear (Args & args) { g_hearing = true; }
void option_deaf (Args & args) { g_deaf = true; }
void option_spectrum (Args & args) { g_spectrum = true; }
void option_sweep (Args & args) { g_sweep = true; }
void option_power (Args & args) { g_power = true; }

//----( harness )-------------------------------------------------------------

const char * long_help_message =
"Usage: kazoo [OPTIONS] COMMAND_SEQUENCE [ARGS]"
"\nOptions:"
"\n  show | hear | deaf | spectrum | sweep | power"
"\n  brightness (0 | ... | 255, default = 0)"
"\n  contrast (0 | ... | 255, default = 32)"
"\n  gain (1 | ... | 63 | auto)"
"\n  exposure (1 | ... | 255 | auto, default = 120)"
"\n  sharpness (0 | ... | 63, default = 0)"
"\n  config FILENAME"
"\n  zoom MIN_WIDTH MIN_HEIGHT"
"\n  nogpu"
"\nCommand Sequences:"
"\n  sim"
"\n    reassign [TIMESCALE] [WIDTH] [HEIGHT] [FRAMERATE]"
"\n    attract [TIMESCALE] [WIDTH] [HEIGHT] [FRAMERATE]"
"\n    tongues"
"\n      phasor | syncopator | shepard4 | shepard7 "
               "| geometric | boltz | phasor2"
"\n        [ACUITY = 7.0] [MAX_STRENGTH = 2] [WIDTH] [HEIGHT]"
"\n    keys"
"\n      phasor | syncopator | shepard4 | shepard7 "
               "| geometric | boltz | phasor2"
"\n        [MIN_ACUITY = 1.0] [MAX_ACUITY = 12.0] [WIDTH] [HEIGHT]"
"\n    islands"
"\n      phasor | syncopator | shepard4 | shepard7 "
               "| geometric | boltz | phasor2"
"\n        [ACUITY = 7.0] [STRENGTH_SCALE = 1] [WIDTH] [HEIGHT]"
"\n    mouse [WIDTH] [HEIGHT]"
"\n  video SOURCE [FILTER [OPTIONS]]"
"\n    camera | region | crop | mask | sub | ceil | disk | fifth | change"
"\n      transpose"
"\n      square [RADIUS = 2]"
"\n      quadratic [RADIUS = 2]"
"\n      highpass [RADIUS = 2]"
"\n      points"
"\n      hands"
"\n      flow [HIGHPASS_RADIUS = 8]"
"\n      krig_flow [SPACESCALE] [HIGHPASS_RADIUS]"
"\n      gloves [SPACESCALE] [HIGHPASS_RADIUS]"
"\n      filter_flow [PROCESS_NOISE] [SPACESCALE] [HIGHPASS_RADIUS]"
"\n    mosaic [NUM_TILES = 1]"
"\n  audio"
"\n    wire | gain"
"\n    play FILENAME"
"\n    fourier | fourier2 | maskogram | psychogram"
"\n      [SIZE]"
"\n    vocoder"
"\n      chirp | chord | drone"
"\n        [NUM_TONES = 1] [PERIOD = 4.0] [DURATION = 512]"
"\n      band [RATIO = 1] [PERIOD = 4.0] [DURATION = 512]"
"\n    beater [COALESCE = true] [BLUR_FACTOR = 1.0]"
"\n    beat [RADIUS = 2.0] [WIDTH = 512]"
"\n    tempo [FILENAME]"
"\n  track"
"\n    table | bucket | ball"
"\n      [MAX_DETECTIONS = 10]"
"\n    table-age | bucket-age | ball-age"
"\n      [MAX_DETECTIONS = 10] [AGE_SCALE = 1]"
"\n    calibrate [MAX_DETECTIONS = 10]"
"\n    fingers | hands"
"\n      [CAPACITY] [WIDTH] [HEIGHT]"
"\n    save [OUTFILE = data/peaks.seq] [MAX_DETECTIONS = 10]"
"\n    sim [NUM_FRAMES = 100] [SPEED = 125] [NUM_DETECTIONS = 1]"
"\n    stats [INFILE = data/peaks.seq] [MAX_DETECTIONS = 10]"
"\n    play [INFILE = data/peaks.seq] [MAX_DETECTIONS = 10] [AGE_SCALE = 1]"
"\n  flock"
"\n    pitch [SIZE = 1024] [ACUITY = 7] [MIN_FREQ] [MAX_FREQ]"
"\n    tempo [SIZE] [MIN_FREQ] [MAX_FREQ]"
"\n    pt [VOICE_SIZE] [PITCH_SIZE] [TEMPO_SIZE] [PITCH_ACUITY] ..."
"\n  bucket"
"\n    formant [PITCH_SHIFT = 0]"
"\n    formant2 [PITCH_SHIFT1 = 0] [PITCH_SHIFT2 = 0]"
"\n    loop FILENAME.RAW [DURATION = 1 sec] [BEGIN (sec)]"
"\n    loop2 FILE1.RAW FILE2.RAW [DURATION1] [DURATION2] [BEGIN1] [BEGIN2]"
"\n    formant_loop [DURATION = 4 sec]"
"\n    pipes [FINGER_CAPACITY] [PITCH_SHIFT = 0]"
"\n    blobs | wobbler | hats | strings"
"\n      [FINGER_CAPACITY]"
"\n    band [NUM_MEMBERS = 4] [FINGER_CAPACITY]"
"\n  table"
"\n    glotta | buzzers | sines | esines | pipes | vibes | gongs"
"\n           | strings | plucked | shepards | shepvibes"
"\n           | csines | shepard4s | shepard7s | sitar | sitar2"
"\n           | wideband | split1 | split2 | split3 | split4"
"\n           | vocosliders | vocopoints | vocoblobs | syncopoints"
"\n      [FINGER_CAPACITY]"
"\n    dirt [COALESCE = true] [BLUR_FACTOR = 1.0]"
"\n    dirt_pitch [MIN_FREQ] [MAX_FREQ]"
"\n  ball"
"\n    pipes [FINGER_CAPACITY] [PITCH_SHIFT = 0]"
"\n    blobs | wobbler | hats | strings"
"\n      [FINGER_CAPACITY]"
"\n    band [NUM_MEMBERS = 4] [FINGER_CAPACITY]"
"\n  sing [ACUITY = 7] [RANDOMIZE_RATE = 10]"
;

const char * short_help_message =
"Usage: kazoo [OPTIONS] COMMAND_SEQUENCE [ARGS]"
"\n try 'kazoo help' for detailed usage"
;

void run_long_help (Args & args) { LOG(long_help_message); }
void run_short_help (Args & args) { LOG(short_help_message); }

int main (int argc, char ** argv)
{
  LOG(kazoo_logo);
  chdir_kazoo();

  Args args(argc, argv, short_help_message);

  args
    .case_("brightness", option_brightness)
    .case_("contrast", option_contrast)
    .case_("gain", option_gain)
    .case_("exposure", option_exposure)
    .case_("sharpness", option_sharpness)
    .case_("config", option_config)
    .case_("zoom", option_zoom)
    .case_("hear", option_hear)
    .case_("show", option_show)
    .case_("deaf", option_deaf)
    .case_("spectrum", option_spectrum)
    .case_("sweep", option_sweep)
    .case_("power", option_power)
    .default_break_else_repeat();

  args
    .case_("sim", run_sim)
    .case_("video", run_video)
    .case_("audio", run_audio)
    .case_("track", run_track)
    .case_("flock", run_flock)
    .case_("bucket", run_bucket)
    .case_("table", run_table)
    .case_("ball", run_ball)
    .case_("sing", run_sing)
    .case_("help", run_long_help)
    .default_error();

  return 0;
}

