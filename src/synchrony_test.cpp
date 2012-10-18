
#include "synchrony.h"
#include "reassign.h"
#include "animate.h"
#include "events.h"
#include "args.h"
#include <cstdlib>
#include <cstdio>

#define MIN_FREQ                        (20.0f / DEFAULT_SAMPLE_RATE)
#define MAX_FREQ                        (7e3f / DEFAULT_SAMPLE_RATE)

using namespace Synchronized;

//----( testing tools )-------------------------------------------------------

void wagging_finger (
    float time,
    Vector<float> & mass,
    float width0 = 0.01,
    float width1 = 0.04,
    float mod_amp = 0.1)
{
  for (size_t i = 0, I = mass.size; i < I; ++i) {
    float pos = ((i + 0.5f) / I) * 2 - 1; // in (-1,1)
    float mean = mod_amp * sin(4 * M_PI * time);
    float width = width0 * pow(width1 / width0, time);

    mass[i] = exp(-sqr((pos - mean) / width) / 2);
  }
}

void crossing_fingers (
    float time,
    Vector<float> & mass,
    float distance = 0.0,
    float offset = 2.0,
    float width = 0.01,
    float octaves = 1.0)
{
  float total_octaves = log(MAX_FREQ / MIN_FREQ) / log(2);
  float mod_amp = octaves / total_octaves;
  float sep = distance / total_octaves;
  float mean = offset / total_octaves;

  for (size_t i = 0, I = mass.size; i < I; ++i) {
    float pos = ((i + 0.5f) / I) * 2 - 1; // in (-1,1)
    float mean1 = +time * mod_amp + sep + mean;
    float mean2 = -time * mod_amp - sep + mean;

    mass[i] = exp(-sqr((pos - mean1) / width) / 2)
            + exp(-sqr((pos - mean2) / width) / 2);
  }
}

//----( fourier bank )--------------------------------------------------------

template<class FB>
void run_time_fourier (Args & args)
{
  size_t size = args.pop(320);
  float freq0 = args.pop(1e-3f);
  float freq1 = args.pop(1e-1f);

  const size_t duration = DEFAULT_SAMPLE_RATE * 60;
  const size_t block_size = DEFAULT_FRAMES_PER_BUFFER;

  FB bank(Bank(size, freq0, freq1));

  // make some example signal
  Vector<float> time_in(block_size);
  for (size_t t = 0; t < block_size; ++t) {
    time_in[t] = sin(2.0 * M_PI * t / block_size);
  }

  Vector<float> freq_out(size);

  LOG("sampling " << size << " oscillators"
      " in frames of size " << block_size);

  Timer timer;
  for (size_t block = 0; block * block_size < duration; ++block) {
    bank.sample(time_in, freq_out);
  }

  float rate = duration / timer.elapsed();
  LOG("averaged " << rate << " samples/sec = "
      << (rate / DEFAULT_SAMPLE_RATE) << " x sample rate");
}

//----( phasor bank )---------------------------------------------------------

template<class PB>
void run_time_bank_ (Args & args)
{
  size_t size = args.pop(320);
  float freq0 = args.pop(1e-3f);
  float freq1 = args.pop(1e-1f);
  float acuity = args.pop(7.0f);
  float strength = args.pop(DEFAULT_SYNCHRONY_STRENGTH);

  const size_t duration = DEFAULT_SAMPLE_RATE * 60;
  const size_t block_size = DEFAULT_FRAMES_PER_BUFFER;

  PB bank(Bank(size, freq0, freq1, acuity, strength));

  Vector<float> mass(size);
  Vector<float> dmass(size);
  Vector<complex> sound_accum(block_size);

  for (size_t i = 0; i < size; ++i) {
    mass[i] = expf(random_std());
  }

  LOG("sampling " << size << " oscillators"
      " in frames of size " << block_size);

  Timer timer;
  for (size_t block = 0; block * block_size < duration; ++block) {
    sound_accum.zero();
    subtract(mass, bank.get_mass(), dmass);
    bank.sample_accum(dmass, sound_accum);
    bank.retune();
  }

  float rate = duration / timer.elapsed();
  LOG("averaged " << rate << " samples/sec = "
      << (rate / DEFAULT_SAMPLE_RATE) << " x sample rate");
}

//----( boltzmann set )-------------------------------------------------------

template<class S>
void run_time_set_ (Args & args)
{
  size_t size = args.pop(10000);

  const float rate = DEFAULT_AUDIO_FRAMERATE;
  const size_t duration = 60 * rate;

  S set(size, 0.25f / rate, 8 / rate);

  LOG("sampling set of " << size << " oscillators");

  Timer timer;
  for (size_t t = 0; t < duration; ++t) {

    float value = random_std();

    set.advance();
    set.learn(value);
    set.predict_value();
  }

  float max_rate = duration / timer.elapsed();
  LOG("averaged " << max_rate << " samples/sec = "
      << (max_rate / rate) << " x sample rate");
}

//----( vector loop bank )----------------------------------------------------

void run_time_vloop (Args & args)
{
  size_t size = args.pop(48);
  float freq0 = args.pop(0.25f / DEFAULT_VIDEO_FRAMERATE);
  float freq1 = args.pop(8.0f / DEFAULT_VIDEO_FRAMERATE);
  float acuity = args.pop(3.0f);
  float strength = args.pop(DEFAULT_SYNCHRONY_STRENGTH);
  size_t width = args.pop(256);

  const size_t height = size;
  const size_t duration = DEFAULT_VIDEO_FRAMERATE * 60;

  VectorLoopBank bank(Bank(size, freq0, freq1, acuity, strength), width);

  Vector<float> mass_yx(height * width);
  Vector<float> dmass_yx(height * width);
  Vector<float> amplitude_x(width);

  for (size_t i = 0; i < size; ++i) {
    dmass_yx[i] = expf(random_std());
  }

  LOG("sampling " << height << " loops of width " << width);

  float dt = 1.0f / DEFAULT_VIDEO_FRAMERATE;
  Timer timer;
  for (size_t t = 0; t < duration; ++t) {
    bank.decay_add_mass(dmass_yx, dt);
    bank.synchronize(dt);
    bank.sample(amplitude_x, dt);
  }

  float rate = duration / timer.elapsed();
  LOG("averaged " << rate << " samples/sec = "
      << (rate / DEFAULT_VIDEO_FRAMERATE) << " x video framerate");
}

//----( phasogram )-----------------------------------------------------------

void run_show_phasogram (Args & args)
{
  string analyzer = args.pop("super");
  string gesture = args.pop("wag");
  float acuity = args.pop(7.0f);
  float strength = args.pop(DEFAULT_SYNCHRONY_STRENGTH);
  size_t size = args.pop(320);

  float freq0 = MIN_FREQ;
  float freq1 = MAX_FREQ;
  size_t duration = 3 * DEFAULT_SAMPLE_RATE;
  size_t fft_exponent = 9;

  ASSERT_DIVIDES(4, size);

  bool use_super = (analyzer == "super");
  ASSERT((analyzer == "super") or (analyzer == "pitch"),
      "unknown analyzer: " << analyzer);

  Supergram super(fft_exponent, 1,1);
  FourierBank pitch(Bank(super.super_size(), freq0, freq1));

  size_t frame_size = super.small_size();
  size_t num_frames = duration / frame_size;

  Phasogram bank(frame_size, Bank(size, freq0, freq1, acuity, strength));

  size_t width = num_frames;
  size_t height = super.super_size();

  std::ostringstream title;
  title
    << "PhasorBank: "
    << "acuity = " << acuity
    << ", "
    << "strength = " << strength
    << ", "
    << "size = " << size
    ;

  Screen screen(Rectangle(width, height), title.str().c_str());

  Vector<float> mass(size);
  Vector<float> amplitude(size);
  Vector<complex> sound(frame_size);
  Vector<float> spectrum(height);

  for (size_t t = 0, T = num_frames; t < T; ++t) {

    float time = (t + 0.5f) / T;
    if (gesture == "wag")   wagging_finger(time, mass); else
    if (gesture == "cross") crossing_fingers(time, mass); else
    if (gesture == "distant") crossing_fingers(time, mass, 3); else
    ERROR("unknown gesture");

    amplitude = mass;

    float flatten = 1.0;
    amplitude += (1 - flatten);
    amplitude += flatten;

    amplitude *= 1.0f / (max(amplitude) * size); // to appease supergram

    bank.transform(mass, amplitude, sound);
    if (use_super) {
      super.transform_fwd(sound, spectrum);
    } else {
      pitch.sample(sound, spectrum);
    }

    // TODO use autogain here
    spectrum /= max(spectrum);
    screen.vertical_sweep(spectrum);
  }

  wait_for_keypress();
}

//----( commands )------------------------------------------------------------

void run_time (Args & args)
{
  args
    .case_("fourier", run_time_fourier<FourierBank>)
    .case_("fourier2", run_time_fourier<FourierBank2>)
    .case_("phasor", run_time_bank_<PhasorBank>)
    .case_("geom", run_time_bank_<GeomBank>)
    .case_("geom-set", run_time_set_<GeomSet>)
    .case_("boltz", run_time_bank_<BoltzBank>)
    .case_("vloop", run_time_vloop)
    .default_error();
}

void run_show (Args & args)
{
  args
    .case_("phasogram", run_show_phasogram)
    .default_error();
}

//----( main )----------------------------------------------------------------

const char * help_message =
"Usage: synchrony_test COMMAND [ARGS]"
"\nCommands:"
"\n  time"
"\n    fourier [size] [freq0] [freq1]"
"\n    fourier2 [size] [freq0] [freq1]"
"\n    phasor [size] [freq0] [freq1] [acuity] [strength]"
"\n    geom [size] [freq0] [freq1] [acuity] [strength]"
"\n    geom-set [size]"
"\n    boltz [size] [freq0] [freq1] [acuity] [strength]"
"\n    vloop [size] [freq0] [freq1] [acuity] [strength] [width]"
"\n  show"
"\n    phasogram [pitch|super] [wag|cross|distant] [acuity] [strength]"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);
  args
    .case_("time", run_time)
    .case_("show", run_show)
    .default_error();

  return 0;
}

