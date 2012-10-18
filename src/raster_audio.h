
#ifndef KAZOO_RASTER_AUDIO_H
#define KAZOO_RASTER_AUDIO_H

#include "common.h"
#include "vectors.h"
#include "streaming.h"

namespace RasterAudio
{

static const float SPECTRUM_MIN_FREQ_HZ = MIN_CHROMATIC_FREQ_HZ;
static const float SPECTRUM_MAX_FREQ_HZ = MAX_AUDIBLE_FREQ_HZ;
static const float SPECTRUM_MAX_TIMESCALE_SEC = 1.0f / LOW_FREQ_FM_JND_HZ;

//----( spectrum param )------------------------------------------------------

/** A psychoacoustically motivated non-uniformly spaced spectrum.

  Let w be frequency,
      T be the maximum timescale,
      p be the density of oscillators, and
      i be an index parameter (as in for (i...))

  We want to space oscillators according to

    dw = ( w/p + 1/T ) di

  so that pitch acuity is lower at low frequencies.
  Integrating, we find

             i + const                                              i
    w = exp[ --------- ] - p/T    =: A exp[i/p] - p/T    =: A exp[ --- ] - W
                 p                                                 T W

  for some constants A,W.
  Now given I bins, we let i range in [0,I], and w range in [w0,wI].
  To solve for A,W, we combine

                                 i
    w0 = A - W      wI = A exp[ --- ] - W
                                T W

  to eliminate A = w0 + W.
  We then solve for W as a fixed point of the iteration

                I
    W = ---------------
               w0 + W
        T log( ------ )
               wI + W

  The SpectrumParam class uses nomenclature:

                     1
    m_pitch_scale = ---     m_freq_scale = A     m_freq_shift = -W
                    T W
*/

class SpectrumParam
{
public:

  const size_t size;
  const float min_freq_hz;
  const float max_freq_hz;
  const float max_timescale_sec;

private:

  float m_pitch_scale;
  float m_freq_scale;
  float m_freq_shift;

public:

  SpectrumParam (size_t s, float f0, float f1, float t = INFINITY);

  float get_omega_at (size_t i) const
  {
    return m_freq_scale * exp(m_pitch_scale * (i + 0.5f)) + m_freq_shift;
  }

  float get_domega_at (size_t i) const
  {
    return m_freq_scale * m_pitch_scale * exp(m_pitch_scale * (i + 0.5f));
  }
};

//----( pitch analyzer )------------------------------------------------------

// transforms signal(time) to energy(pitch)

// TODO allow finer spectrum to compensate phase noise in chirps

class PitchAnalyzer
  : public Streaming::Pulled<Vector<float> >,
    public Streaming::Pushed<MonoAudioFrame>
{
  const SpectrumParam m_param;

  Vector<float> m_rescale;
  Vector<float> m_trans_real;
  Vector<float> m_trans_imag;
  Vector<float> m_stage1_real;
  Vector<float> m_stage1_imag;
  Vector<float> m_stage2_real;
  Vector<float> m_stage2_imag;

  MonoAudioFrame m_signal_in;
  Vector<float> m_energy_out;

public:

  bool debug;

  Streaming::SizedPort<Streaming::Pulled<MonoAudioFrame> > signal_in;
  Streaming::SizedPort<Streaming::Pushed<Vector<float> > > energy_out;

  PitchAnalyzer (const SpectrumParam & param);
  virtual ~PitchAnalyzer ();

  virtual void pull (Seconds time, Vector<float> & energy);
  virtual void push (Seconds time, const MonoAudioFrame & signal);

  void transform (const MonoAudioFrame & signal, Vector<float> & energy);
};

//----( pitch reassigner )----------------------------------------------------

// TODO this is memory-bounded; switch to a windowed version once this works.

class PitchReassigner
  : public Streaming::Pushed<Vector<float> >,
    public Streaming::Pulled<Vector<float> >,
    public Streaming::Thread
{
  const SpectrumParam m_param;

  bool m_frames_have_been_processed;

  std::vector<Seconds> m_times;
  std::vector<Vector<float> *> m_frames;

  Vector<float> * m_image;

  size_t m_frame_pos;

public:

  Streaming::SizedPort<Streaming::Pushed<Vector<float> > > out;

  PitchReassigner (const SpectrumParam & param);
  virtual ~PitchReassigner ();

  size_t size () const { return m_frames.size(); }

  // Step 1: first push() all frames in batch
  virtual void push (Seconds time, const Vector<float> & raw_in);

  // Step 2: process reassignment
  void process ();

  // Step 3: then either pull() or run()
  virtual void pull (Seconds time, Vector<float> & reassigned_out);
  virtual void run ();

  virtual void step () { ERROR("PitchReassigner::step should not be called"); }
};

//----( pitch synthesizer )---------------------------------------------------

class PitchSynthesizer
  : public Streaming::Pulled<StereoAudioFrame>
{
  const SpectrumParam m_param;

  Vector<float> m_trans_real;
  Vector<float> m_trans_imag;
  Vector<float> m_phase_real;
  Vector<float> m_phase_imag;
  Vector<float> m_amplitude1;
  Vector<float> m_damplitude;

public:

  bool debug;

  Streaming::SizedPort<Streaming::Pulled<Vector<float> > > amplitude_in;

  PitchSynthesizer (const SpectrumParam & param);
  virtual ~PitchSynthesizer ();

  virtual void pull (Seconds time, StereoAudioFrame & signal);

  void transform (
      const Vector<float> amplitude_in,
      StereoAudioFrame & signal_out);

private:

  void transform (StereoAudioFrame & signal_out);
};

} // namespace RasterAudio

#endif // KAZOO_RASTER_AUDIO_H
