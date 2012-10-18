
#ifndef KAZOO_PARTICLE_SYNTHESIS_H
#define KAZOO_PARTICLE_SYNTHESIS_H

#include "common.h"
#include "synthesis.h"
#include "streaming.h"
#include "particles.h"

namespace Synthesis
{

//----( noisy peaks )---------------------------------------------------------

struct NoisyPeak
{
  float * restrict data;

  float & trans_real  () { return data[0]; }
  float & trans_imag  () { return data[1]; }
  float & state_real  () { return data[2]; }
  float & state_imag  () { return data[3]; }
  float & noise_scale () { return data[4]; }

  float trans_real  () const { return data[0]; }
  float trans_imag  () const { return data[1]; }
  float state_real  () const { return data[2]; }
  float state_imag  () const { return data[3]; }
  float noise_scale () const { return data[4]; }

  NoisyPeak () : data(NULL) {}
  NoisyPeak (float * d)
    : data(d)
  {
    trans_real() = 0.5f;
    trans_imag() = 0.5f;
    state_real() = 1;
    state_imag() = 0;
    noise_scale() = 0.0f;
  }

  void set (float amp, float freq, float bandwidth_cb)
  {
    float decay = exp(-bandwidth_cb * freq);
    trans_real() = decay * cos(2 * M_PI * freq);
    trans_imag() = decay * sin(2 * M_PI * freq);
    noise_scale() = amp * (1 - decay);
  }

  float energy () const { return sqr(state_real()) + sqr(state_imag()); }
};

class NoisyPeakSynth
{
  Particle::Matrix m_particles;

public:

  typedef NoisyPeak Particle;

  NoisyPeakSynth (size_t size)
    : m_particles(size, 8)
  {}

  float * particle (size_t i) { return m_particles.particle(i); }

  void sample (Vector<complex> & sound_accum);
};

//----( kuramoto )------------------------------------------------------------

struct NoisyKuramoto
{
  float * restrict data;

  float & phase_real () { return data[0]; }
  float & phase_imag () { return data[1]; }
  float & damplitude () { return data[2]; }
  float & amplitude1 () { return data[3]; }
  float & frequency  () { return data[4]; }
  float & bandwidth  () { return data[5]; }

  float phase_real () const { return data[0]; }
  float phase_imag () const { return data[1]; }
  float damplitude () const { return data[2]; }
  float amplitude1 () const { return data[3]; }
  float frequency  () const { return data[4]; }
  float bandwidth  () const { return data[5]; }

  NoisyKuramoto () : data(NULL) {}
  NoisyKuramoto (float * d)
    : data(d)
  {
    phase_real() = 1;
    phase_imag() = 0;
    damplitude() = 0;
    amplitude1() = 0;
    frequency() = 0.1f;
    bandwidth() = 0.0f;
  }

  void set (float amp, float freq, float bandwidth_cb)
  {
    damplitude() = amp - amplitude1();
    amplitude1() = amp;
    frequency() = tanf(2 * M_PI * freq);
    bandwidth() = bandwidth_cb;
  }

  float energy () const { return sqr(amplitude1()); }
};

class NoisyKuramotoSynth
{
  Particle::Matrix m_particles;
  const float m_noise_scale;

public:

  typedef NoisyKuramoto Particle;

  NoisyKuramotoSynth (size_t size)
    : m_particles(size, 8),
      m_noise_scale(1.0f)
  {}

  float * particle (size_t i) { return m_particles.particle(i); }

  void sample (Vector<complex> & sound_accum);
};

//----( syncopated )----------------------------------------------------------

class SyncoParticle
{
  typedef NoisyPeak Particle;

protected:

  static Synchronized::ScaledBeatFun s_param;

  Particle m_particle;
  Synchronized::Syncopator m_tempo;
  float m_mass;
  float m_sharpness;

public:

  typedef StereoAudioFrame Sound;

  SyncoParticle () : m_tempo(Coupled::g_synco_param) {}

  void init_data (float * data) { m_particle.data = data; }

  float energy () const { return m_particle.energy(); }

  complex as_complex () const { return m_mass * m_tempo.phase; }

  void set_timbre (const Finger & polar_finger)
  {
    TODO("set slow timbre");
  }

  Synchronized::Poll poll () const { return m_tempo.poll(); }

  void sample (complex force)
  {
    m_tempo.mass = m_mass;
    complex phase = m_tempo.sample(force);
    //float offbeat = 1 - phase.real();

    TODO("set fast timbre");
  }
};

class SyncoParticleSynth : public Streaming::Pulled<StereoAudioFrame>
{
  NoisyPeakSynth m_synth;
  std::vector<SyncoParticle> m_synco_particles;
  std::vector<Mutex *> m_mutices;

public:

  SyncoParticleSynth (size_t num_particles);

  SyncoParticle & particle (size_t i)
  {
    ASSERT_LE(i, m_synco_particles.size());
    return m_synco_particles[i];
  }
  void add_mutex (Mutex & mutex) { m_mutices.push_back(& mutex); }

  virtual void pull (Seconds time, StereoAudioFrame & sound_accum);
};

//----( chorus )--------------------------------------------------------------

class SyncoParticleChorus
  : public Streaming::Pushed<BoundedMap<Id, Finger> >
{
  Mutex m_mutex;

public:

  SyncoParticleChorus (
      size_t num_particles,
      SyncoParticleSynth & synth)
  {
    synth.add_mutex(m_mutex);
    TODO("grab particles from synth");
  }

  virtual void push (Seconds time, const BoundedMap<Id, Finger> & fingers);

protected:

  virtual void layout (const Finger & finger, SyncoParticle & particle) = 0;
};

} // namespace Streaming

#endif // KAZOO_PARTICLE_SYNTHESIS_H

