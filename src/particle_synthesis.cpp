
#include "particle_synthesis.h"

namespace Synthesis
{

//----( particle synth )------------------------------------------------------

void NoisyPeakSynth::sample (Vector<complex> & sound_accum)
{
  m_particles.pf_to_fp();

  const size_t I = m_particles.num_particles;
  const size_t T = sound_accum.size;

  const float * restrict trans_real = m_particles.field(0);
  const float * restrict trans_imag = m_particles.field(1);
  const float * restrict noise_scale = m_particles.field(4);

  float * restrict state_real = m_particles.field(2);
  float * restrict state_imag = m_particles.field(3);
  complex * restrict sound = sound_accum;

  for (size_t t = 0; t < T; ++t) {

    const complex noise = random_normal_complex();
    const float noise_x = noise.real();
    const float noise_y = noise.imag();

    float sum_x = 0;
    float sum_y = 0;

    for (size_t i = 0; i < I; ++i) {

      float s = noise_scale[i];

      float x = state_real[i];
      float y = state_imag[i];

      float f = trans_real[i];
      float g = trans_imag[i];

      sum_x += state_real[i] = f * x - g * y + s * noise_x;
      sum_y += state_imag[i] = f * y + g * x + s * noise_y;
    }

    sound[t] += complex(sum_x, sum_y);
  }

  m_particles.fp_to_pf();
}

void NoisyKuramotoSynth::sample (Vector<complex> & sound_accum)
{
  m_particles.pf_to_fp();

  const size_t I = m_particles.num_particles;
  const size_t T = sound_accum.size;

  const float * restrict damplitude = m_particles.field(2);
  const float * restrict amplitude1 = m_particles.field(3);
  const float * restrict frequency = m_particles.field(4);
  const float * restrict bandwidth = m_particles.field(5);

  float * restrict phase_real = m_particles.field(0);
  float * restrict phase_imag = m_particles.field(1);
  complex * restrict sound = sound_accum;

  // initialize force
  float sum_m = 0;
  float sum_mx = 0;
  float sum_my = 0;

  for (size_t i = 0; i < I; ++i) {
    float b = bandwidth[i];
    float s = max(0.0f, 1 - b);
    float a = amplitude1[i] + max(0.0f, -damplitude[i]);
    float m = a * s;

    sum_m += m;
    sum_mx += m * phase_real[i];
    sum_my += m * phase_imag[i];
  }

  float force_scale = 1 / max(1e-8f, sum_m);

  for (size_t t = 0; t < T; ++t) {

    float dt = static_cast<float>(t) / T - 1;

    const float noise = m_noise_scale * random_std();
    const float force_x = force_scale * sum_mx;
    const float force_y = force_scale * sum_my;
    sum_mx = 0;
    sum_my = 0;

    float sum_ax = 0;
    float sum_ay = 0;

    for (size_t i = 0; i < I; ++i) {

      float x = phase_real[i];
      float y = phase_imag[i];

      float b = bandwidth[i];
      float s = max(0.0f, 1 - b);

      // update state
      {
        float f = frequency[i];
        float force = force_y * x - force_x * y;
        float bent = f * (1 + s * force + b * noise);

        float dx = -bent * y;
        float dy = bent * x;

        x += dx;
        y += dy;
      }

      // normalize
      {
        float r = sqrt(sqr(x) + sqr(y)); // here norm(x,y) >= 1

        phase_real[i] = x /= r;
        phase_imag[i] = y /= r;
      }

      // update force
      {
        float a = amplitude1[i] + damplitude[i] * dt;
        sum_ax += x;
        sum_ay += y;

        float m = a * s;
        sum_mx += m * x;
        sum_my += m * y;
      }
    }

    sound[t] += complex(sum_ax, sum_ay);
  }

  m_particles.fp_to_pf();
}

//----( syncopated )----------------------------------------------------------

Synchronized::ScaledBeatFun SyncoParticle::s_param(COUPLED_SYNCO_ACUITY);

SyncoParticleSynth::SyncoParticleSynth (size_t num_particles)
  : m_synth(num_particles),
    m_synco_particles(num_particles),
    m_mutices()
{
  for (size_t i = 0; i < num_particles; ++i) {
    m_synco_particles[i].init_data(m_synth.particle(i));
  }
}

void SyncoParticleSynth::pull (Seconds time, StereoAudioFrame & sound_accum)
{
  const size_t I = m_synco_particles.size();

  for (size_t i = 0; i < m_mutices.size(); ++i) m_mutices[i]->lock();

  Synchronized::Poll poll;
  for (size_t i = 0; i < I; ++i) {
    poll += m_synco_particles[i].poll();
  }

  complex force = poll.mean();
  for (size_t i = 0; i < I; ++i) {
    m_synco_particles[i].sample(force);
  }

  m_synth.sample(sound_accum);

  for (size_t i = 0; i < m_mutices.size(); ++i) m_mutices[i]->unlock();
}

//----( chorus )--------------------------------------------------------------

void SyncoParticleChorus::push (
    Seconds time,
    const BoundedMap<Id, Finger> & fingers)
{
  TODO("update existing fingers");
  TODO("sort remaining voices");
  TODO("put new fingers in weakest voices");
}

} // namespace Synthesis

