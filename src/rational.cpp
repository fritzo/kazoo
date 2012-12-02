
#include "rational.h"
#include <Eigen/Eigen>

#define TOL     (1e-20f)

namespace Rational
{

int gcd (int a, int b)
{
  ASSERT(a >= 0, "gcd arg 1 is not positive: " << a);
  ASSERT(b >= 0, "gcd arg 2 is not positive: " << b);
  ASSERT(a % 1 == 0, "gcd arg 1 is not an integer: " << a);
  ASSERT(b % 1 == 0, "gcd arg 2 is not an integer: " << b);

  if (b > a) std::swap(a, b);
  if (b == 0) return 1; // gcd(0, anything) = 0

  while (true) {
    a %= b;
    if (a == 0) return b;
    b %= a;
    if (b == 0) return a;
  }
}

std::vector<Number> ball_of_radius (float radius)
{
  std::vector<Number> result;
  for (int i = 1; i < radius; ++i) {
    for (int j = 1; i * i + j * j <= radius * radius; ++j) {
      if (gcd(i, j) == 1) {
        result.push_back(Number(i, j));
      }
    }
  }
  std::sort(result.begin(), result.end());
  return result;
}

//----( harmony )-------------------------------------------------------------

Harmony::Harmony (
    float max_radius,
    float prior_sec, // TODO what is this used for
    float acuity,
    float sustain_sec,
    float attack_sec,
    float background_gain,
    float update_hz,
    float randomize_rate)
  : m_attack(attack_sec / DEFAULT_AUDIO_FRAMERATE),
    m_sustain(sustain_sec / DEFAULT_AUDIO_FRAMERATE),
    m_randomize_rate(randomize_rate / DEFAULT_AUDIO_FRAMERATE),
    m_points(ball_of_radius(max_radius)),
    m_energy_matrix(* new MatrixXf(m_points.size(), m_points.size())),
    m_mass_vector(* new VectorXf(m_points.size())),
    m_prior_vector(* new VectorXf(m_points.size())),
    m_mass(m_points.size(), m_mass_vector.data()),
    m_prior(m_points.size(), m_prior_vector.data()),
    m_analysis(m_points.size()),
    m_dmass(m_points.size()),
    m_anal(m_points, acuity),
    m_synth(m_points)
{
  size_t size = m_points.size();

  for (size_t i = 0; i < size; ++i) {
    for (size_t j = 0; j < size; ++j) {
      m_energy_matrix(i, j) = m_points[i].dissonance(m_points[j]) / acuity;
    }
  }

  ASSERT(size % 2, "Harmony does not have an odd number of points");
  m_mass.zero();
  m_mass[(size - 1) / 2] = 1; // start with all mass at center
  m_dmass.zero();
}

Harmony::~Harmony ()
{
  delete & m_energy_matrix;
  delete & m_mass_vector;
  delete & m_prior_vector;
}

static const float BOGUS_MIN_FREQ = 0.1f;
static const float BOGUS_MAX_FREQ = 10.0f;

Harmony::Analyzer::Analyzer (
    const std::vector<Number> & points,
    float acuity)
  : Synchronized::FourierBank2(Bank(
        points.size(),
        BOGUS_MIN_FREQ,
        BOGUS_MAX_FREQ,
        acuity))
{
  // adapted from Synchronized::Bank::init_decay_transform

  ASSERT_EQ(points.size(), size);
  const float omega0 = 2 * M_PI * MIDDLE_C_HZ / DEFAULT_SAMPLE_RATE;
  const float order = 2;
  const float damp_factor = pow(2, 1.0 / order) - 1;
  const float dpitch = log(2) / acuity;
  const float min_timescale = DEFAULT_FRAMES_PER_BUFFER;

  for (size_t i = 0, I = size; i < I; ++i) {
    double freq = omega0 * points[i].to_float();
    double dfreq = 1 / (1 / fabs(dpitch * freq) + min_timescale);
    std::complex<double> omega(-damp_factor * dfreq, freq);
    std::complex<double> trans = exp(omega);

    m_trans_real[i] = trans.real();
    m_trans_imag[i] = trans.imag();
    m_rescale[i] = pow(dfreq, order); // = 1 / E(w,0)
  }
}

Harmony::Synthesizer::Synthesizer (const std::vector<Number> & points)
  : Synchronized::SimpleBank(Bank(
        points.size(),
        BOGUS_MIN_FREQ,
        BOGUS_MAX_FREQ))
{
  // adapted from Synchronized::Bank::init_transform

  ASSERT_EQ(points.size(), size);
  float omega0 = 2 * M_PI * MIDDLE_C_HZ / DEFAULT_SAMPLE_RATE;
  for (size_t i = 0; i < size; ++i) {
    float omega = omega0 * points[i].to_float();

    m_frequency[i] = tan(omega);
  }
}

void Harmony::compute_prior ()
{
  const size_t F = m_points.size();

  float radius_scale = 1.0f / sum(m_mass);
  m_prior_vector.noalias() = m_energy_matrix * m_mass_vector;
  m_prior_vector *= radius_scale;

  float prior_total = 0;
  for (size_t i = 0; i < F; ++i) {
    prior_total += m_prior[i] = expf(-m_prior[i]);
  }
  float prior_scale = prior_total > 0 ? 1 / prior_total : 0.0f;
  for (size_t i = 0; i < F; ++i) {
    m_prior[i] *= prior_scale * m_prior[i];
  }
}

void Harmony::sample (
    Vector<complex> & sound_accum)
{
  const size_t F = m_points.size();

  compute_prior();

  for (size_t i = 0; i < F; ++i) {
    m_dmass[i] = (1.0f - m_sustain) * (m_prior[i] - m_mass[i])
               + m_randomize_rate * random_std() * m_mass[i];
  }

  m_synth.sample_accum(m_mass, m_dmass, sound_accum);
  m_mass += m_dmass;
}

void Harmony::sample (
    const Vector<complex> & sound_in,
    Vector<complex> & sound_accum)
{
  ASSERT_EQ(sound_in.size, sound_accum.size);
  const size_t F = m_points.size();

  m_anal.sample(sound_in, m_analysis);
  compute_prior();

  for (size_t i = 0; i < F; ++i) {
    m_dmass[i] = m_attack * m_analysis[i]
               + (1.0f - m_sustain) * (m_prior[i] - m_mass[i])
               + m_randomize_rate * random_std() * m_mass[i];
  }

  m_synth.sample_accum(m_mass, m_dmass, sound_accum);
  m_mass += m_dmass;
}

} // namespace Rational

