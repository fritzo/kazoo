#ifndef KAZOO_RATIONAL_H
#define KAZOO_RATIONAL_H

#include "common.h"
#include "vectors.h"
#include "synchrony.h"
#include "eigen.h"
#include <vector>

namespace Rational
{

static const float HARMONY_MAX_RADIUS = sqrtf(24*24 + 1*1 + 1e-4f); // 279 keys
static const float HARMONY_PRIOR_SEC = 8.0f;
static const float HARMONY_ACUITY = 7.0f;
static const float HARMONY_SUSTAIN_SEC = 1.0f;
static const float HARMONY_ATTACK_SEC = 0.1f;
static const float HARMONY_RANDOMIZE_RATE = 20.0f;

int gcd (int a, int b);

inline int lcm (int a, int b)
{
  return a * b / gcd(a, b);
}

struct Number
{
  int numer;
  int denom;

  Number () { ERROR("default-constructed a Rational::Number"); }
  Number (int n, int d)
  {
    ASSERT(0 <= n && n % 1 == 0, "invalid numer: " << n);
    ASSERT(0 <= d && d % 1 == 0, "invalid denom: " << d);
    ASSERT(n or d, "0/0 is not a rational number");

    int g = gcd(n, d);
    numer = n / g;
    denom = d / g;
  }

  float to_float () const
  {
    return static_cast<float>(numer) / static_cast<float>(denom);
  }

  bool operator< (const Number & other) const
  {
    return numer * other.denom < other.numer * denom;
  }

  bool operator== (const Number & other) const
  {
    return numer * other.denom == other.numer * denom;
  }

  bool operator!= (const Number & other) const
  {
    return numer * other.denom != other.numer * denom;
  }

  float norm () const
  {
    return sqrtf(numer * numer + denom * denom);
  }

  Number inv () const
  {
    ASSERT_NE(numer, 0);
    return Number(denom, numer);
  }

  Number operator* (const Number & other) const
  {
    return Number(numer * other.numer, denom * other.denom);
  }

  Number operator/ (const Number & other) const
  {
    return Number(numer * other.denom, denom * other.numer);
  }

  Number operator+ (const Number & other) const
  {
    return Number(
        numer * other.denom + other.numer * denom,
        denom * other.denom);
  }

  Number operator- (const Number & other) const
  {
    return Number(
        numer * other.denom - other.numer * denom,
        denom * other.denom);
  }

  float distance (const Number & other) const
  {
    return fabsf(to_float() - other.to_float());
  }

  float dissonance (const Number & other) const
  {
    return operator/(other).norm();
  }
};

inline std::ostream & operator<< (ostream & os, const Number & number)
{
  return os << number.numer << "/" << number.denom;
}

static const Number ZERO(0, 1);
static const Number INF(1, 0);
static const Number ONE(1, 1);

// ball does not include ZERO, ONE, or INF
std::vector<Number> ball_of_radius (float radius);

//----( harmony )-------------------------------------------------------------

class Harmony
{
  class Analyzer : public Synchronized::FourierBank2
  {
  public:
    Analyzer (const std::vector<Number> & points, float acuity);
  };

  class Synthesizer : public Synchronized::SimpleBank
  {
  public:
    Synthesizer (const std::vector<Number> & points);
  };

  const float m_attack;
  const float m_sustain;
  const float m_randomize_rate;
  std::vector<Number> m_points;

  MatrixXf & m_energy_matrix;
  VectorXf & m_mass_vector;
  VectorXf & m_prior_vector;
  Vector<float> m_mass; // aliased
  Vector<float> m_prior; // aliased
  Vector<float> m_analysis;
  Vector<float> m_dmass;

  Analyzer m_anal;
  Synthesizer m_synth;

public:

  Harmony (
      float max_radius = HARMONY_MAX_RADIUS,
      float prior_sec = HARMONY_PRIOR_SEC,
      float acuity = HARMONY_ACUITY,
      float sustain_sec = HARMONY_SUSTAIN_SEC,
      float attack_sec = HARMONY_ATTACK_SEC,
      float randomize_rate = HARMONY_RANDOMIZE_RATE);
  ~Harmony ();

  void analyze (const Vector<complex> & sound_in);
  void sample (Vector<complex> & sound_accum);

private:

  void compute_prior ();
};

} // namespace Rational

#endif // KAZOO_RATIONAL_H
