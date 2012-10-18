
#ifndef KAZOO_AUDIO_ALIGNMENT_H
#define KAZOO_AUDIO_ALIGNMENT_H

#include "common.h"
#include "streaming.h"

class ConfigParser;

namespace AudioAlignment
{

inline float prob_to_energy (float prob)
{
  ASSERT_LT(0, prob);
  ASSERT_LT(prob, 1);
  float entropy = -prob * logf(prob) - (1-prob) * logf(1-prob);
  return -logf(prob) - entropy;
}

//----( alignment model )----------------------------------------------------

class AlignmentModel
{
  ConfigParser & m_config;

public:

  const size_t feature_size;
  const float feature_sigma;
  const float feature_dof;
  const float segment_timescale;
  const float log_pos_sigma;
  const float log_vel_sigma;

  AlignmentModel (
      size_t size,
      const char * config_filename = "config/default.alignment.conf");
  ~AlignmentModel ();

  // TODO weight freq bins by 1 / decay timescale, to account for correlation
  //   (timescale can be found in RasterAudio::SpectrumParam or Voice::...)
  inline float feature_distance (
      const uint8_t * restrict x,
      const uint8_t * restrict y) const
  {
    const size_t I = feature_size;

    float sum = 0;

    for (size_t i = 0; i < I; ++i) {
      float xi = x[i];
      float yi = y[i];

      sum += sqr(xi - yi);
    }

    return sum / (2 * sqr(feature_sigma));
  }
  inline float feature_divergence (
      const uint8_t * restrict x,
      const uint8_t * restrict y) const
  {
    const size_t I = feature_size;

    float sum = 0;

    for (size_t i = 0; i < I; ++i) {
      float xi = x[i] + 0.5f;
      float yi = y[i] + 0.5f;

      //sum += (yi - xi) + xi * logf(xi / yi);
      //sum += sqr(max(0.0f, xi - yi));
      sum += xi / (xi + yi) * sqr(xi - yi);
    }

    return sum / (2 * sqr(feature_sigma));
  }

  // TODO rederive these exactly, using eg erf for continue-vs-break
  float get_break_prob () const { return 1 / (1 + segment_timescale); }
  float get_continue_prob () const
  {
    return (1 - get_break_prob()) * (1 - expf(-1 / log_pos_sigma));
  }
  float get_skip_prob () const
  {
    return (1 - get_break_prob()) * 0.5f * expf(-1 / log_pos_sigma);
  }

  float get_break_energy () const { return prob_to_energy(get_break_prob()); }
  float get_continue_energy () const
  {
    return prob_to_energy(get_continue_prob());
  }
  float get_skip_energy () const { return prob_to_energy(get_skip_prob()); }

private:

  AlignmentModel (const AlignmentModel &); // intentionally undefined
  void operator= (const AlignmentModel &); // intentionally undefined
};

//----( feature buffer )-----------------------------------------------------

class FeatureBuffer
  : public Streaming::Pushed<Vector<uint8_t> >
{
protected:

  std::vector<Seconds> m_times;
  std::vector<Vector<uint8_t> *> m_frames;

  Vector<uint8_t> * m_image;

  const size_t m_feature_size;
  const size_t m_max_duration;

public:

  FeatureBuffer (const AlignmentModel & model, size_t max_duration = 0);
  virtual ~FeatureBuffer ();

  virtual void push (Seconds time, const Vector<uint8_t> & features);
  void finish ();

  size_t duration () const { return m_times.size(); }
  size_t feature_size () const { return m_feature_size; }
  Seconds time (size_t i) const { return m_times[i]; }
  const Vector<uint8_t> & image () { finish(); return * m_image; }
};

//----( cost matrix )---------------------------------------------------------

class CostMatrix
{
  const size_t m_size;
  Vector<float> m_cost;
  float m_mean_cost;

public:

  CostMatrix (
      const AlignmentModel & model,
      FeatureBuffer & buffer,
      bool symmetric);

  size_t size () const { return m_size; }
  Vector<float> & cost () { return m_cost; }
  const Vector<float> & cost () const { return m_cost; }
  float mean_cost () const { return m_mean_cost; }
};

//----( alignment matrix )----------------------------------------------------

class AlignmentMatrix
{
  const AlignmentModel & m_model;
  const size_t m_size;
  Vector<float> m_posterior;

public:

  AlignmentMatrix (const AlignmentModel & model, size_t size);

  // these destroy the cost matrix
  void init_marginal (CostMatrix & cost_matrix);
  void init_maxlike (CostMatrix & cost_matrix);

  size_t size () const { return m_size; }
  Vector<float> & posterior () { return m_posterior; }
  const Vector<float> & posterior () const { return m_posterior; }
};

//----( alignment path )------------------------------------------------------

class AlignmentPath
{
  const size_t m_size;
  const AlignmentModel & m_model;

  const Vector<float> m_cost;
  Vector<float> m_post;

  float m_pressure;
  float m_dpressure;

  // TODO add topological data

public:

  AlignmentPath (
      const AlignmentModel & model,
      const CostMatrix & cost);

  size_t size () const { return m_size; }
  float pressure () const { return m_pressure; }
  Vector<float> & posterior () { return m_post; }
  const Vector<float> & posterior () const { return m_post; }

  void advance (float pressure_reltol = 0.1f);

private:

  void propagate ();
};

} // namespace AudioAlignment

#endif // KAZOO_AUDIO_ALIGNMENT_H
