#ifndef KAZOO_CLOUD_POINTS_H
#define KAZOO_CLOUD_POINTS_H

#include "common.h"
#include "cloud_math.h"
#include "cloud_persistence.h"

namespace Gpu { bool using_cuda(); }
namespace Streaming { class VideoSequence; }

// nvcc doesn't like hash_map, so only forward declare
class Histogram;
class ConfigParser;

namespace Cloud
{

class PointSequence;

//----( video formats )-------------------------------------------------------

using Streaming::VideoSequence;

enum VideoFormat { YUV_SINGLE, MONO_SINGLE, MONO_BATCH };

//----( point sets )----------------------------------------------------------

class PointSet : public Persistent
{
protected:

  ConfigParser & m_config;

public:

  enum { max_batch_size = 64 };

  const size_t dim;
  const size_t size;
  const Rectangle shape;

private:

  const bool m_fit_points_to_recon;
  const bool m_fit_radius_to_recon;
  const float m_target_radius;
  const float m_target_dof;
  const float m_target_entropy;

  const size_t m_batch_size;
  const float m_fit_rate_tol;
  const float m_construct_deriv_tol;
  const float m_construct_tol;
  const float m_purturb_factor;

  float m_radius;
  float m_fit_rate;

protected:

  Vector<float> m_prior;

  size_t m_count_stats_accum;
  QuantizeStats m_quantize_stats_accum;
  ConstructStats m_construct_stats_accum;

  mutable Vector<float> m_temp_squared_distances;
  mutable Vector<float> m_temp_recon;

  mutable size_t m_construct_one_stats_total;
  mutable size_t m_construct_one_stats_count;

public:

  static PointSet * create (size_t dim, size_t size, Rectangle shape);
  static PointSet * create (istream & file);

  virtual ~PointSet ();

  virtual void write (ostream & o) const;
  virtual void read (istream & file) { ERROR("use PointSet::create instead"); }

  //----( construction )----

  void init_from_sequence (const VideoSequence & seq);
  void init_from_smaller (const PointSet & smaller);

  void fit_sequence (VideoSequence & seq, size_t num_passes);

  Histogram * get_histogram (
      const VideoSequence & seq,
      float bins_per_unit = 10,
      float tol = 1e-4f,
      size_t max_bins = 10000);

  size_t get_size () const { return size; }

  virtual void get_point (size_t p, Point & point) const = 0;
  virtual void set_point (size_t p, const Point & point) = 0;

  float get_radius () const { return m_radius; }
  void set_radius (float radius);

  const Vector<float> & get_prior () const { return m_prior; }
  void get_prior (VectorXf & prior) const;
  void set_prior (const Vector<float> & prior) { m_prior = prior; }

  virtual void quantize (const Point & point, Vector<float> & likes) const = 0;
  virtual void quantize_batch (
      const Point & points,
      Vector<float> & likes) const = 0;
  void quantize (const Point & point, VectorXf & likes) const;

  virtual void construct (const Vector<float> & likes, Point & point) const = 0;
  void construct (const VectorXf & likes, Point & point) const;

protected:

  bool fitting_points_to_recon () const { return m_fit_points_to_recon; }
  bool fitting_radius_to_recon () const { return m_fit_radius_to_recon; }
  float get_fit_rate_tol () const { return m_fit_rate_tol; }
  float get_construct_deriv_tol () const { return m_construct_deriv_tol; }
  float get_construct_tol () const { return m_construct_tol; }

  float get_fit_rate () const { return m_fit_rate; }
  void set_fit_rate (float rate);

  void init_radius ();
  float fit_radius ();
  void fit_radius (
      const VideoSequence & seq,
      float tol = 1e-2f,
      size_t max_iters = 40);

  void init_stats_accum ();
  void update_stats_accum ();

  virtual void update_fit_rates () = 0;
  virtual void init_prior_accum () = 0;
  virtual void update_prior_accum () = 0;

  virtual void measure (const Point & point) const = 0;
  virtual void measure (size_t p) const = 0;

  virtual void accum_stats (const Point & probes, size_t num_probes) = 0;
  virtual void fit_points (const Point & probes, size_t num_probes) = 0;

  virtual void purturb_points (size_t group_size, float purturb_factor) = 0;

  void load_sequence (const VideoSequence & seq);

  PointSet (
      size_t dim,
      size_t size,
      Rectangle shape,
      const char * config_filename = "config/default.clouds.conf");

private:

  static PointSet * create_cpu (size_t dim, size_t size, Rectangle shape);
  static PointSet * create_gpu (size_t dim, size_t size, Rectangle shape);
};

inline PointSet * PointSet::create (size_t dim, size_t size, Rectangle shape)
{
  if (Gpu::using_cuda()) return create_gpu(dim, size, shape);
  else                   return create_cpu(dim, size, shape);
}

//----( joint priors )--------------------------------------------------------

class JointPrior : public Persistent
{
public:

  const PointSet & dom;
  const PointSet & cod;
  MatrixSf & joint;

protected:

  VectorXf & m_dom_scale;
  VectorXf & m_cod_scale;
  VectorXf & m_dom_temp;
  VectorXf & m_cod_temp;

public:

  JointPrior (const PointSet & points); // initialize to zero
  JointPrior (const PointSet & dom, const PointSet & cod); // initialize to zero
  virtual ~JointPrior ();

  bool empty () const;
  void clear ();

  void update_priors ();
  void fit_sequence (VideoSequence & seq, float tol);
  void init_spline (float tol);

  void get_push_forward (MatrixSf & transform);
  void get_pull_back (MatrixSf & transform);

  void push_forward (const VectorXf & dom_likes, VectorXf & cod_likes) const;
  void push_back (const VectorXf & cod_likes, VectorXf & dom_likes) const;
  void pull_forward (const VectorXf & dom_deriv, VectorXf & cod_deriv) const;
  void pull_back (const VectorXf & cod_deriv, VectorXf & dom_deriv) const;

  virtual void write (ostream & o) const;
  virtual void read (istream & file);
};

} // namespace Cloud

#endif // KAZOO_CLOUD_POINTS_H
