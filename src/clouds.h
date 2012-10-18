#ifndef KAZOO_CLOUDS_H
#define KAZOO_CLOUDS_H

#include "common.h"
#include "cloud_math.h"
#include "cloud_points.h"
#include "eigen_gpu.h"
#include <vector>

namespace Cloud
{

static const float DEFAULT_TRACK_TIMESCALE_SEC = 1.0f;
static const float DEFAULT_OBSERVATION_WEIGHT = 1.0f;

//----( clouds )--------------------------------------------------------------

class Cloud : public Persistent
{
  std::vector<PointSet *> m_points;
  std::vector<JointPrior *> m_flow;

public:

  Cloud (size_t dim, size_t bits, Rectangle shape);
  Cloud (size_t dim, size_t min_bits, size_t max_bits, Rectangle shape);
  Cloud (string filename);
  Cloud (istream & file);
  virtual ~Cloud ();

  size_t num_grids () const { return m_points.size(); }

  PointSet & points (size_t i)
  {
    ASSERT_LT(i, num_grids());
    return * m_points[i];
  }
  JointPrior & flow (size_t i)
  {
    ASSERT_LT(i, num_grids());
    return * m_flow[i];
  }
  const PointSet & points (size_t i) const
  {
    ASSERT_LT(i, num_grids());
    return * m_points[i];
  }
  const JointPrior & flow (size_t i) const
  {
    ASSERT_LT(i, num_grids());
    return * m_flow[i];
  }

  PointSet & points () { return * m_points.back(); }
  JointPrior & flow () { return * m_flow.back(); }
  const PointSet & points () const { return * m_points.back(); }
  const JointPrior & flow () const { return * m_flow.back(); }

  void init_points (VideoSequence & seq, size_t fit_passes = 32);
  void grow_points (VideoSequence & seq, size_t fit_passes = 32);
  void fit_points (VideoSequence & seq, size_t fit_passes = 32);
  void init_flow (VideoSequence & seq, float tol);

  void save_priors (string filename, int transition_order = 4);
  void save_histograms (
      string filename,
      const VideoSequence & seq,
      float bins_per_unit = 50);

  // return this
  Cloud * crop_below (size_t min_size);
  Cloud * crop_above (size_t max_size);

  virtual void write (ostream & o) const;
  virtual void read (istream & file);
};

//----( controllers )---------------------------------------------------------

class Controller : public Persistent
{
  ConfigParser & m_config;

  const bool m_coalesce_track;
  const float m_track_timescale;
  const float m_observation_weight;

  Cloud * m_dom;
  Cloud * m_cod;
  JointPrior * m_map;

  MatrixSf & m_observe;
  MatrixSf & m_advance;

  Gpu::SparseMultiplier * m_observe_gpu;
  Gpu::SparseMultiplier * m_advance_gpu;

  VectorXf & m_dom_prior;
  VectorXf & m_cod_observation;
  VectorXf & m_dom_observation;
  VectorXf & m_dom_prediction;
  VectorXf & m_dom_state;

public:

  Controller (string dom_filename, string cod_filename);
  Controller (string filename);
  virtual ~Controller ();

  bool is_coalescing () const { return m_coalesce_track; }
  bool is_tracking () const { return m_track_timescale > 0; }

  PointSet & dom () { return m_dom->points(); }
  PointSet & cod () { return m_cod->points(); }
  PointSet & dom (size_t i) { return m_dom->points(i); }
  PointSet & cod (size_t i) { return m_cod->points(i); }
  JointPrior & dom_flow () { return m_dom->flow(); }
  JointPrior & cod_flow () { return m_cod->flow(); }
  JointPrior & dom_flow (size_t i) { return m_dom->flow(i); }
  JointPrior & cod_flow (size_t i) { return m_cod->flow(i); }
  JointPrior & map () { return * m_map; }

  void optimize (float tol, size_t max_iters = 100);

  void set_track (const Point & dom_point);
  void update_track (const Point & cod_point, Point & dom_point);

  void crop ();
  void sort ();

  virtual void write (ostream & o) const;
  virtual void read (istream & file);

private:

  void init_transforms ();
};

} // namespace Cloud

#endif // KAZOO_CLOUDS_H
