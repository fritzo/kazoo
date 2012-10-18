#ifndef KAZOO_CALIBRATION_H
#define KAZOO_CALIBRATION_H

/** Camera calibration

  Coordinates:
    image coordinates are orthogonal in [-1,1],
    so typically x:[-1,1], and y:[-0.7,0.7].

  TODO switch to multi-matching for edge-weighting
  TODO when centering, weigh vertices based on edge fitness
*/

#include "common.h"
#include "streaming.h"
#include "vectors.h"
#include "bounded.h"
#include "image_types.h"
#include "gestures.h"
#include "config.h"

#define CALIBRATE_BORDER_PADDING        (0)
#define CALIBRATE_FILTER_RADIUS         (2)
#define CALIBRATE_MIN_INTENSITY         (0.01f)
#define CALIBRATE_MAX_DETECTIONS        (128)

#define CALIBRATE_EM_STEPS        (3)
#define CALIBRATE_NLS_STEPS       (3)
#define CALIBRATE_BP_STEPS        (0)

namespace Calibration
{

using Image::Point;
using Image::Points;
using Image::Peak;
using Image::Peaks;
using Gestures::Finger;

//----( distortion )----------------------------------------------------------

enum { brown_size = 4 };
inline void brown_transform (
    const float param[brown_size],
    Point & point)
{
  const float & u0 = param[0];
  const float & v0 = param[1];
  const float & k2 = param[2];
  const float & k4 = param[3];

  float u = point.x - u0;
  float v = point.y - v0;

  float r2 = sqr(u) + sqr(v);
  float s = 1 + r2 * (k2 + r2 * k4);

  point.x = u0 + s * u;
  point.y = v0 + s * v;
}

enum { affine_size = 3 };
inline void affine_transform (
    const float param[affine_size],
    Point & point)
{
  const float & theta  = param[0]; // disallow skew
  const float & p1     = param[1];
  const float & t1     = param[2];

  // approximate cosine & sine for small angles
  float c = 1 - sqr(theta) / 2;
  float s = theta;

  float rot_x = c * point.x + s * point.y;
  float rot_y = c * point.y - s * point.x;

  point.x = expf(p1) * rot_x;
  point.y = expf(t1) * rot_y;
}

enum { distortion_size = brown_size + affine_size };
inline void distortion_transform (
    const float param[distortion_size],
    Point & point)
{
  const float * brown_param = param;;
  const float * affine_param = param + brown_size;

  brown_transform(brown_param, point);
  affine_transform(affine_param, point);
}

inline void distortion_transform (
    const float param[distortion_size],
    const Point & cam,
    Point & grid)
{
  grid = cam;
  distortion_transform(param, grid);
}

extern float g_distortion_mean[distortion_size];
extern float g_distortion_sigma[distortion_size];

//----( calibrator )----------------------------------------------------------

class Calibrate
  : public Rectangle,
    public Image::Transform,
    public Streaming::Pulled<Finger>,
    public Streaming::Pulled<BoundedMap<Id, Finger> >
{
  ConfigParser m_config;

  Image::PixToRadial m_pix_to_radial;
  Vector<float> m_distortion_param;
  Point m_offset;

  const bool m_debug;
  const size_t m_em_steps;
  const size_t m_nls_steps;
  const size_t m_bp_steps;

  const float m_flip_x;
  const float m_flip_y;

public:

  Streaming::Port<Streaming::Pulled<Finger> > finger_in;
  Streaming::Port<Streaming::Pulled<BoundedMap<Id, Finger> > > fingers_in;

  Calibrate (
      Rectangle shape,
      bool debug = false,
      const char * config_filename = "config/default.calibrate.conf");

  virtual ~Calibrate () {}

  void operator= (const Calibrate & other)
  {
    m_pix_to_radial = other.m_pix_to_radial;
    m_distortion_param = other.m_distortion_param;
    m_offset = other.m_offset;
  }

  virtual void operator() (Point & point) const
  {
    m_pix_to_radial(point);

    //ASSERTW_LE(-1, point.x);
    //ASSERTW_LE(point.x, 1);
    //ASSERTW_LE(-1, point.y);
    //ASSERTW_LE(point.y, 1);

    distortion_transform(m_distortion_param, point);
    point.x = (point.x + m_offset.x) * m_flip_x;
    point.y = (point.y + m_offset.y) * m_flip_y;
  }

  virtual void pull (Seconds time, Finger & finger);
  virtual void pull (Seconds time, BoundedMap<Id, Finger> & fingers);

  void fit_grid (
      const float * restrict background,
      const float * restrict mask = NULL,
      bool transpose        = false,
      size_t radius         = CALIBRATE_FILTER_RADIUS,
      size_t max_detections = CALIBRATE_MAX_DETECTIONS,
      float min_intensity    = CALIBRATE_MIN_INTENSITY);

  void scale_input (float factor) { m_pix_to_radial.scale_input(factor); }
};

//----( grid fitting )--------------------------------------------------------

class Grid : public VectorFunction
{
public:

  struct Edge
  {
    Id head;
    Id tail;
    float weight;
    bool vertical;

    Edge () {}
    Edge (size_t h, size_t t, float w, bool v)
      : head(h), tail(t), weight(w), vertical(v) {}
  };

private:

  Vector<float> m_param_mean;
  Vector<float> m_param_sigma;
  Point m_offset;

  const float m_width_pix;
  const float m_height_pix;
  const float m_radius_pix;

  std::vector<Peak> m_verts;
  std::vector<Edge> m_edges;

  std::vector<Point> m_points;

public:

  Grid (
      float width_pix,
      float height_pix);
  virtual ~Grid () {}

  //----( problem )----

  std::vector<Peak> & verts () { return m_verts; }
  const std::vector<Peak> & verts () const { return m_verts; }

  //----( solution )----

  void fit (size_t em_steps, size_t nls_steps, size_t bp_steps);
  const Vector<float> & distortion_param () const { return m_param_mean; }
  const Point & offset () const { return m_offset; }

  const std::vector<Point> & points () const { return m_points; }
  const std::vector<Edge> & edges () const { return m_edges; }

  void save_soln();

  //----( vector function interface )----

  virtual size_t size_in () const { return distortion_size; }
  virtual size_t size_out () const { return 2 * m_edges.size(); }
  virtual void operator() (const Vector<float> & input, Vector<float> & output);

private:

  inline float vertex_likelihood (float z);
  inline float vertex_fa_cost (float z);
  inline float vertex_md_cost (float distance_to_border);
  inline float edge_likelihood (float dx, float dy);
  inline float edge_cost (float dx, float dy);
  inline float edge_accuracy (Id head, Id tail);

protected:

  void update_points (const float * param = NULL);
  void update_edges_naive ();
  void update_edges_bp (size_t num_iters = 1);

  void clear ()
  {
    m_edges.clear();
    m_points.clear();
  }
};

} // namespace Calibration

inline ostream & operator<< (ostream & o, const Calibration::Grid::Edge & e)
{
  return o << e.head << ' ' << e.tail << ' ' << e.weight << ' ' << e.vertical;
}
inline istream & operator>> (istream & i, Calibration::Grid::Edge & e)
{
  return i >> e.head >> e.tail >> e.weight >> e.vertical;
}

inline ostream & operator<< (
    ostream & o,
    const std::vector<Calibration::Grid::Edge> & edges)
{
  o << edges.size() << '\n';
  for (size_t e = 0; e < edges.size(); ++e) {
    o << edges[e] << '\n';
  }
  return o;
}
inline istream & operator>> (
    istream & i,
    std::vector<Calibration::Grid::Edge> & edges)
{
  size_t size;
  i >> size;
  edges.resize(size);
  for (size_t e = 0; e < size; ++e) {
    i >> edges[e];
  }
  return i;
}

//----( visualization )-------------------------------------------------------

namespace Streaming
{

class CalibrationVisualizer
  : public Rectangle,
    public Pulled<MonoImage>
{
  const float m_decay;
  bool m_initialized;

  BoundedMap<Id, Gestures::Finger> m_fingers;

public:

  SizedPort<Pulled<BoundedMap<Id, Gestures::Finger> > > fingers_in;

  CalibrationVisualizer (Rectangle shape, size_t finger_capacity);
  virtual ~CalibrationVisualizer () {}

  virtual void pull (Seconds time, MonoImage & image);
};

} // namespace Streaming

#endif // KAZOO_CALIBRATION_H
