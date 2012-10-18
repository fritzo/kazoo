
#include "calibration.h"
#include "images.h"
#include "probability.h"
#include "regression.h"
#include "linalg.h"
#include "propagate.h"
#include "splines.h"
#include <limits.h>
#include <algorithm>
#include <fstream>

namespace Calibration
{

//----( distortion )----------------------------------------------------------

float g_distortion_mean[] = {

  0,    // u0 : rad
  0,    // v0 : rad
  0.2f, // k2 : rad^-2
  0.2f, // k4 : rad^-4

  0,    // theta : rad
  1.0f, // p1 : 1
  2.0f  // t1 : 1
};

float g_distortion_sigma[] = {

  0.3f, // u0 : rad
  0.3f, // v0 : rad
  0.1f, // k2 : rad^-2
  0.1f, // k4 : rad^-4

  0.1f, // theta : rad
  0.3f, // p1 : 1
  0.3f  // t1 : 1
};

//----( calibrator )----------------------------------------------------------

Calibrate::Calibrate (
    Rectangle shape,
    bool debug,
    const char * config_filename)

  : Rectangle(shape),

    m_config(config_filename),

    m_pix_to_radial(shape),
    m_distortion_param(distortion_size),
    m_offset(0,0),

    m_debug(debug),
    m_em_steps(m_config("em_steps", 3)),
    m_nls_steps(m_config("nls_steps", 3)),
    m_bp_steps(m_config("bp_steps", 0)),

    m_flip_x(m_config("flip_x", 0) ? 1 : -1),
    m_flip_y(m_config("flip_y", 0) ? 1 : -1),

    finger_in("Calibrate.finger_in"),
    fingers_in("Calibrate.fingers_in")
{
  copy_float(g_distortion_mean, m_distortion_param, distortion_size);
}

void Calibrate::pull (Seconds time, Finger & finger)
{
  finger_in.pull(time, finger);

  const float dt = 1.0f / DEFAULT_VIDEO_FRAMERATE;

  Point pos, vel;
  finger.get_pos(pos);
  finger.get_vel(vel);

  Point prev, next;
  prev.x = pos.x - vel.x * dt / 2;
  prev.y = pos.y - vel.y * dt / 2;
  next.x = pos.x + vel.x * dt / 2;
  next.y = pos.y + vel.y * dt / 2;

  operator()(prev);
  operator()(next);

  pos.x = bound_to(-GRID_SIZE_X, GRID_SIZE_X, next.x + prev.x) / 2;
  pos.y = bound_to(-GRID_SIZE_Y, GRID_SIZE_Y, next.y + prev.y) / 2;
  vel.x = (next.x - prev.x) / dt;
  vel.y = (next.y - prev.y) / dt;

  finger.set_pos(pos);
  finger.set_vel(vel);
}

void Calibrate::pull (Seconds time, BoundedMap<Id, Finger> & fingers)
{
  fingers_in.pull(time, fingers);

  const float dt = 1.0f / DEFAULT_VIDEO_FRAMERATE;

  for (size_t i = 0; i < fingers.size; ++i) {
    Finger & finger = fingers.values[i];

    Point pos, vel;
    finger.get_pos(pos);
    finger.get_vel(vel);

    Point prev, next;
    prev.x = pos.x - vel.x * dt / 2;
    prev.y = pos.y - vel.y * dt / 2;
    next.x = pos.x + vel.x * dt / 2;
    next.y = pos.y + vel.y * dt / 2;

    operator()(prev);
    operator()(next);

    pos.x = bound_to(-GRID_SIZE_X, GRID_SIZE_X, next.x + prev.x) / 2;
    pos.y = bound_to(-GRID_SIZE_Y, GRID_SIZE_Y, next.y + prev.y) / 2;
    vel.x = (next.x - prev.x) / dt;
    vel.y = (next.y - prev.y) / dt;

    finger.set_pos(pos);
    finger.set_vel(vel);
  }
}

void Calibrate::fit_grid (
    const float * restrict background_data,
    const float * restrict mask_data,
    bool transpose,
    size_t R,
    size_t max_detections,
    float min_intensity)
{
  size_t I = width();
  size_t J = height();

  if (transpose) std::swap(I,J);

  Vector<float> image(size());
  Vector<float> features(size());
  Vector<float> temp1(size());
  Vector<float> temp2(size());

  Grid grid(I,J);
  std::vector<Peak> & verts = grid.verts();

  LOG("detecting crosses");
  const Vector<float> background(size(), const_cast<float *>(background_data));
  image = background;
  Image::hdr_real(I,J,R, image, features, temp1, temp2);
  Image::quadratic_blur_scaled(I,J,R, image, features);
  Image::enhance_crosses(I,J,R, image, features);

  if (mask_data) {
    const Vector<float> mask(size(), const_cast<float *>(mask_data));
    features *= mask;
  }

  features /= max(features);

  find_peaks(I,J, max_detections, min_intensity, features, verts);

  for (size_t i = 0; i < verts.size(); ++i) {
    Peak & vert = verts[i];
    if (transpose) std::swap(vert.x, vert.y);
    m_pix_to_radial(vert);

    // compensate for distance from camera
    vert.z *= 1 + sqr(vert.x) + sqr(vert.y);
  }

  LOG("fitting grid to " << grid.verts().size() << " crosses");
  grid.fit(m_em_steps, m_nls_steps, m_bp_steps);
  m_distortion_param = grid.distortion_param();
  m_offset = grid.offset();

  PRINT(grid.distortion_param());
  PRINT(grid.offset());

  if (m_debug) {
    if (transpose) {
      Image::transpose_8(I,J, features, image);
      features = image;
      std::swap(I,J);
    }
    Image::write_image("data/crosses.im", I,J, features);
    grid.save_soln();
  }
}

//----( grit fitting )--------------------------------------------------------

Grid::Grid (
    float width_pix,
    float height_pix)

  : m_param_mean(size_in()),
    m_param_sigma(size_in()),
    m_offset(0,0),

    m_width_pix(width_pix),
    m_height_pix(height_pix),
    m_radius_pix(sqrtf(sqr(width_pix) + sqr(height_pix)) / 2)
{
  copy_float(g_distortion_mean, m_param_mean, distortion_size);
  copy_float(g_distortion_sigma, m_param_sigma, distortion_size);
}

void Grid::update_points (const float * param)
{
  if (not param) param = m_param_mean;

  for (size_t v = 0; v < m_verts.size(); ++v) {
    Point & point = m_points[v];
    distortion_transform(param, m_verts[v], point);
    point.x += m_offset.x;
    point.y += m_offset.y;
  }
}

void Grid::save_soln ()
{
  update_points();

  Image::Peaks peaks;
  for (size_t i = 0; i < m_verts.size(); ++i) {
    peaks.push_back(Image::Peak(m_points[i].x, m_points[i].y, m_verts[i].z));
  }
  LOG("writing solution to data/verts.text, data/edges.text");
  {
    std::ofstream file("data/verts.text");
    file << peaks;
  }
  {
    std::ofstream file("data/edges.text");
    file << m_edges;
  }
}

void Grid::fit (size_t em_steps, size_t nls_steps, size_t bp_steps)
{
  LOG("estimating extent");
  float min_x = INFINITY, max_x = -INFINITY;
  float min_y = INFINITY, max_y = -INFINITY;
  for (size_t v = 0; v < m_verts.size(); ++v) {
    Peak & vert = m_verts[v];
    imin(min_x, vert.x);
    imax(max_x, vert.x);
    imin(min_y, vert.y);
    imax(max_y, vert.y);
  }
  float estimated_p1 = logf(GRID_SIZE_X / (max_x - min_x));
  float estimated_t1 = logf(GRID_SIZE_Y / (max_y - min_y));
  m_param_mean[brown_size + 1] = estimated_p1;
  m_param_mean[brown_size + 2] = estimated_t1;
  PRINT2(estimated_p1, estimated_t1);

  LOG("estimating distortion parameters");
  m_points.resize(m_verts.size());
  Regression::FunctionWithPrior fun(* this, m_param_mean, m_param_sigma);
  float chi2_dof = INFINITY;
  for (size_t em_step = 0; em_step < em_steps; ++em_step) {
    if (bp_steps) update_edges_bp(bp_steps - 1); else update_edges_naive();
    chi2_dof = Regression::nonlinear_least_squares(
        fun,
        m_param_mean,
        fun.cov,
        nls_steps);
  }
  ASSERTW_LT(chi2_dof, 10);

  LOG("estimating alignment");
  update_points();

  for (size_t v = 0; v < m_verts.size(); ++v) {
    m_verts[v].z = 0;
  }
  for (size_t e = 0; e < m_edges.size(); ++e) {
    Edge & edge = m_edges[e];
    m_verts[edge.head].z += edge.weight;
    m_verts[edge.tail].z += edge.weight;
  }

  complex phase_x;
  complex phase_y;
  for (size_t v = 0; v < m_verts.size(); ++v) {
    Peak & vert = m_verts[v];
    Point & point = m_points[v];

    phase_x += vert.z * exp_2_pi_i(point.x);
    phase_y += vert.z * exp_2_pi_i(point.y);
  }

  PRINT2(phase_x, phase_y);
  m_offset.x -= arg(phase_x) / (2 * M_PI);
  m_offset.y -= arg(phase_y) / (2 * M_PI);
}

//----( least-squares objective function )----

void Grid::operator() (const Vector<float> & input, Vector<float> & output)
{
  ASSERT_SIZE(input, size_in());
  ASSERT_SIZE(output, size_out());

  size_t E = m_edges.size();
  size_t o = 0;

  // intrinsic deviations
  update_points(input);

  for (size_t e = 0; e < E; ++e) {
    Edge & edge = m_edges[e];
    Point & head = m_points[edge.head];
    Point & tail = m_points[edge.tail];

    float dx = head.x - tail.x;
    float dy = head.y - tail.y;
    (edge.vertical ? dy : dx) -= 1;

    output[o++] = dx * edge.weight;
    output[o++] = dy * edge.weight;
  }

  ASSERT_EQ(o,size_out());
}

//----( matching probabilities )----

#define GRID_ENERGY_FA                  (bernoulli_energy_gap(1.0f / 8))
#define GRID_ENERGY_MD                  (bernoulli_energy_gap(1.0f / 4))
#define GRID_LENGTH_SIGMA               (0.5f)
#define GRID_ENERGY_ANGLE               (logf(10.0f))
#define GRID_MAX_CHOICES                (6)


inline float Grid::vertex_fa_cost (float z)
{
  return GRID_ENERGY_FA + logf(z) / 2;
}

inline float Grid::vertex_md_cost (float distance_to_border)
{
  return GRID_ENERGY_MD + exponential_free_energy(distance_to_border);
}

inline float Grid::edge_cost (float dx, float dy)
{
  float r2 = sqr(dx) + sqr(dy);
  float cos_2_angle = (sqr(dx) - sqr(dy)) / r2;
  float cos_4_angle = 2 * sqr(cos_2_angle) - 1;
  return normal_free_energy(logf(r2) / 2, 0, GRID_LENGTH_SIGMA)
       - GRID_ENERGY_ANGLE * cos_4_angle / 2
       + logf(r2);
}

inline float Grid::edge_accuracy (Id v1, Id v2)
{
  Point & point1 = m_points[v1];
  Point & point2 = m_points[v2];

  float dx = point2.x - point1.x;
  float dy = point2.y - point1.y;

  float r = sqrt(sqr(dx) + sqr(dy));
  return r;
}

//----( grid point matching )----

struct Neighbor
{
  size_t vert;
  float like;
  float weight;
  Neighbor () : vert(0), like(0), weight(0) {}
  Neighbor (size_t v, float l, float w) : vert(v), like(l), weight(w) {}

  void update (size_t v, float l, float w)
  {
    if (l > like) {
      like = l;
      vert = v;
      weight = w;
    }
  }

  Grid::Edge to (size_t v, bool vertical)
  {
    return Grid::Edge(vert, v, weight, vertical);
  }
  Grid::Edge from (size_t v, bool vertical)
  {
    return Grid::Edge(v, vert, weight, vertical);
  }
};
typedef std::vector<Neighbor> Neighbors;

void Grid::update_edges_naive ()
{
  m_edges.clear();
  update_points();

  LOG("find best NESW neighbor of each grid point");

  const size_t V = m_verts.size();
  Neighbors N(V), S(V), E(V), W(V);

  for (size_t v1 = 0; v1 < V; ++v1) {
    Peak & vert1 = m_verts[v1];
    Point point1 = m_points[v1];

    for (size_t v2 = v1+1; v2 < V; ++v2) {
      Peak & vert2 = m_verts[v2];
      Point point2 = m_points[v2];

      float dx = point2.x - point1.x;
      float dy = point2.y - point1.y;

      bool vertical = sqr(dy) > sqr(dx);

      float like = expf(
          - edge_cost(dx, dy)
          + vertex_fa_cost(vert1.z)
          + vertex_fa_cost(vert2.z)
          // TODO get position-dependent P(MD) working
          + GRID_ENERGY_MD
          + GRID_ENERGY_MD
          );

      // longer edges have better accuracy, in pixels
      float distance_pix = m_radius_pix * sqrtf( sqr(vert2.x - vert1.x)
                                              + sqr(vert2.y - vert1.y) );
      float weight = like * distance_pix;

      if (vertical) {
        if (dy > 0) {
          S[v2].update(v1, like, weight);
          N[v1].update(v2, like, weight);
        } else {
          N[v2].update(v1, like, weight);
          S[v1].update(v2, like, weight);
        }
      } else {
        if (dx > 0) {
          W[v2].update(v1, like, weight);
          E[v1].update(v2, like, weight);
        } else {
          E[v2].update(v1, like, weight);
          W[v1].update(v2, like, weight);
        }
      }
    }
  }

  for (size_t v = 0; v < V; ++v) {
    if (N[v].like) m_edges.push_back(N[v].to(v, true));
    if (S[v].like) m_edges.push_back(S[v].from(v, true));
    if (E[v].like) m_edges.push_back(E[v].to(v, false));
    if (W[v].like) m_edges.push_back(W[v].from(v, false));
  }

  LOG(" found " << m_edges.size() << " edges");
  ASSERT_LT(1, m_edges.size());
}

void Grid::update_edges_bp (size_t num_iters)
{
  typedef Propagate::GridMatchingProblem<GRID_MAX_CHOICES> Matching;
  Matching matching;

  LOG("add vertices to matching problem");
  PRINT(GRID_ENERGY_FA);
  PRINT(GRID_ENERGY_MD);
  PRINT(GRID_ENERGY_ANGLE);
  update_points();

  const size_t V = m_verts.size();
  for (size_t v = 0; v < V; ++v) {
    Peak & vert = m_verts[v];
    float fa = vertex_fa_cost(vert.z);

    // TODO get position-dependent P(MD) working
    //Point point = m_points[v];
    float N = GRID_ENERGY_MD; //vertex_md_cost(GRID_SIZE_Y / 2 - point.y);
    float S = GRID_ENERGY_MD; //vertex_md_cost(GRID_SIZE_Y / 2 + point.y);
    float E = GRID_ENERGY_MD; //vertex_md_cost(GRID_SIZE_X / 2 - point.x);
    float W = GRID_ENERGY_MD; //vertex_md_cost(GRID_SIZE_X / 2 + point.x);

    matching.add_point(v, fa, N,S,E,W);
  }

  LOG("add vertical + horizontal edges to matching problem");
  m_edges.clear();

  for (size_t v1 = 0; v1 < V; ++v1) {
    Point point1 = m_points[v1];

    for (size_t v2 = v1+1; v2 < V; ++v2) {
      Point point2 = m_points[v2];

      float dx = point2.x - point1.x;
      float dy = point2.y - point1.y;
      bool vertical = sqr(dy) > sqr(dx);
      float cost = edge_cost(dx, dy);

      if (vertical) {
        if (dy > 0) {
          matching.add_varc(v1, v2, cost);
        } else {
          matching.add_varc(v2, v1, cost);
        }
      } else {
        if (dx > 0) {
          matching.add_harc(v1, v2, cost);
        } else {
          matching.add_harc(v2, v1, cost);
        }
      }
    }
  }

  matching.solve(num_iters);

  LOG("add nonzero edges to grid");

  size_t num_varcs = 0;
  for (Matching::ArcIterator a = matching.varcs(); a; a.next()) {
    Id v1 = a.tail();
    Id v2 = a.head();
    m_edges.push_back(Edge(v1,v2, a.prob() * edge_accuracy(v1,v2), true));
    ++num_varcs;
  }

  size_t num_harcs = 0;
  for (Matching::ArcIterator a = matching.harcs(); a; a.next()) {
    Id v1 = a.tail();
    Id v2 = a.head();
    m_edges.push_back(Edge(v1,v2, a.prob() * edge_accuracy(v1,v2), false));
    ++num_harcs;
  }

  PRINT2(num_varcs, num_harcs);
}

} // namespace Calibration

//----( visualization )-------------------------------------------------------

namespace Streaming
{

CalibrationVisualizer::CalibrationVisualizer (
    Rectangle shape,
    size_t finger_capacity)

  : Rectangle(shape),

    m_decay(expf(-1.0f / DEFAULT_SCREEN_FRAMERATE)),
    m_initialized(false),

    m_fingers(finger_capacity),

    fingers_in("CalibrationVisualizer.fingers_in", finger_capacity)
{}

void CalibrationVisualizer::pull (Seconds time, MonoImage & image)
{
  const size_t X = width();
  const size_t Y = height();
  const float scale_x = X / GRID_SIZE_X;
  const float scale_y = Y / GRID_SIZE_Y;
  const float shift_x = 0.5f * X;
  const float shift_y = 0.5f * Y;

  if (not m_initialized) {
    image.zero();
    m_initialized = true;
  }

  image *= m_decay;

  fingers_in.pull(time, m_fingers);

  for (size_t i = 0; i < m_fingers.size; ++i) {
    const Gestures::Finger & finger = m_fingers.values[i];

    float pos_x = finger.get_x() * scale_x + shift_x;
    float pos_y = finger.get_y() * scale_y + shift_y;
    float pos_z = finger.get_z();

    BilinearInterpolate(pos_x, X, pos_y, Y).imax(image, pos_z);
  }
}

} // namespace Streaming

