#ifndef KAZOO_TRACKER_H
#define KAZOO_TRACKER_H

/** A Gaussian blob tracker.

  Purpose:
    The Tracker forms tracks from frames of (position,intensity) peaks,
    and provides interpolation and extrapolation of blob states.

  Method:
    Pos3s are assumed exact in position, extent, and intensity,
    but at a possibly poorly-resolved time.
    Blobs are modeled as nearly-constant velocity in all 4 parameters.

  Assumptions:
    We model each coordinate independently with an NCV dynamics model.

  Coordinate System:
    x,y are usually in finger units.

  TODO generalize from 3D Point to 5D finger or 6D Blob tracking
  TODO allow for time uncertainty in updating
*/

#include "common.h"
#include "streaming.h"
#include "image_types.h"
#include "vectors.h"
#include "bounded.h"
#include "array.h"
#include "filters.h"
#include "probability.h"
#include "gestures.h"
#include "matching.h"
#include "config.h"
#include "hash_map.h"

#define DEFAULT_TRACKER_VIS_AGE_SCALE         (10.0f)

namespace Tracking
{

using namespace Filters;
using namespace Gestures;

//----( data structures )-----------------------------------------------------

using Image::Peak;
using Image::Peaks;

enum { STATE_DIM = 3 }; // x (position), y (position), z (intensity)

typedef Array<float, STATE_DIM> Position;

typedef NCP<STATE_DIM> Detection;

//----( models )--------------------------------------------------------------

class Model : public Aligned<Model>
{
  ConfigParser m_config;

public:

  const Position measurement_noise;
  const Position process_noise;
  const Position initial_vel_variance;
  const float track_timescale;
  const float intensity_scale;

  const float max_assoc_radius_xy;
  const float max_assoc_radius_z;
  const Position max_assoc_radius_inverse;
  const float max_assoc_cost;
  const size_t min_track_updates;
  const size_t matching_iterations;

protected:

  Position m_chi2;
  float m_dof;

public:

  Model (const char * config_filename = NULL);
  ~Model ();

  void detect (const Peak & peak_in, Detection & detection_out)
  {
    detection_out.Ex[0] = peak_in.x;
    detection_out.Ex[1] = peak_in.y;
    detection_out.Ex[2] = peak_in.z;
    detection_out.Vxx = measurement_noise;
  }

  // goodness of fit
  void sample (Position chi2, float dof) { m_chi2 += chi2; m_dof += dof; }
  Position chi2_dof () const { return m_chi2 / (process_noise * m_dof); }
};

//----( tracks )--------------------------------------------------------------

/** Tracks.
  Track states consist of position + velocity.
  Tracks do not retain past measurements.
*/

class Track : public NCV<3>
{
  size_t m_num_updates;
  static uint64_t s_sum_updates;
  static uint64_t s_sum_sqr_updates;

  float m_age;
  static double s_sum_ages;

  // don't make vectors of these
  void * operator new[] (size_t) { ERROR("called new Track[];" ); }
  void operator delete[] (void *) { ERROR("called Track::delete[] _"); }

  static size_t s_new_id;
  static Id new_id () { return s_new_id++; }

public:

  const size_t id;

  Track ()
    : m_num_updates(0),
      m_age(0),
      id(new_id())
  {}
  Track (
      const Detection & detection,
      const Position & vel_variance)

    : NCV<3>(detection, vel_variance),
      m_num_updates(0),
      m_age(0),
      id(new_id())
  {}

  ~Track ();

  Position predict (float dt) const { return NCV<3>::predict(dt); }
  void predict (float dt, Finger & finger) const
  {
    finger.set_energy(1);
    finger.set_pos(predict(dt));
    finger.set_vel(Ey);
    finger.set_age(m_age + dt);
  }

  void advance (float dt, Position process_noise)
  {
    m_age += dt;
    NCV<3>::advance(dt, process_noise);
  }

  Position update (const Detection & observed)
  {
    ++m_num_updates;
    Position chi = NCV<3>::update(observed);
    return chi;
  }

  size_t num_updates () const { return m_num_updates; }
  static float mean_num_updates ()
  {
    return static_cast<float>(s_sum_sqr_updates) / s_sum_updates;
  }

  float age () const { return m_age; }
  static float mean_age () { return s_sum_ages / s_new_id; }
};

inline ostream & operator << (ostream & o, const Track & t)
{
  return o << "Track " << t.Ex << " @ " << t.Ey;
}

//----( tracker )-------------------------------------------------------------

/** A sparse density tracker
*/

class Tracker
  : public Aligned<Tracker>,
    public Streaming::Pushed<Image::Peaks>,
    public Streaming::Pulled<BoundedMap<Id, Position> >,
    public Streaming::Pulled<BoundedMap<Id, Finger> >
{
  Model m_model;

  Seconds m_time;
  typedef std::hash_map<Id, Track *> Tracks;
  Tracks m_tracks;
  std::vector<std::pair<float, Id> > m_best_tracks;

  std::vector<Id> m_ids;
  Vector<Detection> m_detections;
  Matching::HardMatching m_matching;

  // This mutex governs writing to tracks and tracker time.
  mutable Mutex m_mutex;

  // statistics
  size_t m_num_frames;
  size_t m_num_detections;
  float m_var_track_intensity;
  float m_var_det_intensity;
  float m_cov_track_det_intensity;
  Filters::DebugStats<float> m_timestep;
  Filters::DebugStats<float> m_num_arcs;

public:

  Tracker (
      size_t detection_capacity,
      const char * config_filename = "config/default.tracker.conf");
  virtual ~Tracker ();

  //----( thread-safe public interface )----

  Seconds get_time () const { return m_time; }
  void set_time (Seconds time);

  virtual void push (Seconds time, const Image::Peaks & peaks);

  virtual void pull (Seconds time, BoundedMap<Id, Position> & positions);
  virtual void pull (Seconds time, BoundedMap<Id, Finger> & fingers);

  void clear (); // not thread safe

private:

  void get_best_tracks (size_t capacity);

  //----( association tools )----

  bool nearby (const NCP<3> & pos1, const NCP<3> & pos2) const
  {
    return sum(sqr((pos1.Ex - pos2.Ex) * m_model.max_assoc_radius_inverse)) < 1;
  }

  //----( track events )----

  inline Track & begin_track (const Detection & initial);
  inline void end_track (Id id);

  //----( likelihood computations )----

  float intensity (const NCP<3> & pos) const
  {
    return pos.Ex[2] / m_model.intensity_scale;
  }

  // track topology costs
  float track_begin_cost (const Detection & detection, float dt) const
  {
    return exponential_cdf_energy_gap(dt / m_model.track_timescale)
         + exponential_free_energy(intensity(detection));
  }
  float track_end_cost (const Track & track, float dt) const
  {
    return exponential_cdf_energy_gap(dt / m_model.track_timescale)
         + exponential_free_energy(intensity(track));
  }
  float track_continue_cost (
      const Track & track,
      const Detection & detection) const
  {
    return track.free_energy(detection);
  }
};

} // namespace Tracking

//----( visualization )-------------------------------------------------------

namespace Streaming
{

class TrackVisualizer
  : public Rectangle,
    public Pulled<RgbImage>
{
  const float m_track_decay;

  Image::Peaks m_peaks;
  BoundedMap<Id, Tracking::Position> m_tracks;

public:

  Shared<MonoImage, size_t> image_in;
  Shared<Image::Peaks> detections_in;
  SizedPort<Pulled<BoundedMap<Id, Tracking::Position> > > tracks_in;

  TrackVisualizer (Rectangle shape, size_t track_capacity);
  virtual ~TrackVisualizer () {}

  virtual void pull (Seconds time, RgbImage & image);
};

class TrackAgeVisualizer
  : public Rectangle,
    public Pulled<RgbImage>
{
  const float m_track_decay;
  const float m_age_scale;

  BoundedMap<Id, Tracking::Position> m_tracks;

  std::hash_map<Id, size_t> m_ages, m_ended;

public:

  SizedPort<Pulled<BoundedMap<Id, Tracking::Position> > > tracks_in;

  TrackAgeVisualizer (
      Rectangle shape,
      size_t track_capacity,
      float age_scale = DEFAULT_TRACKER_VIS_AGE_SCALE);
  virtual ~TrackAgeVisualizer () {}

  virtual void pull (Seconds time, RgbImage & image);
};

} //namespace Streaming

#endif // KAZOO_TRACKER_H
