
#include "tracker.h"
#include "images.h"
#include "splines.h"
#include <vector>
#include <utility>
#include <algorithm>

#define LOG1(message)
#define LOG_EVENT LOG1
#define ASSERT2_EQ(x,y)
#define ASSERT2_LT(x,y)

#define DEFAULT_MEASUREMENT_NOISE_XY    (sqr(1.0f))
#define DEFAULT_MEASUREMENT_NOISE_Z     (sqr(0.2f))
#define DEFAULT_PROCESS_NOISE_XY        (sqr(1e1f))
#define DEFAULT_PROCESS_NOISE_Z         (sqr(1.0f))
#define DEFAULT_VEL_VARIANCE_XY         (sqr(1e1f))
#define DEFAULT_VEL_VARIANCE_Z          (sqr(1e1f))
#define DEFAULT_TRACK_TIMESCALE         (4.0f)
#define DEFAULT_INTENSITY_SCALE         (0.15f)

#define DEFAULT_MAX_ASSOC_RADIUS_XY     (6.0f)
#define DEFAULT_MAX_ASSOC_RADIUS_Z      (1.0f)
#define DEFAULT_MAX_ASSOC_COST          (10.0f)
#define DEFAULT_MIN_TRACK_UPDATES       (1)
#define DEFAULT_MATCHING_ITERATIONS     (20)

#define TOL (1e-4f)

namespace Tracking
{

//----( models )--------------------------------------------------------------

Model::Model (const char * config_filename)
  : m_config(config_filename),

    measurement_noise(
        m_config("measurement_noise_xy", DEFAULT_MEASUREMENT_NOISE_XY),
        m_config("measurement_noise_xy", DEFAULT_MEASUREMENT_NOISE_XY),
        m_config("measurement_noise_z", DEFAULT_MEASUREMENT_NOISE_Z)),
    process_noise(
        m_config("process_noise_xy", DEFAULT_PROCESS_NOISE_XY),
        m_config("process_noise_xy", DEFAULT_PROCESS_NOISE_XY),
        m_config("process_noise_z", DEFAULT_PROCESS_NOISE_Z)),
    initial_vel_variance(
        m_config("initial_vel_variance_xy", DEFAULT_VEL_VARIANCE_XY),
        m_config("initial_vel_variance_xy", DEFAULT_VEL_VARIANCE_XY),
        m_config("initial_vel_variance_z", DEFAULT_VEL_VARIANCE_Z)),
    track_timescale(m_config("track_timescale", DEFAULT_TRACK_TIMESCALE)),
    intensity_scale(m_config("intensity_scale", DEFAULT_INTENSITY_SCALE)),

    max_assoc_radius_xy(
        m_config("max_assoc_radius_xy", DEFAULT_MAX_ASSOC_RADIUS_XY)),
    max_assoc_radius_z(
        m_config("max_assoc_radius_z", DEFAULT_MAX_ASSOC_RADIUS_Z)),
    max_assoc_radius_inverse(
      1 / max_assoc_radius_xy,
      1 / max_assoc_radius_xy,
      1 / max_assoc_radius_z),
    max_assoc_cost(m_config("max_assoc_cost", DEFAULT_MAX_ASSOC_COST)),

    min_track_updates(
      m_config("min_track_updates", DEFAULT_MIN_TRACK_UPDATES)),

    matching_iterations(
        m_config("matching_iterations", DEFAULT_MATCHING_ITERATIONS)),

    m_chi2(0,0,0),
    m_dof(0)
{
  ASSERT_LT(0, track_timescale);
  ASSERT_LT(0, intensity_scale);
  ASSERT_LE(TOL, max_assoc_radius_xy);
  ASSERT_LE(TOL, max_assoc_radius_z);
  ASSERT_LT(0, matching_iterations);
  ASSERTW_LE(matching_iterations, 100);
}

Model::~Model ()
{
  if (m_dof) {
    PRINT(process_noise);
    PRINT(chi2_dof());
  }
}

//----( tracks )--------------------------------------------------------------

size_t Track::s_new_id = 0;

uint64_t Track::s_sum_updates = 0;
uint64_t Track::s_sum_sqr_updates = 0;

double Track::s_sum_ages = 0;

Track::~Track ()
{
  s_sum_updates += 1 + m_num_updates;
  s_sum_sqr_updates += m_num_updates * (1 + m_num_updates);

  s_sum_ages += m_age;
}

//----( tracker )-------------------------------------------------------------

Tracker::Tracker (
    size_t detection_capacity,
    const char * config_filename)

  : m_model(config_filename),

    m_time(Seconds::now()),

    m_detections(detection_capacity),

    m_num_frames(0),
    m_num_detections(0),
    m_var_track_intensity(0),
    m_var_det_intensity(0),
    m_cov_track_det_intensity(0)
{
}

Tracker::~Tracker ()
{
  float dets_per_frame = 1.0f * m_num_detections / m_num_frames;
  LOG("Tracker averaged " << dets_per_frame
      << " = " << m_num_detections << " detections"
      << " / " << m_num_frames << " frames");

  PRINT(Track::mean_num_updates());
  PRINT(Track::mean_age());

  float cov_scale = sqrt(m_var_track_intensity * m_var_det_intensity);
  if (cov_scale > 0) {
    float track_det_intensity_correlation = m_cov_track_det_intensity
                                         / cov_scale;
    PRINT(track_det_intensity_correlation);
  }

  PRINT(m_timestep);
  PRINT(m_num_arcs);

  clear();
}

//----( high-level operations )----

void Tracker::set_time (Seconds time)
{
  m_mutex.lock(); //----( begin lock )----------------------------------------

  m_time = time;

  m_mutex.unlock(); //----( end lock )----------------------------------------
}

void Tracker::get_best_tracks (size_t capacity)
{
  // this assumes m_mutex is locked

  m_best_tracks.clear();

  typedef Tracks::iterator Auto;
  for (Auto t = m_tracks.begin(); t != m_tracks.end(); ++t) {
    const Track & track = *(t->second);
    if (track.num_updates() < m_model.min_track_updates) continue;
    Id id = t->first;

    m_best_tracks.push_back(std::make_pair(-intensity(track), id));
  }

  if (m_best_tracks.size() > capacity) {
    std::nth_element(
        m_best_tracks.begin(),
        m_best_tracks.begin() + capacity,
        m_best_tracks.end());
    m_best_tracks.resize(capacity);
  }
}

void Tracker::pull (Seconds time, BoundedMap<Id, Position> & positions)
{
  positions.clear();

  m_mutex.lock(); //----( begin lock )----------------------------------------

  float dt = time - m_time;
  ASSERTW_LE(0, dt);

  get_best_tracks(positions.capacity);
  size_t num_tracks = m_best_tracks.size();
  positions.resize(num_tracks);
  for (size_t i = 0; i < num_tracks; ++i) {
    Id id = m_best_tracks[i].second;
    positions.keys[i] = id;
    positions.values[i] = m_tracks[id]->predict(dt);
  }

  m_mutex.unlock(); //----( end lock )----------------------------------------
}

void Tracker::pull (Seconds time, BoundedMap<Id, Finger> & fingers)
{
  fingers.clear();

  m_mutex.lock(); //----( begin lock )----------------------------------------

  float dt = time - m_time;
  ASSERTW_LE(0, dt);

  get_best_tracks(fingers.capacity);
  size_t num_tracks = m_best_tracks.size();
  fingers.resize(num_tracks);
  for (size_t i = 0; i < num_tracks; ++i) {
    Id id = m_best_tracks[i].second;
    fingers.keys[i] = id;
    m_tracks[id]->predict(dt, fingers.values[i]);
  }

  m_mutex.unlock(); //----( end lock )----------------------------------------

  m_best_tracks.clear();
}

void Tracker::push (Seconds time, const Image::Peaks & peaks)
{
  ASSERTW_LE(peaks.size(), m_detections.size);
  const size_t num_detections = min(m_detections.size, peaks.size());
  m_num_detections += num_detections;
  ++m_num_frames;

  for (size_t j = 0, J = num_detections; j < J; ++j) {
    m_model.detect(peaks[j], m_detections[j]);
  }

  m_mutex.lock(); //----( begin lock )----------------------------------------

  LOG1("advancing by " << dt << "s");

  float dt = time - m_time;
  ASSERT_LT(0, dt);
  m_time = time;
  m_timestep.add(dt);

  typedef Tracks::iterator Auto;
  for (Auto t = m_tracks.begin(); t != m_tracks.end(); ++t) {
    Track & track = *(t->second);
    track.advance(dt, m_model.process_noise);
    m_var_track_intensity += sqr(intensity(track));
  }

  for (size_t j = 0, J = num_detections; j < J; ++j) {
    m_var_det_intensity += sqr(intensity(m_detections[j]));
  }

  m_mutex.unlock(); //----( end lock )----------------------------------------

  LOG1("probabilistically associate observations to tracks");

  for (size_t j = 0, J = num_detections; j < J; ++j) {
    m_matching.add2(track_begin_cost(m_detections[j], dt));
  }

  typedef Tracks::iterator Auto;
  for (Auto t = m_tracks.begin(); t != m_tracks.end(); ++t) {
    size_t i = m_ids.size();
    m_ids.push_back(t->first);

    const Track & track = *(t->second);
    m_matching.add1(track_end_cost(track, dt));

    for (size_t j = 0; j < num_detections; ++j) {
      const Detection & detection = m_detections[j];
      if (not nearby(track, detection)) continue;

      float cost = track_continue_cost(track, detection);
      if (not (cost < m_model.max_assoc_cost)) continue;

      m_matching.add12(i, j, cost);
    }
  }

  m_num_arcs.add(m_matching.size_arc());
  //m_matching.print_prior(); // DEBUG
  //m_matching.validate_problem(); // DEBUG
  m_matching.solve(m_model.matching_iterations);
  //m_matching.print_post(); // DEBUG
  //m_matching.validate_solution(); // DEBUG

  m_mutex.lock(); //----( begin lock )----------------------------------------

  LOG1("end tracks with no associated observations")
  for (size_t i = 0, I = m_matching.size_1(); i < I; ++i) {
    if (m_matching.post_1_non(i)) {
      Id id = m_ids[i];

      LOG_EVENT("ending track " << id);
      end_track(id);
    }
  }

  LOG1("update continuing tracks");
  for (size_t ij = 0, IJ = m_matching.size_arc(); ij < IJ; ++ij) {
    if (m_matching.post_ass(ij)) {
      Matching::Arc arc = m_matching.arc(ij);
      Id id = m_ids[arc.i];

      LOG_EVENT("continuing track " << id << " with detection " << arc.j);
      Track & track = *(m_tracks.find(id)->second);
      const Detection & detection = m_detections[arc.j];

      m_cov_track_det_intensity += intensity(track) * intensity(detection);

      Position chi = track.update(detection);
      m_model.sample(sqr(chi), dt);
    }
  }

  LOG1("begin tracks for new unassociated observations");
  for (size_t j = 0, J = m_matching.size_2(); j < J; ++j) {
    if (m_matching.post_2_non(j)) {

      Track & track __attribute__ ((unused)) = begin_track(m_detections[j]);
      LOG_EVENT("begining track " << track.id << " with detection " << j);
    }
  }

  m_mutex.unlock(); //----( end lock )----------------------------------------

  m_ids.clear();
  m_matching.clear();
}

void Tracker::clear ()
{
  typedef Tracks::iterator Auto;
  for (Auto t = m_tracks.begin(); t != m_tracks.end(); ++t) {
    delete t->second;
  }
  m_tracks.clear();

  m_num_frames = 0;
  m_num_detections = 0;
}

//----( track operations )----

inline Track & Tracker::begin_track (const Detection & detection)
{
  Track * track = new Track(detection, m_model.initial_vel_variance);
  m_tracks.insert(std::make_pair(track->id, track));
  return * track;
}

inline void Tracker::end_track (Id id)
{
  typedef Tracks::iterator Auto;
  Auto t = m_tracks.find(id);
  ASSERT(t != m_tracks.end(), "track not found when ending");
  delete t->second;
  m_tracks.erase(t);
}

} // namespace Tracking

//----( visualization )-------------------------------------------------------

namespace Streaming
{

TrackVisualizer::TrackVisualizer (
    Rectangle shape,
    size_t track_capacity)

  : Rectangle(shape),

    m_track_decay(expf(-1.0f / DEFAULT_SCREEN_FRAMERATE)),

    m_peaks(),
    m_tracks(track_capacity),

    image_in(shape.size()),
    detections_in(Image::Peaks()),
    tracks_in("TrackVisualizer.tracks_in", track_capacity)
{
  ASSERT_LT(0, track_capacity)
}

void TrackVisualizer::pull (Seconds time, RgbImage & image)
{
  const size_t X = width();
  const size_t Y = height();

  // image in blue
  MonoImage blue(image.blue);
  image_in.pull(time, blue);

  // detections in green
  image.green.zero();
  detections_in.pull(time, m_peaks);
  for (size_t i = 0; i < m_peaks.size(); ++i) {
    Image::Peak & peak = m_peaks[i];

    BilinearInterpolate(peak.x, X, peak.y, Y).imax(image.green, 1);
  }

  // tracks in red
  image.red *= m_track_decay;
  tracks_in.pull(time, m_tracks);
  for (size_t i = 0; i < m_tracks.size; ++i) {
    Tracking::Position & position = m_tracks.values[i];

    BilinearInterpolate(position[0], X, position[1], Y).imax(image.red, 1);
  }
}

TrackAgeVisualizer::TrackAgeVisualizer (
    Rectangle shape,
    size_t track_capacity,
    float age_scale)

  : Rectangle(shape),

    m_track_decay(expf(-1.0f / DEFAULT_SCREEN_FRAMERATE)),
    m_age_scale(age_scale),

    m_tracks(track_capacity),

    tracks_in("TrackAgeVisualizer.tracks_in", track_capacity)
{
  ASSERT_LT(0, track_capacity)
  ASSERT_LE(1, age_scale)
}

void TrackAgeVisualizer::pull (Seconds time, RgbImage & image)
{
  ASSERT_SIZE(image.red, size());

  tracks_in.pull(time, m_tracks);

  // update ages
  std::swap(m_ages, m_ended);
  for (size_t i = 0; i < m_tracks.size; ++i) {
    Id id = m_tracks.keys[i];
    if (m_ended.find(id) == m_ended.end()) {
      m_ages[id] = 0;
    } else {
      m_ages[id] = 1 + m_ended[id];
    }
  }
  m_ended.clear();

  // draw tracks
  const size_t X = width();
  const size_t Y = height();

  image *= m_track_decay;

  for (size_t i = 0; i < m_tracks.size; ++i) {
    Id id = m_tracks.keys[i];

    float age = m_ages[id] / m_age_scale;
    float r = expf(-age);
    float g = age * expf(1 - age);
    float b = 1 - (r + g) / 2;

    Tracking::Position & position = m_tracks.values[i];
    BilinearInterpolate lin(position[0], X, position[1], Y);

    lin.imax(image.red, r);
    lin.imax(image.green, g);
    lin.imax(image.blue, b);
  }
}

} // namespace Streaming

