
#include "streaming_clouds.h"

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace Streaming
{

//----( distance logger )-----------------------------------------------------

DistanceLogger::DistanceLogger (size_t size, bool verbose)
  : m_prev(size),
    m_continuing(false),
    m_total_squared_distance(0),
    m_num_frames(0),
    m_verbose(verbose)
{}

DistanceLogger::~DistanceLogger ()
{
  size_t num_pairs = m_num_frames - 1;
  if (num_pairs) {
    double rms_point_distance = sqrt(m_total_squared_distance / num_pairs);
    PRINT2(rms_point_distance, num_pairs);
  }
}

double DistanceLogger::get_rms_distance ()
{
  size_t num_pairs = m_num_frames - 1;
  if (num_pairs) {
    return sqrt(m_total_squared_distance / num_pairs);
  } else {
    return 0;
  }
}

void DistanceLogger::add (const Cloud::Point & point)
{
  if (m_continuing) {
    float squared_distance = Cloud::squared_distance(point, m_prev);
    m_total_squared_distance += squared_distance;

    if (m_verbose) {
      float point_distance = sqrtf(squared_distance);
      PRINT(point_distance);
    }
  }

  m_prev = point;
  m_continuing = true;
}

void DistanceLogger::add (
    const Cloud::Point & point,
    const Cloud::Point & prev)
{
  ++m_num_frames;

  m_total_squared_distance += Cloud::squared_distance(point, prev);

  // this is unnecessary in the pairwise mode of use
  //m_prev = point;
}

//----( denoiser )------------------------------------------------------------

CloudDenoiser::CloudDenoiser (const Cloud::PointSet & points, size_t iters)
  : m_points(points),
    m_iters(iters),
    m_likes(points.size),
    m_point(points.dim),
    out("CloudDenoiser.out", points.dim)
{
  ASSERT_LT(0, iters);
}

void CloudDenoiser::push (Seconds time, const Cloud::Point & point)
{
  m_points.quantize(point, m_likes);
  m_points.construct(m_likes, m_point);

  for (size_t iters = 1; iters < m_iters; ++iters) {
    m_points.quantize(m_point, m_likes);
    m_points.construct(m_likes, m_point);
  }

  out.push(time, m_point);
}

//----( simulator )-----------------------------------------------------------

CloudSimulator::CloudSimulator (
    const Cloud::JointPrior & flow,
    float framerate)

  : TimedThread(framerate),

    m_points(flow.dom),
    m_flow(flow),

    m_state(0),
    m_next(* new VectorXf(m_points.size)),
    m_point(m_points.dim),

    out("CloudSimulator.out", m_points.dim)
{
  reset();
}

CloudSimulator::~CloudSimulator ()
{
  delete & m_next;
}

void CloudSimulator::reset ()
{
  m_mutex.lock();

  m_points.get_prior(m_next);
  m_state = Cloud::random_index(m_next);

  m_mutex.unlock();
}

void CloudSimulator::step ()
{
  m_mutex.lock();

  VectorSf next_like = m_flow.joint.col(m_state);
  m_state = Cloud::random_index(next_like);

  m_points.get_point(m_state, m_point);

  m_mutex.unlock();

  out.push(Seconds::now(), m_point);

  PROGRESS_TICKER('s');
}

void CloudSimulator::keyboard (const SDL_KeyboardEvent & event)
{
  switch (event.keysym.sym) {
    case SDLK_SPACE:
      if (event.type == SDL_KEYDOWN) reset();
      break;

    default:
      break;
  }
}

//----( diffuser )------------------------------------------------------------

CloudDiffuser::CloudDiffuser (
    const Cloud::JointPrior & flow,
    float framerate)

  : TimedThread(framerate),

    m_points(flow.dom),
    m_flow(flow),


    m_state(* new VectorXf(m_points.size)),
    m_next(* new VectorXf(m_points.size)),
    m_point(m_points.dim),

    out("CloudDiffuser.out", m_points.dim)
{
  reset();
}

CloudDiffuser::~CloudDiffuser ()
{
  delete & m_state;
  delete & m_next;
}

void CloudDiffuser::reset ()
{
  m_mutex.lock();

  m_points.get_prior(m_next);
  int p = Cloud::random_index(m_next);

  m_next.setZero();
  m_next(p) = m_next.size();

  m_points.construct(m_next, m_point);
  m_points.quantize(m_point, m_state);

  m_mutex.unlock();
}

void CloudDiffuser::step ()
{
  m_mutex.lock();

  m_flow.push_forward(m_state, m_next);
  Cloud::normalize_l1(m_next, m_next.size());
  PRINT(Cloud::likelihood_entropy(m_next));

  m_points.construct(m_next, m_point);
  m_points.quantize(m_point, m_state);

  m_mutex.unlock();

  out.push(Seconds::now(), m_point);

  PROGRESS_TICKER('s');
}

void CloudDiffuser::keyboard (const SDL_KeyboardEvent & event)
{
  switch (event.keysym.sym) {
    case SDLK_SPACE:
      if (event.type == SDL_KEYDOWN) reset();
      break;

    default:
      break;
  }
}

//----( gloves <- gloves )----------------------------------------------------

GlovesToGloves::GlovesToGloves (Cloud::Controller & controller)
  : Rectangle(controller.cod().shape),
    m_controller(controller),
    m_image(controller.dom().shape.size()),
    out("GlovesToGloves.out", controller.dom().shape)
{
}

void GlovesToGloves::push (Seconds time, const Gloves8Image & image)
{
  ASSERT_SIZE(image, size() * Gloves8Image::num_channels);

  m_controller.update_track(image, m_image);

  out.push(time, m_image);
}

//----( gloves <- voice )-----------------------------------------------------

GlovesToVoice::GlovesToVoice (
      Cloud::Controller & controller,
      const char * voice_config)
  : Rectangle(controller.cod().shape),

    m_controller(controller),
    m_feature_processor(voice_config),
    m_voice_features(controller.dom().dim),

    out("GlovesToVoice.out", controller.dom().dim)
{
}

void GlovesToVoice::push (Seconds time, const Gloves8Image & image)
{
  ASSERT_SIZE(image, size() * Gloves8Image::num_channels);

  m_controller.update_track(image, m_voice_features);

  if (m_controller.is_tracking() and m_controller.is_coalescing()) {
    m_feature_processor.update_history(m_voice_features);
    m_controller.set_track(m_voice_features);
  }

  out.push(time, m_voice_features);
}

} // namespace Streaming

