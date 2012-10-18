
#ifndef KAZOO_BALL_H
#define KAZOO_BALL_H

#include "common.h"
#include "streaming_audio.h"
#include "streaming_video.h"
#include "streaming_camera.h"
#include "streaming_devices.h"
#include "synthesis.h"
#include "tracker.h"
#include "filters.h"

#define BALL_FINGER_CAPACITY          (24)

namespace Streaming
{

//----( ball components )-----------------------------------------------------

class ShadowBall
{
  enum { finger_radius = 1 };
  RegionThread m_camera;
  ShrinkBy2 m_shrink;
  EnhancePoints m_fingers;

public:

  RectangularPort<Pushed<MonoImage> > & out;

  ShadowBall ()
    : m_camera(),
      m_shrink(m_camera.out),
      m_fingers(m_shrink.out, finger_radius),

      out(m_fingers.out)
  {
    m_camera.out - m_shrink;
    m_shrink.out - m_fingers;
  }
};

class FingersBall : protected ShadowBall
{
  PeakDetector m_peaks;
  Tracking::Tracker m_tracker;
  ImpactDistributor m_distributor;
  PixToPolar m_calibrate;

  Denoiser m_denoiser;
  Shared<float> m_impact;

public:

  Pushed<float> & power_in;
  Pulled<BoundedMap<Id, Gestures::Finger> > & fingers_out;

  FingersBall (
      bool deaf = false,
      size_t finger_capacity = BALL_FINGER_CAPACITY)

    : ShadowBall(),

      m_peaks(ShadowBall::out, 2 * finger_capacity),
      m_tracker(2 * finger_capacity),
      m_distributor(),
      m_calibrate(ShadowBall::out, false),
      m_denoiser(),
      m_impact(0),

      power_in(m_denoiser),
      fingers_out(m_calibrate)
  {
    ShadowBall::out - m_peaks;
    m_peaks.out - m_tracker;
    m_distributor.fingers_in - m_tracker;
    m_calibrate.in - m_distributor;

    if (not deaf) {
      m_denoiser.out - m_impact;
      m_distributor.impact_in - m_impact;
    }
  }
};

} // namespace Streaming

#endif // KAZOO_BALL_H

