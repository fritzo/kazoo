
#include "table.h"

#define TOL                             (1e-20f)

namespace Streaming
{

using Gestures::Finger;

//----( shadow table )--------------------------------------------------------

ShadowTable::ShadowTable (bool deaf)
  : m_camera(),

    m_audio(not deaf),
    m_power(),
    m_denoiser(),
    m_gain(),

    shadow_out(m_camera.out),
    impact_out(m_denoiser.out),
    sound_in(m_gain.in)
{
  if (not deaf) {
    m_audio.out - m_power;
    m_power.out - m_denoiser;
  }

  m_audio.in - m_gain;
}

//----( finger table )--------------------------------------------------------

FingerTable::FingerTable (bool deaf)
  : ShadowTable(deaf),

    m_points(m_camera.out),
    m_moments(m_points.out),
    m_finger(),

    m_impact(0),

    finger_out(m_finger),
    impact_out(m_impact),
    sound_in(ShadowTable::sound_in)
{
  shadow_out - m_points;
  m_points.out - m_moments;
  m_moments.out - m_finger;

  if (not deaf) {
    ShadowTable::impact_out - m_impact;
    m_finger.impact_in - m_impact;
  }
}

//----( fingers table )-------------------------------------------------------

FingersTable::FingersTable (
    bool deaf,
    size_t finger_capacity,
    bool debug_calibration)

  : ShadowTable(deaf),

    m_points(m_camera.out),
    m_peaks(m_points.out, 2 * finger_capacity),
    m_tracker(2 * finger_capacity),
    m_distributor(),
    m_calibrate(m_points.out, debug_calibration),

    m_impact(0),

    fingers_out(m_calibrate),
    impact_out(m_impact),
    sound_in(ShadowTable::sound_in)
{
  m_calibrate.fit_grid(m_camera.background(), m_camera.mask(), true);

  shadow_out - m_points;
  m_points.out - m_peaks;
  m_peaks.out - m_tracker;
  m_distributor.fingers_in - m_tracker;
  m_calibrate.fingers_in - m_distributor;

  if (not deaf) {
    ShadowTable::impact_out - m_impact;
    m_distributor.impact_in - m_impact;
  }
}

//----( hands table )---------------------------------------------------------

HandsTable::HandsTable (
    bool deaf,
    size_t hand_capacity,
    bool debug_calibration)

  : ShadowTable(deaf),

    m_hand_capacity(hand_capacity),
    m_finger_capacity(hand_capacity * FINGERS_PER_HAND),

    m_splitter(),

    m_shrink(m_camera.out),
    m_pusher(m_shrink.out.size()),
    m_hand_image(m_shrink.out, HAND_BLUR_RADIUS),
    m_hand_peaks(m_hand_image.out, 2 * m_hand_capacity, HAND_DETECTOR_POWER),
    m_hand_tracker(2 * m_hand_capacity),
    m_hand_calibrate(m_hand_image.out, debug_calibration),

    m_finger_image(m_camera.out, FINGER_BLUR_RADIUS),
    m_finger_peaks(m_finger_image.out, 2 * m_finger_capacity),
    m_finger_tracker(2 * m_finger_capacity),
    m_finger_calibrate(m_finger_image.out, debug_calibration),

    m_relativize(m_hand_capacity),
    m_distributor(),

    m_impact(0),

    hands_out(m_hand_calibrate),
    fingers_out(m_distributor),
    impact_out(m_impact),
    sound_in(ShadowTable::sound_in)
{
  m_finger_calibrate.fit_grid(m_camera.background(), m_camera.mask(), true);
  m_hand_calibrate = m_finger_calibrate;
  m_hand_calibrate.scale_input(0.25f);

  shadow_out - m_splitter;

  m_splitter.out1 - m_shrink;
  m_shrink.out - m_pusher;
  m_pusher.out - m_hand_image;
  m_hand_image.out - m_hand_peaks;
  m_hand_peaks.out - m_hand_tracker;
  m_hand_calibrate.fingers_in - m_hand_tracker;

  m_splitter.out2 - m_finger_image;
  m_finger_image.out - m_finger_peaks;
  m_finger_peaks.out - m_finger_tracker;
  m_finger_calibrate.fingers_in - m_finger_tracker;

  m_relativize.hands_in - m_hand_calibrate;
  m_relativize.fingers_in - m_finger_calibrate;
  m_distributor.fingers_in - m_relativize;

  if (not deaf) {
    ShadowTable::impact_out - m_impact;
    m_distributor.impact_in - m_impact;
  }
}

} // namespace Streaming

