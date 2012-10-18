
#ifndef KAZOO_TABLE_H
#define KAZOO_TABLE_H

#include "common.h"
#include "streaming_audio.h"
#include "streaming_video.h"
#include "streaming_camera.h"
#include "streaming_devices.h"
#include "streaming_synthesis.h"
#include "gestures.h"
#include "tracker.h"
#include "calibration.h"

#define TABLE_HAND_CAPACITY             (4)
#define TABLE_FINGER_CAPACITY           (TABLE_HAND_CAPACITY * FINGERS_PER_HAND)
#define HAND_DETECTOR_POWER             (0.5f)

namespace Streaming
{

//----( table components )----------------------------------------------------

class ShadowTable
{
protected:

  RegionThread m_camera;

  StereoAudioThread m_audio;
  PowerMeter m_power;
  Denoiser m_denoiser;
  SpeakerGain m_gain;

public:

  RectangularPort<Pushed<MonoImage> > & shadow_out;
  Port<Pushed<float> > & impact_out;
  Port<Pulled<StereoAudioFrame> > & sound_in;

  ShadowTable (bool deaf = false);

  size_t finger_diameter () const { return 2+1+2; }
};

// TODO add Calibration::Calibrate
class FingerTable : protected ShadowTable
{
protected:

  EnhancePoints m_points;
  ExtractMoments m_moments;
  MomentsToFinger m_finger;

  Shared<float> m_impact;

public:

  Pulled<Gestures::Finger> & finger_out;
  Pulled<float> & impact_out;
  Port<Pulled<StereoAudioFrame> > & sound_in;

  FingerTable (bool deaf = false);
};

class FingersTable : protected ShadowTable
{
protected:

  EnhancePoints m_points;
  PeakDetector m_peaks;
  Tracking::Tracker m_tracker;
  ImpactDistributor m_distributor;
  Calibration::Calibrate m_calibrate;

  Shared<float> m_impact;

public:

  Pulled<BoundedMap<Id, Gestures::Finger> > & fingers_out;
  Pulled<float> & impact_out;
  Port<Pulled<StereoAudioFrame> > & sound_in;

  FingersTable (
      bool deaf = false,
      size_t finger_capacity = TABLE_FINGER_CAPACITY,
      bool debug_calibration = false);
};

class HandsTable : protected ShadowTable
{
  const size_t m_hand_capacity;
  const size_t m_finger_capacity;

protected:

  Splitter<MonoImage> m_splitter;

  ShrinkBy4 m_shrink;
  Pusher<MonoImage, size_t> m_pusher;
  EnhancePoints m_hand_image;
  PeakDetector m_hand_peaks;
  Tracking::Tracker m_hand_tracker;
  Calibration::Calibrate m_hand_calibrate;

  EnhancePoints m_finger_image;
  PeakDetector m_finger_peaks;
  Tracking::Tracker m_finger_tracker;
  Calibration::Calibrate m_finger_calibrate;

  RelativizeFingers m_relativize;
  ImpactDistributor m_distributor;

  Shared<float> m_impact;

public:

  Pulled<BoundedMap<Id, Gestures::Finger> > & hands_out;
  Pulled<BoundedMap<Id, Gestures::Finger> > & fingers_out;
  Pulled<float> & impact_out;
  Port<Pulled<StereoAudioFrame> > & sound_in;

  HandsTable (
      bool deaf = false,
      size_t hand_capacity = TABLE_HAND_CAPACITY,
      bool debug_calibration = false);
};

} // namespace Streaming

#endif // KAZOO_TABLE_H

