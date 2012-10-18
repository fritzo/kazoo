
#ifndef KAZOO_STREAMING_VIDEO_H
#define KAZOO_STREAMING_VIDEO_H

#include "common.h"
#include "streaming.h"
#include "splines.h"
#include "psycho.h"
#include "filters.h"
#include "image_types.h"
#include "gestures.h"

#define FINGER_BLUR_RADIUS              (2)
#define HAND_BLUR_RADIUS                (4)

#define PEAK_DETECTOR_POWER             (1.0f)
#define PEAK_DETECTOR_INIT_GAIN         (0.05)
#define PEAK_DETECTOR_MIN_RATIO         (0.02)

#define OPTICAL_FLOW_HIGHPASS_RADIUS    (8)
#define OPTICAL_FLOW_LOWPASS_RADIUS     (4)
#define OPTICAL_FLOW_SPACESCALE         (2)
#define OPTICAL_FLOW_PROCESS_NOISE      (1e-2f)
#define OPTICAL_FLOW_PRIOR_PER_PIX      (1e-1f)

namespace Streaming
{

//----( mosaic )--------------------------------------------------------------

class Mosaic : public Pulled<MonoImage>
{
  std::vector<Shared<MonoImage, size_t> *> m_images;

public:

  const Rectangle shape_in;
  const Rectangle shape_out;
  size_t size () { return m_images.size(); }

  Pushed<MonoImage> & in (size_t i);

  Mosaic (Rectangle in, size_t size);
  virtual ~Mosaic ();

  virtual void pull (Seconds time, MonoImage & image);
};

class Yuv420p8Mosaic : public Pulled<Yuv420p8Image>
{
  std::vector<Shared<Yuv420p8Image, size_t> *> m_images;
  Yuv420p8Image m_block;

public:

  const Rectangle shape_in;
  const Rectangle shape_out;
  size_t size () { return m_images.size(); }

  Pushed<Yuv420p8Image> & in (size_t i);

  Yuv420p8Mosaic (Rectangle in, size_t size);
  virtual ~Yuv420p8Mosaic ();

  virtual void pull (Seconds time, Yuv420p8Image & image);
};

//----( filtering framework )-------------------------------------------------

template<class In, class Out = In>
class Filter
  : public Pushed<In>,
    public Pulled<Out>
{
protected:

  In m_image;
  Out m_filtered;

public:

  RectangularPort<Pulled<In> > in;
  RectangularPort<Pushed<Out> > out;

  Filter (
      string name,
      Rectangle shape_in,
      Rectangle shape_out)

    : m_image(shape_in.size()),
      m_filtered(shape_out.size()),

      in(name + string(".in"), shape_in),
      out(name + string(".out"), shape_out)
  {}

  Filter (
      string name,
      Rectangle shape)

    : m_image(shape.size()),
      m_filtered(shape.size()),

      in(name + string(".in"), shape),
      out(name + string(".out"), shape)
  {}

  virtual ~Filter () {}

  virtual void filter (Seconds time, const In & image, Out & filtered) = 0;

  virtual void push (Seconds time, const In & image)
  {
    ASSERT_SIZE(image, in.size() * In::num_channels);

    filter(time, image, m_filtered);
    out.push(time, m_filtered);
  }
  virtual void pull (Seconds time, Out & filtered)
  {
    ASSERT_SIZE(filtered, out.size() * Out::num_channels);

    in.pull(time, m_image);
    filter(time, m_image, filtered);
  }
};

//----( filters )-------------------------------------------------------------

class WireFilter : public Filter<MonoImage>
{
public:

  WireFilter (Rectangle shape)
    : Filter<MonoImage>(string("WireFilter"), shape)
  {}
  virtual ~WireFilter () {}

  virtual void push (Seconds time, const MonoImage & image)
  {
    out.push(time, image);
  }
  virtual void pull (Seconds time, MonoImage & wired)
  {
    in.pull(time, wired);
  }
  virtual void filter (Seconds time, const MonoImage & image, MonoImage & wired)
  {
    wired = image;
  }
};

class Transpose : public Filter<MonoImage>
{
public:

  Transpose (Rectangle shape_in)
    : Filter<MonoImage>(
        string("Transpose"),
        shape_in,
        shape_in.transposed())
  {}
  virtual ~Transpose () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & transposed);
};

//----( projection )----

class ProjectAxes
  : public Rectangle,
    public Pushed<MonoImage>
{
  Vector<float> m_image_x;
  Vector<float> m_image_y;

public:

  SizedPort<Pushed<Vector<float> > > x_out;
  SizedPort<Pushed<Vector<float> > > y_out;

  ProjectAxes (Rectangle shape);
  virtual ~ProjectAxes () { PRINT2(sum(m_image_x), sum(m_image_y)); }

  virtual void push (Seconds time, const MonoImage & image);
};

//----( sliders )----

class SliderColumns
  : public Rectangle,
    public Pushed<MonoImage>
{
  Spline m_spline;

  Vector<float> m_y;
  Vector<float> m_sum_m;
  Vector<float> m_sum_my;
  Sliders m_sliders;

public:

  Port<Pushed<Sliders> > out;

  SliderColumns (Rectangle shape_in, size_t size_out);

  virtual void push (Seconds time, const MonoImage & image);
};

//----( resizing )----

class ShrinkBy2 : public Filter<MonoImage>
{
public:

  ShrinkBy2 (Rectangle shape_in)
    : Filter<MonoImage>(string("ShrinkBy2"), shape_in, shape_in.scaled(0.5))
  {}
  virtual ~ShrinkBy2 () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & shrunk);
};

class ShrinkBy4 : public Filter<MonoImage>
{
  MonoImage m_shrunk2;

public:

  ShrinkBy4 (Rectangle shape)
    : Filter<MonoImage>(string("ShrinkBy4"), shape, shape.scaled(0.25)),
      m_shrunk2(shape.size() / 4)
  {}
  virtual ~ShrinkBy4 () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & shrunk);
};

class Shrink : public Filter<MonoImage>
{
  Spline2DSeparable m_spline;

public:

  Shrink (Rectangle shape_in, Rectangle shape_out)
    : Filter<MonoImage>(string("Shrink"), shape_in, shape_out),
      m_spline(shape_in, shape_out)
  {
    ASSERT_LE(shape_out.width(), shape_in.width());
    ASSERT_LE(shape_out.height(), shape_in.height());
  }
  virtual ~Shrink () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & shrunk);
};

class ZoomMono : public Filter<MonoImage>
{
  Spline2DSeparable m_spline;

public:

  ZoomMono (Rectangle shape_in, Rectangle shape_out)
    : Filter<MonoImage>(string("ZoomMono"), shape_in, shape_out),
      m_spline(shape_out, shape_in)
  {
    ASSERT_LE(shape_in.width(), shape_out.width());
    ASSERT_LE(shape_in.height(), shape_out.height());
  }

  virtual ~ZoomMono () {}

  virtual void pull (Seconds time, MonoImage & zoomed);
  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & zoomed);
};

class ZoomRgb : public Filter<RgbImage>
{
  Spline2DSeparable m_spline;

public:

  ZoomRgb (Rectangle shape_in, Rectangle shape_out)
    : Filter<RgbImage>(string("ZoomRgb"), shape_in, shape_out),
      m_spline(shape_out, shape_in)
  {
    ASSERT_LE(shape_in.width(), shape_out.width());
    ASSERT_LE(shape_in.height(), shape_out.height());
  }

  virtual ~ZoomRgb () {}

  virtual void pull (Seconds time, RgbImage & zoomed);
  virtual void filter (Seconds time, const RgbImage & image, RgbImage & zoomed);
};

class ZoomMono8 : public Filter<Mono8Image>
{
public:

  ZoomMono8 (Rectangle shape_in, Rectangle shape_out)
    : Filter<Mono8Image>(string("ZoomMono8"), shape_in, shape_out)
  {
  }

  virtual ~ZoomMono8 () {}

  virtual void filter (
      Seconds time,
      const Mono8Image & image,
      Mono8Image & zoomed);
};

class ZoomRgb8 : public Filter<Rgb8Image>
{
public:

  ZoomRgb8 (Rectangle shape_in, Rectangle shape_out)
    : Filter<Rgb8Image>(string("ZoomRgb8"), shape_in, shape_out)
  {
  }

  virtual ~ZoomRgb8 () {}

  virtual void filter (
      Seconds time,
      const Rgb8Image & image,
      Rgb8Image & zoomed);
};

//----( blurring )----

class SquareBlur : public Filter<MonoImage>
{
  const size_t m_blur_radius;

public:

  SquareBlur (Rectangle shape_in, size_t radius);
  virtual ~SquareBlur () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & blurred);
};

class QuadraticBlur : public Filter<MonoImage>
{
  const size_t m_blur_radius;

public:

  QuadraticBlur (Rectangle shape_in, size_t radius);
  virtual ~QuadraticBlur () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & blurred);
};

class ImageHighpass : public Filter<MonoImage>
{
  const size_t m_blur_radius;

  MonoImage m_temp;

public:

  ImageHighpass (Rectangle shape, size_t radius);
  virtual ~ImageHighpass () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & highpass);
};

//----( masking )----

class DiskMask
  : public Rectangle,
    public Pushed<MonoImage>
{
  MonoImage m_mask;

public:

  RectangularPort<Pushed<MonoImage> > out;

  DiskMask (Rectangle shape);

  virtual void push (Seconds time, const MonoImage & image);
};

//----( feature enhancement )----

class EnhancePoints : public Filter<MonoImage>
{
  const size_t m_blur_radius;
  const float m_timescale;
  Filters::MaxGain m_gain;

public:

  EnhancePoints (
      Rectangle shape_in,
      size_t blur_radius = FINGER_BLUR_RADIUS);
  virtual ~EnhancePoints () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MonoImage & points);
};

class EnhanceFingers : public Filter<MonoImage, MomentImage>
{
  const size_t m_blur_radius;
  const float m_timescale;

  Filters::MaxGain m_tips_gain;
  Filters::MaxGain m_grad_gain;

  MonoImage m_temp;

public:

  EnhanceFingers (
      Rectangle shape,
      size_t blur_radius = FINGER_BLUR_RADIUS);
  virtual ~EnhanceFingers () {}

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      MomentImage & moments);
};

class EnhanceHands : public Filter<MonoImage, HandImage>
{
  const size_t m_blur_radius;
  const size_t m_hand_radius;
  const float m_timescale;

  Filters::MaxGain m_tip_gain;
  Filters::MaxGain m_shaft_gain;
  Filters::MaxGain m_palm_gain;

  MonoImage m_blurred;

public:

  EnhanceHands (
      Rectangle shape_in,
      size_t blur_radius = FINGER_BLUR_RADIUS,
      size_t hand_radius = HAND_BLUR_RADIUS);
  virtual ~EnhanceHands ();

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      HandImage & moments);
};

class HandsToColor : public Filter<HandImage, RgbImage>
{
public:

  HandsToColor (Rectangle shape)
    : Filter<HandImage, RgbImage>("HandsToColor", shape)
  {}
  virtual ~HandsToColor () {}

  virtual void push (Seconds time, const HandImage & hands);
  virtual void pull (Seconds time, RgbImage & rgb);
  virtual void filter (Seconds time, const HandImage & hands, RgbImage & rgb);
};

//----( moments )----

class ExtractMoments
  : public Rectangle,
    public Pushed<MonoImage>
{
  Image::Peak m_moments;
  Filters::MaxGain m_gain;

public:

  Port<Pushed<Image::Peak> > out;

  ExtractMoments (
      Rectangle shape,
      float gain_timescale = DEFAULT_GAIN_TIMESCALE_SEC)
    : Rectangle(shape),
      m_moments(),
      m_gain(gain_timescale),

      out("ExtractMoments.out")
  {}
  virtual ~ExtractMoments () { PRINT(ExtractMoments::m_gain); }

  virtual void push (Seconds time, const MonoImage & image);
};

//----( reassignment )----

class ReassignAccum : public Filter<MonoImage>
{
  const float m_decay;
  Vector<float> m_accum;

public:

  ReassignAccum (Rectangle shape, float timescale);
  virtual ~ReassignAccum () {}

  virtual void filter (Seconds time, const MonoImage & dmass, MonoImage & reas);
};

class AttractRepelAccum : public Filter<MonoImage>
{
  const float m_decay;
  Vector<float> m_accum;
  Vector<float> m_dx;
  Vector<float> m_dy;
  Vector<float> m_blur;

public:

  AttractRepelAccum (Rectangle shape, float timescale);
  virtual ~AttractRepelAccum () {}

  virtual void filter (Seconds time, const MonoImage & dmass, MonoImage & reas);
};

//----( optical flow )----

// TODO make optical flows : public Filter<YyuvImage, FlowImage>

class OpticalFlow : public Filter<MonoImage, FlowImage>
{
  const size_t m_highpass_radius;

  MonoImage m_highpass0;
  MonoImage m_highpass1;
  MonoImage m_temp1;
  MonoImage m_temp2;

public:

  OpticalFlow (
      Rectangle shape_in,
      size_t highpass_radius = OPTICAL_FLOW_HIGHPASS_RADIUS);
  virtual ~OpticalFlow () {}

  virtual void filter (Seconds time, const MonoImage & image, FlowImage & flow);
};

class KrigOpticalFlow : public Filter<MonoImage, FlowImage>
{
  const size_t m_highpass_radius;
  const float m_spacescale;
  const float m_prior;

  MonoImage m_highpass0;
  MonoImage m_highpass1;
  FlowInfoImage m_flow_full;
  FlowInfoImage m_flow_half;

  Seconds m_time;

public:

  KrigOpticalFlow (
      Rectangle shape_in,
      float spacescale = OPTICAL_FLOW_SPACESCALE,
      size_t highpass_radius = OPTICAL_FLOW_HIGHPASS_RADIUS,
      float prior = OPTICAL_FLOW_PRIOR_PER_PIX);
  virtual ~KrigOpticalFlow ();

  virtual void filter (Seconds time, const MonoImage & image, FlowImage & flow);
};

class GlovesFlow : public Filter<MonoImage, GlovesImage>
{
  const float m_lowpass_radius;
  const float m_spacescale;
  const float m_prior;

  MonoImage m_lowpass0;
  MonoImage m_lowpass1;
  FlowInfoImage m_flow_full;
  FlowInfoImage m_flow_half;

  Seconds m_time;

public:

  GlovesFlow (
      Rectangle shape_in,
      float spacescale = OPTICAL_FLOW_SPACESCALE,
      float lowpass_radius = OPTICAL_FLOW_LOWPASS_RADIUS,
      float prior = OPTICAL_FLOW_PRIOR_PER_PIX);
  virtual ~GlovesFlow ();

  virtual void filter (
      Seconds time,
      const MonoImage & image,
      GlovesImage & yuv);
};

class FilterOpticalFlow : public Filter<MonoImage, FlowImage>
{
  const size_t m_highpass_radius;
  const float m_process_noise;
  const float m_prior;

  MonoImage m_highpass0;
  MonoImage m_highpass1;
  FlowInfoImage m_flow_full;
  FlowInfoImage m_flow_half;
  FlowInfoImage m_flow_old;

  Seconds m_time;

public:

  FilterOpticalFlow (
      Rectangle shape_in,
      float process_noise = OPTICAL_FLOW_PROCESS_NOISE,
      size_t highpass_radius = OPTICAL_FLOW_HIGHPASS_RADIUS,
      float prior = OPTICAL_FLOW_PRIOR_PER_PIX);
  virtual ~FilterOpticalFlow ();

  virtual void filter (Seconds time, const MonoImage & image, FlowImage & flow);
};

class FlowToColor : public Filter<FlowImage, RgbImage>
{
  Seconds m_time;
  float m_max_flow;

public:

  FlowToColor (Rectangle shape);
  virtual ~FlowToColor () { PRINT(m_max_flow); }

  virtual void filter (Seconds time, const FlowImage & flow, RgbImage & rgb);
};

class GlovesToColor : public Filter<GlovesImage, RgbImage>
{
  FlowToColor m_flow_to_color;
  FlowImage m_flow_half;
  RgbImage m_rgb_half;

public:

  GlovesToColor (Rectangle shape);
  virtual ~GlovesToColor () {}

  virtual void filter (
      Seconds time,
      const GlovesImage & gloves,
      RgbImage & rgb);
};

//----( change detector )----

class ChangeFilter
  : public Rectangle,
    public Pushed<YyuvImage>
{
  const float m_uv_scale;

  const float m_change_timescale;
  const float m_lowpass_timescale;

  Seconds m_time;

  Vector<float> m_mean;
  Vector<float> m_variance;
  Vector<float> m_lowpass;

public:

  RectangularPort<Pushed<MonoImage> > out;

  ChangeFilter (Rectangle shape);
  virtual ~ChangeFilter ();

  virtual void push (Seconds time, const YyuvImage & image);
};

//----( detectors )----

class PeakDetector
  : public Rectangle,
    public Pushed<MonoImage>
{
  const size_t m_peak_capacity;
  const float m_power;
  const float m_min_ratio;

  Filters::MaxGain m_gain;

  Image::Peaks m_peaks;

public:

  Port<Pushed<Image::Peaks> > out;

  PeakDetector (
      Rectangle shape,
      size_t peak_capacity,
      float power          = PEAK_DETECTOR_POWER,
      float init_gain      = PEAK_DETECTOR_INIT_GAIN,
      float min_ratio      = PEAK_DETECTOR_MIN_RATIO,
      float gain_timescale = DEFAULT_GAIN_TIMESCALE_SEC);
  virtual ~PeakDetector ();

  virtual void push (Seconds time, const MonoImage & image);
};

class PeakTransform : public Pushed<Image::Peaks>
{
  Image::Transform & m_transform;

public:

  Port<Pushed<Image::Peaks> > out;

  PeakTransform (Image::Transform & transform)
    : m_transform(transform),
      out("PeakTransform.out")
  {}

  virtual void push (Seconds time, const Image::Peaks & const_peaks);
};

class MomentsToFinger
  : public Pushed<Image::Peak>,
    public Pulled<Gestures::Finger>
{
  Seconds m_time0;
  Seconds m_time1;

  Gestures::Finger m_finger0;
  Gestures::Finger m_finger1;

  Mutex m_mutex;

public:

  Port<Pulled<float> > impact_in;

  MomentsToFinger ();
  virtual ~MomentsToFinger ();

  virtual void push (Seconds time, const Image::Peak & moment);
  virtual void pull (Seconds time, Gestures::Finger & finger);
};

//----( pulled )--------------------------------------------------------------

class CombineRgb
  : public Rectangle,
    public Pulled<RgbImage>
{
public:

  RectangularPort<Pulled<MonoImage> > red_in;
  RectangularPort<Pulled<MonoImage> > green_in;
  RectangularPort<Pulled<MonoImage> > blue_in;

  CombineRgb (Rectangle shape);
  virtual ~CombineRgb () {}

  virtual void pull (Seconds time, RgbImage & rgb);
};

class NormalizeTo01
  : public Rectangle,
    public Pulled<MonoImage>
{
  Seconds m_time;

  float m_LB;
  float m_UB;

public:

  RectangularPort<Pulled<MonoImage> > in;

  NormalizeTo01 (Rectangle shape);
  virtual ~NormalizeTo01 () {}

  virtual void pull (Seconds time, MonoImage & scaled);
};

class LiftAxes
  : public Rectangle,
    public Pulled<MonoImage>
{
  Vector<float> m_image_x;
  Vector<float> m_image_y;

public:

  SizedPort<Pulled<Vector<float> > > x_in;
  SizedPort<Pulled<Vector<float> > > y_in;

  LiftAxes (Rectangle shape);
  virtual ~LiftAxes () { PRINT2(sum(m_image_x), sum(m_image_y)); }

  virtual void pull (Seconds time, MonoImage & image_yx);
};

class History
  : public Rectangle,
    public Pushed<Vector<float> >,
    public Pushed<MonoImage>,
    public Pulled<MonoImage>
{
  Psycho::History m_history;
  Mutex m_mutex;

public:

  History (size_t size, size_t duration)
    : Rectangle(duration, size),
      m_history(size, duration)
  {}

  virtual void push (Seconds, const Vector<float> & present)
  {
    m_mutex.lock();
    m_history.add(present);
    m_mutex.unlock();
  }

  virtual void push (Seconds, const MonoImage & present)
  {
    m_mutex.lock();
    m_history.add(present);
    m_mutex.unlock();
  }

  virtual void pull (Seconds, MonoImage & history)
  {
    m_mutex.lock();
    m_history.get(history);
    m_mutex.unlock();
  }
};

class RgbHistory
  : public Rectangle,
    public Pushed<RgbImage>,
    public Pulled<RgbImage>
{
  Psycho::History m_history;
  Vector<float> m_transposed;
  Mutex m_mutex;

public:

  RgbHistory (size_t size, size_t duration)
    : Rectangle(duration, size),
      m_history(3 * size, duration),
      m_transposed(3 * size * duration)
  {}

  virtual void push (Seconds, const RgbImage & present)
  {
    m_mutex.lock();
    m_history.add(present);
    m_mutex.unlock();
  }

  virtual void pull (Seconds, RgbImage & history);
};

class Oscilloscope
  : public Rectangle,
    public Pushed<complex>,
    public Pushed<std::vector<complex> >,
    public Pulled<MonoImage>
{
  const float m_radius;
  const float m_timescale;
  MonoImage m_image;
  Seconds m_time;
  Mutex m_mutex;

public:

  Oscilloscope (
      Rectangle shape,
      float radius = 1.0f,
      float timescale_sec = 0.5f);
  virtual ~Oscilloscope () {}

  virtual void push (Seconds time, const complex & signal);
  virtual void push (Seconds time, const std::vector<complex> & signals);
  virtual void pull (Seconds time, MonoImage & image);
};

//----( features )----

class ImpactDistributor : public Pulled<BoundedMap<Id, Gestures::Finger> >
{
public:

  Port<Pulled<float> > impact_in;
  Port<Pulled<BoundedMap<Id, Gestures::Finger> > > fingers_in;

  ImpactDistributor ()
    : impact_in("ImpactDistributor.impact_in"),
      fingers_in("ImpactDistributor.fingers_in")
  {}
  virtual ~ImpactDistributor () {}

  virtual void pull (Seconds time, BoundedMap<Id, Gestures::Finger> & fingers);
};

// converts (x,y,z) to (r,phase,z)
class PixToPolar
  : public Rectangle,
    public Pulled<BoundedMap<Id, Gestures::Finger> >
{
  const bool m_mask_to_disk;

public:

  Port<Pulled<BoundedMap<Id, Gestures::Finger> > > in;

  PixToPolar (Rectangle shape, bool mask_to_disk = true)
    : Rectangle(shape),
      m_mask_to_disk(mask_to_disk),
      in("PixToPolar.in")
  {}
  virtual ~PixToPolar () {}

  virtual void pull (Seconds time, BoundedMap<Id, Gestures::Finger> & fingers);
};

class FingersToChords
  : public Pulled<BoundedMap<Id, Gestures::Chord> >
{
  BoundedMap<Id, Gestures::Finger> m_fingers;

public:

  SizedPort<Pulled<BoundedMap<Id, Gestures::Finger> > > in;

  FingersToChords (size_t finger_capacity)
    : m_fingers(finger_capacity),
      in("FingersToChords.in", finger_capacity)
  {}
  virtual ~FingersToChords () {}

  virtual void pull (Seconds time, BoundedMap<Id, Gestures::Chord> & chords)
  {
    in.pull(time, m_fingers);
    Gestures::fingers_to_chords(m_fingers, chords);
  }
};

class RelativizeFingers
  : public Pulled<BoundedMap<Id, Gestures::Finger> >
{
  BoundedMap<Id, Gestures::Finger> m_hands;

public:

  SizedPort<Pulled<BoundedMap<Id, Gestures::Finger> > > hands_in;
  SizedPort<Pulled<BoundedMap<Id, Gestures::Finger> > > fingers_in;

  RelativizeFingers (size_t hand_capacity);
  virtual ~RelativizeFingers () {}

  virtual void pull (Seconds time, BoundedMap<Id, Gestures::Finger> & fingers);
};

//----( threads )-------------------------------------------------------------

class TimedVideo
  : public Rectangle,
    public Thread
{
protected:

  const float m_timestep;
  Seconds m_time;
  MonoImage m_image;

public:

  RectangularPort<Pushed<MonoImage> > out;

  TimedVideo (Rectangle shape, float framerate = DEFAULT_VIDEO_FRAMERATE);
  virtual ~TimedVideo () {}

protected:

  virtual void run ();
};

class Random01Video : public TimedVideo
{
public:

  Random01Video (Rectangle shape, float framerate = DEFAULT_VIDEO_FRAMERATE)
    : TimedVideo(shape, framerate)
  {}
  virtual ~Random01Video () {}
  virtual void step ();
};

class RandomStdVideo : public TimedVideo
{
public:

  RandomStdVideo (Rectangle shape, float framerate = DEFAULT_VIDEO_FRAMERATE)
    : TimedVideo(shape, framerate)
  {}
  virtual ~RandomStdVideo () {}
  virtual void step ();
};

} // namespace Streaming

#endif // KAZOO_STREAMING_VIDEO_H

