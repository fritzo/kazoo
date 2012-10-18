#ifndef KAZOO_STREAMING_CLOUDS_H
#define KAZOO_STREAMING_CLOUDS_H

#include "common.h"
#include "clouds.h"
#include "voice.h"
#include "streaming.h"
#include "events.h"

namespace Streaming
{

//----( distance logger )-----------------------------------------------------

class DistanceLogger : public Pushed<Cloud::Point>
{
  Cloud::Point m_prev;
  bool m_continuing;

  double m_total_squared_distance;
  size_t m_num_frames;
  bool m_verbose;

public:

  DistanceLogger (size_t size, bool verbose = false);
  virtual ~DistanceLogger ();

  double get_rms_distance ();

  void add (const Cloud::Point & point);
  void add (const Cloud::Point & point,
            const Cloud::Point & point2);
  virtual void push (Seconds time, const Cloud::Point & point) { add(point); }
  void done () { m_continuing = false; }
};

//----( denoiser )------------------------------------------------------------

class CloudDenoiser : public Pushed<Cloud::Point>
{
  const Cloud::PointSet & m_points;
  const size_t m_iters;

  Vector<float> m_likes;
  Cloud::Point m_point;

public:

  SizedPort<Pushed<Cloud::Point> > out;

  CloudDenoiser (const Cloud::PointSet & points, size_t iters = 1);
  virtual ~CloudDenoiser () {}

  virtual void push (Seconds time, const Cloud::Point & point);
};

//----( simulator )-----------------------------------------------------------

class CloudSimulator
  : public TimedThread,
    public EventHandler
{
  const Cloud::PointSet & m_points;
  const Cloud::JointPrior & m_flow;

  size_t m_state;
  VectorXf & m_next;
  Cloud::Point m_point;

  Mutex m_mutex;

public:

  SizedPort<Pushed<Cloud::Point> > out;

  CloudSimulator (
      const Cloud::JointPrior & flow,
      float farmerate = DEFAULT_VIDEO_FRAMERATE);
  virtual ~CloudSimulator ();

  void reset ();

protected:

  virtual void step ();

  virtual void keyboard (const SDL_KeyboardEvent & event);
};

//----( diffuser )------------------------------------------------------------

class CloudDiffuser
  : public TimedThread,
    public EventHandler
{
  const Cloud::PointSet & m_points;
  const Cloud::JointPrior & m_flow;

  VectorXf & m_state;
  VectorXf & m_next;
  Cloud::Point m_point;

  Mutex m_mutex;

public:

  SizedPort<Pushed<Cloud::Point> > out;

  CloudDiffuser (
      const Cloud::JointPrior & flow,
      float farmerate = DEFAULT_VIDEO_FRAMERATE);
  virtual ~CloudDiffuser ();

  void reset ();

protected:

  virtual void step ();

  virtual void keyboard (const SDL_KeyboardEvent & event);
};

//----( gloves <- gloves )----------------------------------------------------

class GlovesToGloves
  : public Rectangle,
    public Pushed<Gloves8Image>
{
  Cloud::Controller & m_controller;

  Gloves8Image m_image;

public:

  RectangularPort<Pushed<Gloves8Image> > out;

  GlovesToGloves (Cloud::Controller & controller);
  virtual ~GlovesToGloves () {}

  virtual void push (Seconds time, const Gloves8Image & image);
};

//----( gloves <- voice )-----------------------------------------------------

class GlovesToVoice
  : public Rectangle,
    public Pushed<Gloves8Image>
{
  Cloud::Controller & m_controller;
  Voice::FeatureProcessor m_feature_processor;

  Cloud::Point m_voice_features;

public:

  SizedPort<Pushed<Cloud::Point> > out;

  GlovesToVoice (
      Cloud::Controller & controller,
      const char * voice_config = "config/default.voice.conf");
  virtual ~GlovesToVoice () {}

  virtual void push (Seconds time, const Gloves8Image & image);
};

} // namespace Streaming

#endif // KAZOO_STREAMING_CLOUDS_H
