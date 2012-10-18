#ifndef KAZOO_GLOVES_H
#define KAZOO_GLOVES_H

#include "common.h"
#include "streaming_video.h"
#include "filters.h"
#include "config.h"

namespace Streaming
{

class GlovesFilter
  : public Rectangle,
    public Pushed<Mono8Image>
{
  ConfigParser m_config;

  const float m_lowpass_radius_px;
  const float m_krig_radius_px;
  const float m_speed_rad_per_sec;
  const float m_image_snr;

  MonoImage m_lowpass0;
  MonoImage m_lowpass1;
  FlowInfoImage m_info_full;
  FlowInfoImage m_info_half;
  FlowInfoImage m_info_quart;
  FlowImage m_flow_quart;
  Gloves8Image m_gloves;

  Seconds m_time;

  bool m_initialized;

  Filters::DebugStats<float> m_flow_stats;

public:

  RectangularPort<Pushed<Gloves8Image> > out;

  GlovesFilter (
      Rectangle shape_in,
      const char * config_filename = "config/default.gloves.conf");
  virtual ~GlovesFilter ();

  void reset () { m_initialized = false; }

  virtual void push (Seconds time, const Mono8Image & image);
};

class Gloves8ToColor
  : public Filter<Gloves8Image, Rgb8Image>
{
public:

  Gloves8ToColor (Rectangle shape);
  virtual ~Gloves8ToColor () {}

  virtual void filter (
      Seconds time,
      const Gloves8Image & gloves,
      Rgb8Image & rgb);
};

} // namespace Streaming

#endif // KAZOO_GLOVES_H
