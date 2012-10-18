
#ifndef KAZOO_DIRT_H
#define KAZOO_DIRT_H

#include "common.h"
#include "table.h"
#include "synchrony.h"
#include "streaming_shared.h"

namespace Streaming
{

/** Dirt, an instrument.

  Controller: video shadow + impacts
  Synthesis: oscillator banks for pitch & tempo
*/

//----( control )-------------------------------------------------------------

class DirtPitchControl
  : public Rectangle,
    public Pushed<MonoImage>
{
  Vector<float> m_image_x;
  Vector<float> m_image_y;

public:

  Port<Pulled<float> > impact_in;
  SizedPort<Pushed<Vector<float> > > amplitude_out;
  SizedPort<Pushed<MonoImage> > shadow_monitor;

  DirtPitchControl (Rectangle shape);
  virtual ~DirtPitchControl ();

  virtual void push (Seconds time, const MonoImage & image);
};

class DirtControl
  : public Rectangle,
    public Pushed<MonoImage>
{
  Vector<float> m_old_image_xy;
  Vector<float> m_image_y;
  Vector<float> m_impact_y;

  Vector<float> m_amplitude_y;
  Vector<float> m_amplitude_x;

public:

  Port<Pulled<float> > impact_in;
  SizedPort<Bounced<Vector<float>, Vector<float>, void> > tempo_io;
  SizedPort<Pushed<Vector<float> > > amplitude_out;

  DirtControl (Rectangle shape);
  virtual ~DirtControl ();

  virtual void push (Seconds time, const MonoImage & image);
};

//----( tempo )---------------------------------------------------------------

// TODO refactor DirtTempoSynth as Beater
class DirtTempoSynth
  : public Synchronized::LoopBank,
    public Bounced<Vector<float>, Vector<float>, void>
{
  const size_t m_blur_radius;

  Seconds m_time;

public:

  DirtTempoSynth (
      size_t size,
      float expected_dt,
      bool coalesce = true,
      float blur_factor = 1.0f);
  virtual ~DirtTempoSynth ();

  virtual void bounce (
      Seconds time,
      const Vector<float> & impact_in,
      Vector<float> & amplitude_out);
};

//----( pitch )---------------------------------------------------------------

// TODO replace DirtPitchSynth with Vocoder everywhere
class DirtPitchSynth
  : public Synchronized::PhasorBank,
    public Pulled<StereoAudioFrame>
{
  Vector<float> m_amplitude0;
  Vector<float> m_damplitude;

public:

  Shared<Vector<float>, size_t> amplitude_in;

  SizedPort<Pushed<Vector<float> > > amplitude_monitor;
  SizedPort<Pushed<Vector<float> > > mass_monitor;
  SizedPort<Pushed<Vector<float> > > bend_monitor;

  DirtPitchSynth (size_t size, float freq0 = 0, float freq1 = 0);
  virtual ~DirtPitchSynth ();

  virtual void pull (Seconds time, StereoAudioFrame & sound_out);
};

//----( keyboard )------------------------------------------------------------

class DirtKeys
  : public Rectangle,
    public Pulled<MonoImage>
{

  Vector<float> m_image_x;
  Vector<float> m_image_y;

public:

  SharedLowpass<Vector<float> > bend_in;

  DirtKeys (Rectangle shape_in)
    : Rectangle(shape_in.transposed()),
      m_image_x(shape_in.width()),
      m_image_y(shape_in.height()),
      bend_in(2.0f / DEFAULT_SCREEN_FRAMERATE, shape_in.width())
  {
    m_image_x.set(1.0f);
    m_image_y.set(1.0f);
  }
  virtual ~DirtKeys () {}

  virtual void pull (Seconds time, MonoImage & image_yx);
};

class DirtColorKeys
  : public Rectangle,
    public Pulled<RgbImage>
{
  Vector<float> m_image_x;

public:

  Shared<MonoImage, size_t> shadow_in;
  SharedLowpass<Vector<float> > amp0_in;
  SharedLowpass<Vector<float> > amp1_in;
  SharedLowpass<Vector<float> > mass_in;
  SharedLowpass<Vector<float> > bend_in;

  DirtColorKeys (Rectangle shape_in)
    : Rectangle(shape_in.transposed()),

      m_image_x(shape_in.width()),

      shadow_in(size()),
      amp0_in(2.0f / DEFAULT_SCREEN_FRAMERATE, shape_in.width()),
      amp1_in(2.0f / DEFAULT_SCREEN_FRAMERATE, shape_in.width()),
      mass_in(2.0f / DEFAULT_SCREEN_FRAMERATE, shape_in.width()),
      bend_in(2.0f / DEFAULT_SCREEN_FRAMERATE, shape_in.width())
  {}
  virtual ~DirtColorKeys () {}

  virtual void pull (Seconds time, RgbImage & image);
};

//----( systems )-------------------------------------------------------------

class Dirt
{
protected:

  ShadowTable m_table;
  EnhancePoints m_points;

  SharedMaxLowpass m_impact;

  DirtControl m_control;
  DirtTempoSynth m_tempo;
  DirtPitchSynth m_pitch;

public:

  Dirt (
      bool deaf = false,
      bool coalesce = true,
      float blur_factor = 1.0f,
      float freq0 = 0,
      float freq1 = 0);
};

class ShowDirt : public Dirt
{
protected:

  DirtKeys m_keys;
  ShowMonoZoom m_screen;

public:

  ShowDirt (
      Rectangle screen_shape,
      bool deaf = false,
      bool coalesce = true,
      float blur_factor = 1.0f,
      float freq0 = 0,
      float freq1 = 0);
};

class DirtPitchTest
{
  ShadowTable m_table;
  EnhancePoints m_points;

  Shared<float> m_impact;

  DirtPitchControl m_control;
  DirtPitchSynth m_pitch;

  DirtColorKeys m_keys;
  ShowRgbZoom m_screen;

public:

  DirtPitchTest (
      float freq0 = 0,
      float freq1 = 0,
      bool deaf = false,
      Rectangle screen_shape = Rectangle(0,0));
};

} // namespace Streaming

#endif // KAZOO_DIRT_H

