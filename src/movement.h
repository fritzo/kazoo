
#ifndef KAZOO_MOVEMENT_H
#define KAZOO_MOVEMENT_H

#include "common.h"
#include "streaming_video.h"
#include "streaming_camera.h"
#include "streaming_devices.h"

namespace Streaming
{

class ChangeThread
{
  EyeThread m_eye;
  ChangeFilter m_change;

public:

  RectangularPort<Pushed<MonoImage> > & out;

  ChangeThread ()
    : m_eye(),
      m_change(m_eye.out),

      out(m_change.out)
  {
    m_eye.out - m_change;
  }
};

} // namespace Streaming

#endif // KAZOO_MOVEMENT_H

