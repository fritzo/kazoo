
#include "streaming_camera.h"

namespace Streaming
{

void CameraThread::step ()
{
  Seconds time = m_camera.capture(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void RegionThread::step ()
{
  Seconds time = m_camera.capture(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void RegionCropThread::step ()
{
  Seconds time = m_camera.capture_crop(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void RegionMaskThread::step ()
{
  Seconds time = m_camera.capture_crop_mask(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void RegionMaskSubThread::step ()
{
  Seconds time = m_camera.capture_crop_mask_sub(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void RegionMaskCeilThread::step ()
{
  Seconds time = m_camera.capture_crop_mask_ceil(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void EyeThread::step ()
{
  Seconds time = m_camera.capture_yuyv(m_image.yy, m_image.u, m_image.v);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void FifthCameraThread::step ()
{
  Seconds time = m_camera.capture_fifth(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void Mono8CameraThread::step ()
{
  Seconds time = m_camera.capture(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void FifthMono8CameraThread::step ()
{
  Seconds time = m_camera.capture_fifth(m_image);
  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void Yuv420p8CameraThread::step ()
{
  Seconds time = m_camera.capture_yuv420p8(
      m_image.y,
      m_image.u,
      m_image.v);

  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

void FifthYuv420p8CameraThread::step ()
{
  Seconds time = m_camera.capture_fifth_yuv420p8(
      m_image.y,
      m_image.u,
      m_image.v);

  out.push(time, m_image);
  PROGRESS_TICKER('|');
}

} // namespace Streaming

