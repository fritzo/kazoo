
#ifndef KAZOO_STREAMING_CAMERA_H
#define KAZOO_STREAMING_CAMERA_H

#include "common.h"
#include "streaming.h"
#include "streaming_video.h"
#include "camera.h"

namespace Streaming
{

class CameraThread : public Thread
{
  Camera m_camera;
  MonoImage m_image;

public:

  RectangularPort<Pushed<MonoImage> > out;

  CameraThread ()
    : m_camera(),
      m_image(m_camera.size()),
      out("CameraThread.out", m_camera.transposed())
  {}
  virtual ~CameraThread () {}

protected:

  virtual void step ();
};

class RegionThread : public Thread
{
protected:

  CameraRegion m_camera;
  MonoImage m_image;

public:

  RectangularPort<Pushed<MonoImage> > out;

  RegionThread (CameraRegion::Region * region = NULL)
    : m_camera(region),
      m_image(m_camera.size()),
      out("RegionThread.out", m_camera.transposed())
  {}
  virtual ~RegionThread () {}

  const float * mask () const { return m_camera.mask(); }
  const float * background () const { return m_camera.background(); }

protected:

  virtual void step ();
};

class RegionCropThread : public RegionThread
{
public:

  RegionCropThread (CameraRegion::Region * r = NULL) : RegionThread(r) {}
  virtual ~RegionCropThread () {}

protected:

  virtual void step ();
};

class RegionMaskThread : public RegionThread
{
public:

  RegionMaskThread (CameraRegion::Region * r = NULL) : RegionThread(r) {}
  virtual ~RegionMaskThread () {}

protected:

  virtual void step ();
};

class RegionMaskSubThread : public RegionThread
{
public:

  RegionMaskSubThread (CameraRegion::Region * r = NULL) : RegionThread(r) {}
  virtual ~RegionMaskSubThread () {}

protected:

  virtual void step ();
};

class RegionMaskCeilThread : public RegionThread
{
public:

  RegionMaskCeilThread (CameraRegion::Region * r = NULL) : RegionThread(r) {}
  virtual ~RegionMaskCeilThread () {}

protected:

  virtual void step ();
};

class DiskThread : public RegionThread
{
public:

  DiskThread () : RegionThread(new CameraRegion::Disk()) {}
  virtual ~DiskThread () {}
};

class EyeThread : public Thread
{
  Camera m_camera;
  YyuvImage m_image;

public:

  RectangularPort<Pushed<YyuvImage> > out;

  EyeThread ()
    : m_image(m_camera.size()),
      out("EyeThread.out", m_camera.transposed())
  {}
  virtual ~EyeThread () {}

protected:

  virtual void step ();
};

class FifthCameraThread : public Thread
{
  Camera m_camera;
  MonoImage m_image;

public:

  RectangularPort<Pushed<MonoImage> > out;

  FifthCameraThread ()
    : m_camera(),
      m_image(m_camera.size() / sqr(5)),
      out("FifthCameraThread.out", m_camera.transposed().scaled(0.2f))
  {}
  virtual ~FifthCameraThread () {}

protected:

  virtual void step ();
};

class Mono8CameraThread : public Thread
{
  Camera m_camera;
  Mono8Image m_image;

public:

  RectangularPort<Pushed<Mono8Image> > out;

  Mono8CameraThread ()
    : m_camera(),
      m_image(m_camera.size()),
      out("Mono8CameraThread.out", m_camera.transposed())
  {}
  virtual ~Mono8CameraThread () {}

protected:

  virtual void step ();
};

class FifthMono8CameraThread : public Thread
{
  Camera m_camera;
  Mono8Image m_image;

public:

  RectangularPort<Pushed<Mono8Image> > out;

  FifthMono8CameraThread ()
    : m_camera(),
      m_image(m_camera.size() / sqr(5)),
      out("FifthMono8CameraThread.out", m_camera.transposed().scaled(0.2f))
  {}
  virtual ~FifthMono8CameraThread () {}

protected:

  virtual void step ();
};

class Yuv420p8CameraThread : public Thread
{
  Camera m_camera;
  Yuv420p8Image m_image;

public:

  RectangularPort<Pushed<Yuv420p8Image> > out;

  Yuv420p8CameraThread ()
    : m_camera(),
      m_image(m_camera.size()),
      out("Yuv420p8CameraThread.out", m_camera.transposed())
  {}
  virtual ~Yuv420p8CameraThread () {}

protected:

  virtual void step ();
};

class FifthYuv420p8CameraThread : public Thread
{
  Camera m_camera;
  Yuv420p8Image m_image;

public:

  RectangularPort<Pushed<Yuv420p8Image> > out;

  FifthYuv420p8CameraThread ()
    : m_camera(),
      m_image(m_camera.size() / sqr(5)),
      out("FifthYuv420p8CameraThread.out", m_camera.transposed().scaled(0.2f))
  {}
  virtual ~FifthYuv420p8CameraThread () {}

protected:

  virtual void step ();
};

} // namespace Streaming

#endif // KAZOO_STREAMING_CAMERA_H

