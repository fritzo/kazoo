#ifndef KAZOO_CAMERA_H
#define KAZOO_CAMERA_H

/** Video capture using V4L2.

  (R1) The Video4Linux2 API: an introduction -corbet
    http://lwn.net/Articles/203924/
  (R2) V4L2 API - http://v4l2spec.bytesex.org/spec/book1.htm
  (R3) example usage of v4l2 with PS3 Eye
    wget http://kaswy.free.fr/sites/default/files/download/ps3eye/0.5/gspca-ps3eyeMT-0.5.tar.gz
    gvim gspca-ps3eyeMT/v4l2-apps/test/capture-example.c
  (R4) Tutorial explores V4L2 buffer management
    http://www.linuxfordevices.com/c/a/News/Tutorial-explores-V4L2-buffer-management/
  (R5) Video Grabber example using libv4l (a program in C)
    http://docs.blackfin.uclinux.org/kernel/generated/media/apd.html
  (R6) Kwasy's driver, a blogpost "PS3eye on Linux: Yes we can!!!"
    http://kaswy.free.fr/?q=node/42
  (R7) "ps3 eye driver patch"
    http://bear24rw.blogspot.com/2009/11/ps3-eye-driver-patch.html
  (R8) Kwasy's ps3 eye driver patch in ubuntu
    http://kaswy.free.fr/?q=en/node/53

  (N1) PS3 Eye driver must reload gspca_ov534 module to set video mode (R6).
    Available Modes:
      00: 640x480@15
      01: 640x480@30
      02: 640x480@40
      03: 640x480@50
      04: 640x480@60
      10: 320x240@30
      11: 320x240@40
      12: 320x240@50
      13: 320x240@60
      14: 320x240@75
      15: 320x240@100
      16: 320x240@125
    To Load:
      modprobe gspca_ov534 videomode=04
    To Reload:
      modprobe -r gspca_ov534; modprobe gspca_ov534 videomode=04
*/

#include "common.h"
#include "vectors.h"
#include "cyclic_time.h"
#include "threads.h"
#include "config.h"

#define DEFAULT_CAMERA_WIDTH_PIX        (320)
#define DEFAULT_CAMERA_HEIGHT_PIX       (240)
#define DEFAULT_CAMERA_WIDTH_FINGERS    (100.0)

//----( bounding boxes )------------------------------------------------------

struct BoundingBox
{
  size_t x0, x1, y0, y1;

  BoundingBox () {}
  BoundingBox (
      size_t a_x0,
      size_t a_x1,
      size_t a_y0,
      size_t a_y1)

    : x0(a_x0),
      x1(a_x1),
      y0(a_y0),
      y1(a_y1)
  {
    ASSERT_LE(x0,x1);
    ASSERT_LE(y0,y1);
  }

  size_t width () const { return x1 - x0; }
  size_t height () const { return y1 - y0; }
  size_t size () const { return width() * height(); }
  float radius () const { return sqrtf(sqr(width()) + sqr(height())); }

  BoundingBox round_to (size_t multiple) const;
};

inline ostream & operator<< (ostream & o, const BoundingBox & b)
{
  return o << '[' << b.x0 << ',' << b.x1 << ')' << " x "
           << '[' << b.y0 << ',' << b.y1 << ')';
}

//----( abstract video sources )----------------------------------------------

class VideoSource
  : public Aligned<VideoSource>,
    public Rectangle
{
protected:

  size_t m_framerate;

public:

  VideoSource (
      size_t width = 0,
      size_t height = 0,
      size_t framerate = 0)
    : Rectangle(width, height),
      m_framerate(framerate)
  {}
  virtual ~VideoSource () {}

  size_t framerate () const { return m_framerate; }

  virtual Seconds capture (Vector<float> & frame_out) const = 0;
  Seconds capture_transpose (Vector<float> & frame_out, Vector<float> & temp);
};

//----( simple frame-grabbing camera )----------------------------------------

class CameraRegion;

class Camera : public VideoSource
{
  int m_camera_fd;
  string m_name;

  size_t m_buffer_size;
  mutable uint8_t * restrict m_buffer_data;

  mutable uint64_t m_frame_count;

  static int s_brightness;
  static int s_contrast;
  static int s_gain;
  static int s_exposure;
  static bool s_auto_gain;
  static bool s_auto_white_balance;
  static bool s_auto_exposure;
  static int s_sharpness;

public:

  Camera (
      size_t default_width      = DEFAULT_CAMERA_WIDTH_PIX,
      size_t default_height     = DEFAULT_CAMERA_HEIGHT_PIX,
      size_t default_framerate  = DEFAULT_VIDEO_FRAMERATE);
  virtual ~Camera ();

  // does anyone really use this ?
  //float scale () const { return 255 * 2; }

  void read_buffer () const;

  // WARNING: frame returned is transposed
  virtual Seconds capture (Vector<float> & frame_out) const;
  Seconds capture (Vector<uint8_t> & frame_out) const;
  Seconds capture (Vector<float> & frame_out, BoundingBox box) const;
  Seconds capture_yuyv (Vector<float> & yuyv_out) const;
  Seconds capture_yuyv (
      Vector<float> & yy_out,
      Vector<float> & u_out,
      Vector<float> & v_out) const;
  Seconds capture_fifth (Vector<float> & frame_out) const;
  Seconds capture_fifth (Vector<uint8_t> & frame_out) const;
  Seconds capture_yuv420p8 (
      Vector<uint8_t> & y_out,
      Vector<uint8_t> & u_out,
      Vector<uint8_t> & v_out) const;
  Seconds capture_fifth_yuv420p8 (
      Vector<uint8_t> & y_out,
      Vector<uint8_t> & u_out,
      Vector<uint8_t> & v_out) const;
  Seconds accumulate (Vector<float> & frame_accum) const;
  Seconds accumulate (Vector<float> & frame_accum, BoundingBox box) const;

  BoundingBox full_box () { return BoundingBox(0, m_width, 0, m_height); }
  BoundingBox autocrop (float threshold = 0.05, size_t num_samples = 16);

  // WARNING autogain can reach lower gain levels than any manual gain
  //   (this may be a but in the ov534 driver)
  static void set_brightness (int value = 0);
  static void set_contrast (int value = 32);
  static void set_gain (int value = 20);
  static void set_exposure (int value = 120);
  static void set_auto_gain (bool value = true);
  static void set_auto_white_balance (bool value = true);
  static void set_auto_exposure (bool value = true);
  static void set_sharpness (int value = 0);

  static void set_config (const char * filename);

  friend class CameraRegion;
};

//----( camera region )-------------------------------------------------------

/** Camera regions.
  A camera region is an image mask together with the mask's bounding box and
  the statically estimated image background inside the mask region.
*/

class CameraRegion : public VideoSource
{
public:

  class Region
  {
  public:

    virtual ~Region () {}

    virtual void init (
        Camera & camera,
        BoundingBox & bbox,
        float * restrict & restrict mask,
        float * restrict & restrict background) = 0;
  };

  class FloodFill : public Region
  {
    ConfigParser m_config;

    const size_t m_num_bg_frames;
    const float m_lowpass_units;
    const float m_floodfill_threshold;
    const float m_mask_box_threshold;
    const size_t m_dilate_radius;
    const float m_dilate_sharpness;

  public:

    FloodFill (const char * config_filename = "config/default.floodfill.conf");
    virtual ~FloodFill () {}

    virtual void init (
        Camera & camera,
        BoundingBox & bbox,
        float * restrict & restrict mask,
        float * restrict & restrict background);
  };

  class Disk : public Region
  {
    const size_t m_num_bg_frames;
    const float m_softness;

  public:

    Disk (
        size_t num_bg_frames = 32,
        float softness = 8);
    virtual ~Disk () {}

    virtual void init (
        Camera & camera,
        BoundingBox & bbox,
        float * restrict & restrict mask,
        float * restrict & restrict background);
  };

private:

  Camera m_camera;
  BoundingBox m_bbox;
  float * restrict m_mask;
  float * restrict m_background;
  mutable Seconds m_time;

public:

  CameraRegion (
      Region * region = NULL,
      size_t default_width  = DEFAULT_CAMERA_WIDTH_PIX,
      size_t default_height = DEFAULT_CAMERA_HEIGHT_PIX);
  virtual ~CameraRegion ();

  const BoundingBox & bbox () const { return m_bbox; }

  // WARNING: image returned is transposed
  const float * mask () const { return m_mask; }
  const float * background () const { return m_background; }
  void interior (
      size_t padding,
      Vector<float> & frame,
      Vector<float> & temp) const;

  Seconds capture_raw (Vector<float> & frame) const
  {
    return m_camera.capture(frame);
  }

  // WARNING: frame returned is transposed
  Seconds capture_crop (Vector<float> & frame) const;
  Seconds capture_crop_mask (Vector<float> & frame) const;
  Seconds capture_crop_mask_sub (Vector<float> & frame) const;
  Seconds capture_crop_mask_ceil (Vector<float> & frame) const;

  virtual Seconds capture (Vector<float> & frame) const
  {
    return capture_crop_mask_sub(frame);
  }
};

class CameraDisk : public CameraRegion
{
public:
  CameraDisk () : CameraRegion(new CameraRegion::Disk()) {}
  virtual ~CameraDisk () {}
};

//----( color camera )--------------------------------------------------------

class ColorCamera : public VideoSource
{
  Camera m_camera;

public:

  ColorCamera (
      size_t default_width      = DEFAULT_CAMERA_WIDTH_PIX,
      size_t default_height     = DEFAULT_CAMERA_HEIGHT_PIX,
      size_t default_framerate  = DEFAULT_VIDEO_FRAMERATE)
    : VideoSource(),
      m_camera(default_width, default_height)
  {
    m_width = 2 * m_camera.width();
    m_height = m_camera.height();
    m_framerate = m_camera.framerate();
  }
  virtual ~ColorCamera () {}

  virtual Seconds capture (Vector<float> & frame) const
  {
    return m_camera.capture_yuyv(frame);
  }

  Seconds capture (
      Vector<float> & yy,
      Vector<float> & u,
      Vector<float> & v) const
  {
    return m_camera.capture_yuyv(yy, u, v);
  }
};

//----( video stream operations )---------------------------------------------

class ShrinkToHalf : public VideoSource
{
  VideoSource * const m_video;
  mutable Vector<float> m_frame;

public:

  ShrinkToHalf (VideoSource * video)
    : VideoSource(
        video->width() / 2,
        video->height() / 2,
        video->framerate()),

      m_video(video),
      m_frame(video->size())
  {}
  virtual ~ShrinkToHalf () { delete m_video; }

  virtual Seconds capture (Vector<float> & frame_out) const;
};

//----( video threads )-------------------------------------------------------

class CameraThread : private Thread
{
  VideoSource & m_video;
  Vector<float> m_frame;
  bool m_running;

public:

  CameraThread (VideoSource & video);

  bool running () const { return m_running; }

  void start ();
  void stop ();

protected:

  virtual void process (Seconds time, Vector<float> & frame) = 0;

private:

  virtual void run ();
};

#endif // KAZOO_CAMERA_H
