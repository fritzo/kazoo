
#include "camera.h"
#include "images.h"
#include <set>
#include <algorithm>

#include <unistd.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <fcntl.h>
#include <errno.h>
#include <sys/ioctl.h>
#include <sys/types.h>
#include <sys/mman.h>
#include <sys/poll.h>
#include <linux/videodev2.h>
#include <libv4l2.h>

#define LOG1(message)

//----( bounding boxes )------------------------------------------------------

BoundingBox BoundingBox::round_to (size_t multiple) const
{
  ASSERT_LT(0, multiple);

  size_t w = width() / multiple * multiple;
  size_t h = height() / multiple * multiple;

  size_t dx = (width() - w) / 2;
  size_t dy = (height() - h) / 2;

  BoundingBox result(x0 + dx, x1 - dx,
                     y0 + dy, y1 - dy);

  if (w < result.width()) --result.x1;
  if (h < result.height()) --result.y1;

  ASSERT_EQ(result.width(), w);
  ASSERT_EQ(result.height(), h);
  ASSERT_DIVIDES(multiple, result.width());
  ASSERT_DIVIDES(multiple, result.height());

  return result;
}

//----( abstract video sources )----------------------------------------------

Seconds VideoSource::capture_transpose (
    Vector<float> & frame_out,
    Vector<float> & temp)
{
  ASSERT_SIZE(frame_out, size());
  ASSERT_SIZE(temp, size());

  Seconds time = capture(temp);
  Image::transpose_8(m_height, m_width, temp, frame_out);

  return time;
}

//----( safe low-level wrappers )---------------------------------------------

#define CLEAR(data) memset(&data, 0, sizeof(data))

static int safe_ioctl(
    int fd,
    unsigned long int request,
    void *arg,
    bool exit_on_error = true)
{
  int info;

  do {
    info = v4l2_ioctl(fd, request, arg);
  } while (info == -1 && ((errno == EINTR) || (errno == EAGAIN)));

  if (exit_on_error) {
    ASSERT(info != -1, "ioctl error " << errno << ", " << strerror(errno));
  }

  return info;
}

static void safe_read (int fd, void * data, size_t size)
{
  int info;
  do {
    info = read(fd, data, size);
  } while (info != static_cast<int>(size));
}

//----( parameters )----------------------------------------------------------

/** Pixel formats
  V4L2_PIX_FMT_GREY  = 8-bit grayscale
  V4L2_PIX_FMT_Y16   = 16-bit grayscale
  V4L2_PIX_FMT_YUYV  = 16-bit color
  V4L2_PIX_FMT_RGB24 = 24-bit color
*/
#define PIXEL_FORMAT V4L2_PIX_FMT_YUYV

inline float yuyv_luminance (uint8_t * restrict data, size_t ij)
{
  return static_cast<float>(data[2 * ij]);
}

/** Channel width.  This is for YUYV pixel format.
*/
#define NUM_CHANNELS (2)

#define NUM_BUFFERS (3)

/** Field type
      V4L2_FIELD_NONE = progressive (ie non-interlaced)
      V4L2_FIELD_ANY  = let driver decide
*/
#define FIELD_TYPE V4L2_FIELD_NONE

//----( simple frame-grabbing camera )----------------------------------------

// values, defaults, and ranges are from drivers/media/video/gspca/ov534.c
int Camera::s_brightness          = 0;    // 0-255
int Camera::s_contrast            = 32;   // 0-255
int Camera::s_gain                = 20;   // 0-63
int Camera::s_exposure            = 120;  // 0-255
bool Camera::s_auto_gain          = true;
bool Camera::s_auto_white_balance = true;
bool Camera::s_auto_exposure      = true;
int Camera::s_sharpness           = 0;    // 0-63

void Camera::set_brightness (int x) { s_brightness = bound_to(0, 255, x); }
void Camera::set_contrast (int x) { s_contrast = bound_to(0, 255, x); }
void Camera::set_gain (int x) { s_gain = bound_to(0, 63, x); }
void Camera::set_exposure (int x) { s_exposure = bound_to(0, 63, x); }
void Camera::set_auto_gain (bool x) { s_auto_gain = x; }
void Camera::set_auto_white_balance (bool x) { s_auto_white_balance = x; }
void Camera::set_auto_exposure (bool x) { s_auto_exposure = x; }
void Camera::set_sharpness (int x) { s_sharpness = bound_to(0, 63, x); }

void Camera::set_config (const char * filename)
{
  ConfigParser config(filename);

  set_brightness(config("brightness", s_brightness));
  set_contrast(config("contrast", s_contrast));
  set_gain(config("gain", s_gain));
  set_exposure(config("exposure", s_exposure));
  set_auto_gain(config("auto_gain", s_auto_gain));
  set_auto_white_balance(config("auto_white_balance", s_auto_white_balance));
  set_auto_exposure(config("auto_exposure", s_auto_exposure));
  set_sharpness(config("sharpness", s_sharpness));
}

static std::set<string> g_used_devices;

Camera::Camera (
    size_t default_width,
    size_t default_height,
    size_t default_framerate)

  : VideoSource(
      default_width,
      default_height,
      default_framerate),

    m_camera_fd(-1),

    m_buffer_size(0),
    m_buffer_data(NULL),

    m_frame_count(0)
{
  bool reject_bad_framerate = true;
  for (int i = 0;; ++i) {

  char device_name[64];
  sprintf(device_name, "/dev/video%i", i);
  if (g_used_devices.find(device_name) != g_used_devices.end()) continue;

  // open v4l2 device
  LOG("opening camera " << device_name);
  m_camera_fd = v4l2_open(device_name, O_RDWR | O_NONBLOCK, 0);
  if (reject_bad_framerate and not (m_camera_fd >= 0)) {
    LOG("relaxing framerate constraint and trying again...");
    reject_bad_framerate = false;
    i = -1;
    continue;
  }
  ASSERT(m_camera_fd >= 0, "cannot open camera " << device_name);

  // check capabilities
  v4l2_capability capability;
  CLEAR(capability);

  safe_ioctl(m_camera_fd, VIDIOC_QUERYCAP, &capability);

  PRINT(capability.driver);
  PRINT(capability.card);
  PRINT(capability.bus_info);
  PRINT(capability.version);

  ASSERT(capability.capabilities & V4L2_CAP_VIDEO_CAPTURE,
         "camera does not support video capture");
  ASSERT(capability.capabilities & V4L2_CAP_READWRITE,
         "camera does not support read() method");

  // check audio channels (doesn't work with playstation eye)
  if (capability.capabilities & V4L2_CAP_AUDIO) {
    v4l2_audio audio;

    for (size_t channel = 0;; ++channel) {
      CLEAR(audio);
      audio.index = channel;
      int info = safe_ioctl(m_camera_fd, VIDIOC_G_AUDIO, &audio);
      if (info == EINVAL) break;

      bool stereo = audio.capability & V4L2_AUDCAP_STEREO;
      LOG("audio input #" << channel << " = " << audio.name
          << (stereo ? " (stereo)" : " (mono)"));
    }

    CLEAR(audio);
    safe_ioctl(m_camera_fd, VIDIOC_G_AUDIO, &audio);
    LOG("current audio input is #" << audio.index << " = " << audio.name);

  } else {
    LOG("camera does not support audio input");
  }

  // query device controls
  v4l2_queryctrl qctrl;

  LOG("control values:");
  qctrl.id = V4L2_CTRL_FLAG_NEXT_CTRL;
  while (0 == safe_ioctl(m_camera_fd, VIDIOC_QUERYCTRL, &qctrl, false)) {
    if (V4L2_CTRL_ID2CLASS(qctrl.id) == V4L2_CTRL_CLASS_USER) {

      v4l2_control control;
      control.id = qctrl.id;
      safe_ioctl(m_camera_fd, VIDIOC_G_CTRL, &control);
      LOG("  " << qctrl.name << " = " << control.value);

#define SET_IF_CHANGED(name,field) \
      if ((control.id == name) and (control.value != (field))) { \
        control.value = (field); \
        safe_ioctl(m_camera_fd, VIDIOC_S_CTRL, &control); \
        LOG("    changed to " << control.value); \
      }

      SET_IF_CHANGED(V4L2_CID_BRIGHTNESS, s_brightness)
      SET_IF_CHANGED(V4L2_CID_CONTRAST, s_contrast)
      SET_IF_CHANGED(V4L2_CID_GAIN, s_gain)
      SET_IF_CHANGED(V4L2_CID_EXPOSURE, s_exposure)
      SET_IF_CHANGED(V4L2_CID_AUTOGAIN, s_auto_gain)
      SET_IF_CHANGED(V4L2_CID_AUTO_WHITE_BALANCE, s_auto_white_balance)
      SET_IF_CHANGED(V4L2_CID_EXPOSURE_AUTO,
          s_auto_exposure ? V4L2_EXPOSURE_AUTO : V4L2_EXPOSURE_MANUAL)
      SET_IF_CHANGED(V4L2_CID_SHARPNESS, s_sharpness)

#undef SET_IF_CHANGED

    }

    qctrl.id |= V4L2_CTRL_FLAG_NEXT_CTRL;
  }

  // set device properties
  v4l2_format format;
  CLEAR(format);

  format.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  format.fmt.pix.width       = m_width;
  format.fmt.pix.height      = m_height;
  format.fmt.pix.pixelformat = PIXEL_FORMAT;
  format.fmt.pix.field       = FIELD_TYPE;

  safe_ioctl(m_camera_fd, VIDIOC_S_FMT, &format);

  ASSERT(format.fmt.pix.pixelformat == PIXEL_FORMAT,
         "libv4l did not accept pixel format");

  if ( (format.fmt.pix.width  != m_width)
    or (format.fmt.pix.height != m_height) )
  {
    WARN("libv4l did not accept resolution " << m_width << " x " << m_height);
    m_width = format.fmt.pix.width;
    m_height = format.fmt.pix.height;
  }
  LOG("resolution = " << m_width << " x " << m_height);

  // allocate buffer
  m_buffer_size = NUM_CHANNELS * size();
  m_buffer_data = (uint8_t *) malloc_aligned(m_buffer_size);

  // set streaming parameters
  v4l2_streamparm param;
  CLEAR(param);
  param.type = V4L2_BUF_TYPE_VIDEO_CAPTURE;
  v4l2_captureparm & capture = param.parm.capture;

  safe_ioctl(m_camera_fd, VIDIOC_G_PARM, &param);
  capture.timeperframe.numerator    = 1;
  capture.timeperframe.denominator  = m_framerate;
  safe_ioctl(m_camera_fd, VIDIOC_S_PARM, &param);

  if (m_framerate * capture.timeperframe.numerator
      != capture.timeperframe.denominator)
  {
    LOG(device_name << " did not accept framerate " << m_framerate);
    if (reject_bad_framerate) {
      LOG("looking for camera with higher framerate...");

      v4l2_close(m_camera_fd);
      free_aligned(m_buffer_data);

      m_width = default_width;
      m_height = default_height;

      continue;
    } else {
      m_framerate = capture.timeperframe.denominator
                  / capture.timeperframe.numerator;
    }
  }
  LOG("framerate = " << capture.timeperframe.denominator
              << '/' << capture.timeperframe.numerator);
  LOG("read buffers = " << capture.readbuffers);
  LOG("quality = " << ( capture.capturemode & V4L2_MODE_HIGHQUALITY
                      ? "high" : "low" ));

  m_name = device_name;
  g_used_devices.insert(m_name);

  break;
  }
}

Camera::~Camera ()
{
  float average_frame_rate = m_frame_count / get_elapsed_time();
  PRINT(average_frame_rate);

  v4l2_close(m_camera_fd);
  free_aligned(m_buffer_data);

  g_used_devices.erase(m_name);
}

void Camera::read_buffer () const
{
  safe_read(m_camera_fd, m_buffer_data, m_buffer_size);
  ++m_frame_count;
}

Seconds Camera::capture (Vector<float> & frame_out) const
{
  ASSERT_SIZE(frame_out, size());

  read_buffer();
  Seconds time = Seconds::now();

  float * restrict frame = frame_out;

  for (size_t i = 0, I = size(); i < I; ++i) {
    frame[i] = yuyv_luminance(m_buffer_data, i);
  }

  return time;
}

Seconds Camera::capture (Vector<uint8_t> & frame_out) const
{
  ASSERT_SIZE(frame_out, size());

  read_buffer();
  Seconds time = Seconds::now();

  uint8_t * restrict frame = frame_out;
  uint8_t * restrict buffer = m_buffer_data;

  for (size_t i = 0, I = size(); i < I; ++i) {
    frame[i] = buffer[2 * i];
  }

  return time;
}

Seconds Camera::accumulate (Vector<float> & frame_accum) const
{
  ASSERT_SIZE(frame_accum, size());

  read_buffer();
  Seconds time = Seconds::now();

  float * restrict frame = frame_accum;

  for (size_t i = 0, I = size(); i < I; ++i) {
    frame[i] += yuyv_luminance(m_buffer_data, i);
  }

  return time;
}

Seconds Camera::capture (Vector<float> & frame_out, BoundingBox box) const
{
  ASSERT_SIZE(frame_out, box.size());

  read_buffer();
  Seconds time = Seconds::now();

  float * restrict frame = frame_out;

  const size_t I = box.width();
  const size_t X = width();
  for (size_t j = 0, y = box.y0; y < box.y1; ++j, ++y)
  for (size_t i = 0, x = box.x0; x < box.x1; ++i, ++x) {
    size_t ij = i + j * I;
    size_t xy = x + y * X;

    frame[ij] = yuyv_luminance(m_buffer_data, xy);
  }

  return time;
}

Seconds Camera::accumulate (Vector<float> & frame_accum, BoundingBox box) const
{
  ASSERT_SIZE(frame_accum, box.size());

  read_buffer();
  Seconds time = Seconds::now();

  float * restrict frame = frame_accum;

  const size_t I = box.width();
  const size_t X = width();
  for (size_t j = 0, y = box.y0; y < box.y1; ++j, ++y)
  for (size_t i = 0, x = box.x0; x < box.x1; ++i, ++x) {
    size_t ij = i + j * I;
    size_t xy = x + y * X;

    frame[ij] += yuyv_luminance(m_buffer_data, xy);
  }

  return time;
}

Seconds Camera::capture_yuyv (Vector<float> & yuyv_out) const
{
  ASSERT_SIZE(yuyv_out, 2 * size());

  read_buffer();
  Seconds time = Seconds::now();

  float * restrict yuyv = yuyv_out;

  for (size_t i = 0, I = yuyv_out.size; i < I; ++i) {
    yuyv[i] = static_cast<float>(m_buffer_data[i]);
  }

  return time;
}

Seconds Camera::capture_yuyv (
    Vector<float> & yy_out,
    Vector<float> & u_out,
    Vector<float> & v_out) const
{
  ASSERT_SIZE(yy_out, size());
  ASSERT_SIZE(u_out, size() / 2);
  ASSERT_SIZE(v_out, size() / 2);

  read_buffer();
  Seconds time = Seconds::now();

  float * restrict yy = yy_out;
  float * restrict u = u_out;
  float * restrict v = v_out;

  for (size_t i = 0, I = u_out.size; i < I; ++i) {
    yy[2 * i + 0] = static_cast<float>(m_buffer_data[4 * i + 0]);
    yy[2 * i + 1] = static_cast<float>(m_buffer_data[4 * i + 2]);
    u[i] = static_cast<float>(m_buffer_data[4 * i + 1]);
    v[i] = static_cast<float>(m_buffer_data[4 * i + 3]);
  }

  return time;
}

Seconds Camera::capture_fifth (Vector<float> & frame_out) const
{
  ASSERT_DIVIDES(5, width());
  ASSERT_DIVIDES(5, height());
  ASSERT_SIZE(frame_out, size() / sqr(5));

  read_buffer();
  Seconds time = Seconds::now();

  float * restrict frame = frame_out;
  uint8_t * restrict buffer = m_buffer_data;

  const size_t I = width() / 5;
  const size_t J = height() / 5;
  for (size_t j = 0; j < J; ++j)
  for (size_t i = 0; i < I; ++i) {
    size_t ij = i + j * I;

    float f = 0;

    for (size_t j5 = 0; j5 < 5; ++j5)
    for (size_t i5 = 0; i5 < 5; ++i5) {
      size_t i5j5 = (i*5+i5) + (j*5+j5) * (I*5);

      f += yuyv_luminance(buffer, i5j5);
    }

    frame[ij] = f * 0.04f;
  }

  return time;
}

Seconds Camera::capture_fifth (Vector<uint8_t> & frame_out) const
{
  ASSERT_DIVIDES(5, width());
  ASSERT_DIVIDES(5, height());
  ASSERT_SIZE(frame_out, size() / sqr(5));

  read_buffer();
  Seconds time = Seconds::now();

  uint8_t * restrict frame = frame_out;
  uint8_t * restrict buffer = m_buffer_data;

  const size_t I = width() / 5;
  const size_t J = height() / 5;
  for (size_t j = 0; j < J; ++j)
  for (size_t i = 0; i < I; ++i) {
    size_t ij = i + j * I;

    int f = 0;

    for (size_t j5 = 0; j5 < 5; ++j5)
    for (size_t i5 = 0; i5 < 5; ++i5) {
      size_t i5j5 = (i*5+i5) + (j*5+j5) * (I*5);

      f += buffer[2 * i5j5];
    }

    frame[ij] = (f + 12) / 25;
  }

  return time;
}

Seconds Camera::capture_yuv420p8 (
    Vector<uint8_t> & y_out,
    Vector<uint8_t> & u_out,
    Vector<uint8_t> & v_out) const
{
  ASSERT_SIZE(y_out, size());
  ASSERT_SIZE(u_out, size() / 4);
  ASSERT_SIZE(v_out, size() / 4);

  read_buffer();
  Seconds time = Seconds::now();

  uint8_t * restrict y = y_out;
  uint8_t * restrict u = u_out;
  uint8_t * restrict v = v_out;
  uint8_t * restrict buffer = m_buffer_data;

  const size_t W = width();
  const size_t I = width();
  const size_t J = height();
  for (size_t j = 0; j < J; ++j)
  for (size_t i = 0; i < I; ++i) {
    size_t ij = i + j * I;

    y[ij] = buffer[2 * ij];
  }

  const size_t I2 = I / 2;
  const size_t J2 = J / 2;
  for (size_t j = 0; j < J2; ++j)
  for (size_t i = 0; i < I2; ++i) {
    size_t ij = i + j * I2;

    u[ij] = (int(buffer[4 * ij + 1]) + int(buffer[4 * ij + 1 + 2 * W])) / 2;
    v[ij] = (int(buffer[4 * ij + 3]) + int(buffer[4 * ij + 3 + 2 * W])) / 2;
  }

  return time;
}

Seconds Camera::capture_fifth_yuv420p8 (
    Vector<uint8_t> & y_out,
    Vector<uint8_t> & u_out,
    Vector<uint8_t> & v_out) const
{
  ASSERT_DIVIDES(10, width());
  ASSERT_DIVIDES(10, height());
  ASSERT_SIZE(y_out, size() / 25);
  ASSERT_SIZE(u_out, size() / 100);
  ASSERT_SIZE(v_out, size() / 100);

  read_buffer();
  Seconds time = Seconds::now();

  uint8_t * restrict y = y_out;
  uint8_t * restrict u = u_out;
  uint8_t * restrict v = v_out;
  uint8_t * restrict buffer = m_buffer_data;

  const size_t W = width();
  const size_t I = width() / 5;
  const size_t J = height() / 5;
  for (size_t j = 0; j < J; ++j)
  for (size_t i = 0; i < I; ++i) {

    int y_ij = 0;

    for (size_t j5 = 0; j5 < 5; ++j5)
    for (size_t i5 = 0; i5 < 5; ++i5) {
      size_t i5j5 = (i*5+i5) + (j*5+j5) * W;

      y_ij += buffer[2 * i5j5];
    }

    size_t ij = i + j * I;
    y[ij] = (y_ij + 12) / 25;
  }

  const size_t W2 = W / 2;
  const size_t I2 = I / 2;
  const size_t J2 = J / 2;
  for (size_t j = 0; j < J2; ++j)
  for (size_t i = 0; i < I2; ++i) {

    int u_ij = 0;
    int v_ij = 0;

    for (size_t j5 = 0; j5 < 5; ++j5)
    for (size_t i10 = 0; i10 < 10; ++i10) {
      size_t i10j5 = (i*10+i10) + (j*5+j5) * W2;

      u_ij += buffer[4 * i10j5 + 1];
      v_ij += buffer[4 * i10j5 + 3];
    }

    size_t ij = i + j * I2;
    u[ij] = (u_ij + 25) / 50;
    v[ij] = (v_ij + 25) / 50;
  }

  return time;
}

BoundingBox Camera::autocrop (float threshold, size_t num_samples)
{
  LOG("accumulating frames");
  Vector<float> frame(size());
  frame.zero();
  for (size_t i = 0; i < num_samples; ++i) {
    accumulate(frame);
  }

  LOG("projecting image to axes");
  // assume image is arranged in horizontal lines
  Vector<float> horiz(m_width);
  horiz.zero();
  for (size_t i = 0; i < m_width; ++i) {
    for (size_t j = 0; j < m_height; ++j) {
      horiz[i] += frame[i + j * m_width];
      //horiz[i] += frame[i * m_height + j];
    }
  }
  horiz -= min(horiz);
  horiz /= max(horiz);

  Vector<float> vert(m_height);
  vert.zero();
  for (size_t i = 0; i < m_width; ++i) {
    for (size_t j = 0; j < m_height; ++j) {
      vert[j] += frame[i + j * m_width];
      //vert[j] += frame[i * m_height + j];
    }
  }
  vert -= min(vert);
  vert /= max(vert);

  LOG1("determine image parity");
  {
    size_t w = m_width / 4;
    size_t h = m_height / 4;
    float center = sum(horiz.block(w, 1))
                + sum(horiz.block(w, 2))
                + sum(vert.block(h, 1))
                + sum(vert.block(h, 2));
    float periph = sum(horiz.block(w, 0))
                + sum(horiz.block(w, 3))
                + sum(vert.block(h, 0))
                + sum(vert.block(h, 3));

    if (center > periph) {
      LOG("finding light central box");
    } else {
      LOG("finding dark central box");
      horiz *= -1.0f;
      vert *= -1.0f;
    }
  }

  LOG("finding image edges");
  BoundingBox box;

  for (box.x0 = m_width / 2; box.x0 > 0; --box.x0) {
    if (horiz[box.x0] < threshold) break;
  }
  for (box.x1 = m_width / 2; box.x1 < m_width - 1; ++box.x1) {
    if (horiz[box.x1] < threshold) break;
  }

  for (box.y0 = m_height / 2; box.y0 > 0; --box.y0) {
    if (vert[box.y0] < threshold) break;
  }
  for (box.y1 = m_height / 2; box.y1 < m_height - 1; ++box.y1) {
    if (vert[box.y1] < threshold) break;
  }

  LOG("bounded image to " << box);

  return box;
}

//----( flood fill region )---------------------------------------------------

CameraRegion::FloodFill::FloodFill (const char * config_filename)
  : m_config(config_filename),

    m_num_bg_frames(m_config("num_bg_frames", 32)),
    m_lowpass_units(m_config("lowpass_units", 4)),
    m_floodfill_threshold(m_config("floodfill_threshold", 0.3f)),
    m_mask_box_threshold(m_config("mask_box_threshold", 0.2f)),
    m_dilate_radius(m_config("dilate_radius", 2)),
    m_dilate_sharpness(m_config("dilate_sharpness", 20))
{}

void CameraRegion::FloodFill::init (
    Camera & camera,
    BoundingBox & bbox,
    float * restrict & restrict mask,
    float * restrict & restrict background)
{
  const size_t W = camera.width();
  const size_t H = camera.height();
  const size_t R = min(W,H) / m_lowpass_units;

  LOG("accumulate background");
  Vector<float> bg(W * H);
  bg.zero();
  for (size_t i = 0; i < m_num_bg_frames; ++i) {
    camera.accumulate(bg);
  }
  bg *= 1.0f / m_num_bg_frames;

  Vector<float> hi(W * H);
  Vector<float> lo(W * H);

// TODO test camera region hdr
//#define CAMERA_REGION_HDR
#ifdef CAMERA_REGION_HDR
  LOG("hdr filter background");
  Vector<float> temp1(W * H);
  Vector<float> temp2(W * H);
  multiply(1/255.0f, bg, hi);
  Image::hdr_01(H,W,R, hi, lo, temp1, temp2);
#else // CAMERA_REGION_HDR
  LOG("highpass filter background");
  lo = bg;
  Image::quadratic_blur_scaled(H,W,R, lo, hi);
  for (size_t xy = 0; xy < W*H; ++xy) {
    hi[xy] = bg[xy] - lo[xy];
  }
  hi -= min(hi);
  hi /= max(hi);
#endif // CAMERA_REGION_HDR

  LOG("dilate");
  Image::dilate(H,W, m_dilate_radius, hi, lo, m_dilate_sharpness);

  LOG("find floodfill mask");
  Vector<uint8_t> int_mask(W * H);
  Image::vh_convex_floodfill(H, W, m_floodfill_threshold, hi, int_mask);

  LOG("blur mask");
  for (size_t xy = 0; xy < W*H; ++xy) {
    lo[xy] = int_mask[xy] ? 1 : -3;
  }
  Image::quadratic_blur_scaled(H, W, W / 100, lo, hi);
  for (size_t xy = 0; xy < W*H; ++xy) {
    lo[xy] = max(0.0f, lo[xy]);
  }

  LOG("find bounding box");
  Vector<float> mask_x(W);  mask_x.zero();
  Vector<float> mask_y(H);  mask_y.zero();
  for (size_t x = 0; x < W; ++x)
  for (size_t y = 0; y < H; ++y) {
    mask_x[x] += lo[x + y * W] / H;
    mask_y[y] += lo[x + y * W] / W;
  }

  bbox = camera.full_box();
  float M = m_mask_box_threshold;
  while ((bbox.x0 < W / 2) and (mask_x[bbox.x0] < M)) ++bbox.x0;
  while ((bbox.x1 > W / 2) and (mask_x[bbox.x1] < M)) --bbox.x1;
  while ((bbox.y0 < H / 2) and (mask_y[bbox.y0] < M)) ++bbox.y0;
  while ((bbox.y1 > H / 2) and (mask_y[bbox.y1] < M)) --bbox.y1;

  bbox = bbox.round_to(8);
  ASSERT_LT(bbox.x0, bbox.x1);
  ASSERT_LT(bbox.y0, bbox.y1);

  LOG("resize mask and background");
  size_t w = bbox.width();
  size_t h = bbox.height();
  size_t x0 = bbox.x0;
  size_t y0 = bbox.y0;

  background = malloc_float(w * h);
  mask = malloc_float(w * h);

  for (size_t y = 0; y < h; ++y)
  for (size_t x = 0; x < w; ++x) {
    size_t XY = x + x0 + (y + y0) * W;
    size_t xy = x + y * w;

    background[xy] = bg[XY];
    mask[xy] = lo[XY];
  }
}

//----( disk region )---------------------------------------------------------

CameraRegion::Disk::Disk (
    size_t num_bg_frames,
    float softness)
  : m_num_bg_frames(num_bg_frames),
    m_softness(softness)
{
  ASSERT_LE(0.5f, softness);
}

void CameraRegion::Disk::init (
    Camera & camera,
    BoundingBox & bbox,
    float * restrict & restrict mask,
    float * restrict & restrict background)
{
  const size_t W = camera.width();
  const size_t H = camera.height();

  const size_t w = min(W,H);
  const size_t h = w;
  const float R = 0.5f * w;

  const size_t x0 = bbox.x0 = (W - w) / 2;
  const size_t y0 = bbox.y0 = (H - h) / 2;
  bbox.x1 = x0 + w;
  bbox.y1 = y0 + h;

  background = malloc_float(w * h);
  mask = malloc_float(w * h);

  for (size_t y = 0; y < h; ++y)
  for (size_t x = 0; x < w; ++x) {
    size_t xy = x + y * w;

    float r = sqrt( sqr(x + 0.5f - R)
                 + sqr(y + 0.5f - R) );

    mask[xy] = bound_to(0.0f, 1.0f, 1 + (R - r) / m_softness);
  }

  LOG("accumulate background");
  Vector<float> bg(w * h, background);
  bg.zero();
  for (size_t i = 0; i < m_num_bg_frames; ++i) {
    camera.accumulate(bg, bbox);
  }
  bg *= 1.0f / m_num_bg_frames;
}

//----( camera region )-------------------------------------------------------

CameraRegion::CameraRegion (
    Region * region,
    size_t default_width,
    size_t default_height)

  : VideoSource(),
    m_camera(default_width, default_height),
    m_mask(NULL),
    m_background(NULL)
{
  if (not region) region = new FloodFill();

  region->init(m_camera, m_bbox, m_mask, m_background);

  delete region;

  m_width = m_bbox.width();
  m_height = m_bbox.height();
  m_framerate = m_camera.framerate();
  PRINT3(width(), height(), framerate());

  m_time = Seconds::now();
}

CameraRegion::~CameraRegion ()
{
  free_float(m_mask);
  free_float(m_background);
}

void CameraRegion::interior (
    size_t padding,
    Vector<float> & frame,
    Vector<float> & temp) const
{
  Vector<float> mask(size(), const_cast<float*>(m_mask));
  Vector<float> bg(size(), const_cast<float*>(m_background));

  const size_t I = width();
  const size_t J = height();

  frame = mask;
  {
    size_t x,y;
    for (x = 0; x < I; ++x) {
      y = 0;      frame[x+I*y] = 0;
      y = J - 1;  frame[x+I*y] = 0;
    }
    for (y = 0; y < J; ++y) {
      x = 0;      frame[x+I*y] = 0;
      x = I - 1;  frame[x+I*y] = 0;
    }
  }

  Image::quadratic_blur_scaled(J,I, padding, frame, temp);

  for (size_t xy = 0; xy < I*J; ++xy) {
    frame[xy] = max(0.0f, 4 * frame[xy] - 3);
  }

  frame *= bg;
}

Seconds CameraRegion::capture_crop (Vector<float> & frame) const
{
  ASSERT_SIZE(frame, size());

  m_camera.read_buffer();
  m_time = Seconds::now();

  const size_t w = width();
  const size_t h = height();
  const size_t x0 = m_bbox.x0;
  const size_t y0 = m_bbox.y0;
  const size_t W = m_camera.width();

  for (size_t y = 0; y < h; ++y) {
    size_t Y = y + y0;

    const uint8_t * restrict data = m_camera.m_buffer_data + 2 * W * Y;
    float * restrict f = frame + w * y;

    for (size_t x = 0; x < w; ++x) {
      size_t X = x + x0;

      f[x] = data[2 * X];
    }
  }

  return m_time;
}

Seconds CameraRegion::capture_crop_mask (Vector<float> & frame) const
{
  ASSERT_SIZE(frame, size());

  m_camera.read_buffer();
  m_time = Seconds::now();

  const size_t w = width();
  const size_t h = height();
  const size_t x0 = m_bbox.x0;
  const size_t y0 = m_bbox.y0;
  const size_t W = m_camera.width();

  for (size_t y = 0; y < h; ++y) {
    size_t Y = y + y0;

    const uint8_t * restrict data = m_camera.m_buffer_data + 2 * W * Y;
    const float * restrict mask = m_mask + w * y;
    float * restrict f = frame + w * y;

    for (size_t x = 0; x < w; ++x) {
      size_t X = x + x0;

      f[x] = data[2 * X] * mask[x];
    }
  }

  return m_time;
}

Seconds CameraRegion::capture_crop_mask_sub (Vector<float> & frame) const
{
  ASSERT_SIZE(frame, size());

  m_camera.read_buffer();
  Seconds time = Seconds::now();
  float dt = time - m_time;
  m_time = time;
  const float drift = 1.0f - expf(-dt / DEFAULT_GAIN_TIMESCALE_SEC);

  const size_t w = width();
  const size_t h = height();
  const size_t x0 = m_bbox.x0;
  const size_t y0 = m_bbox.y0;
  const size_t W = m_camera.width();

  for (size_t y = 0; y < h; ++y) {
    size_t Y = y + y0;

    const uint8_t * restrict data = m_camera.m_buffer_data + 2 * W * Y;
    const float * restrict mask = m_mask + w * y;
    float * restrict bg = m_background + w * y;
    float * restrict f = frame + w * y;

    for (size_t x = 0; x < w; ++x) {
      size_t X = x + x0;

      float df = data[2 * X] - bg[x];
      bg[x] += drift * df;
      f[x] = df * mask[x];
    }
  }

  return m_time;
}

Seconds CameraRegion::capture_crop_mask_ceil (Vector<float> & frame) const
{
  ASSERT_SIZE(frame, size());

  m_camera.read_buffer();
  Seconds time = Seconds::now();
  float dt = time - m_time;
  m_time = time;
  const float decay = expf(-dt / DEFAULT_GAIN_TIMESCALE_SEC);

  const size_t w = width();
  const size_t h = height();
  const size_t x0 = m_bbox.x0;
  const size_t y0 = m_bbox.y0;
  const size_t W = m_camera.width();

  for (size_t y = 0; y < h; ++y) {
    size_t Y = y + y0;

    const uint8_t * restrict data = m_camera.m_buffer_data + 2 * W * Y;
    const float * restrict mask = m_mask + w * y;
    float * restrict bg = m_background + w * y;
    float * restrict f = frame + w * y;

    for (size_t x = 0; x < w; ++x) {
      size_t X = x + x0;

      float f_new = data[2 * X];
      float f_old = bg[x];

      bg[x] = max(decay * f_old, f_new);
      f[x] = (f_new - f_old) * mask[x];
    }
  }

  return m_time;
}

//----( video stream operations )---------------------------------------------

Seconds ShrinkToHalf::capture (Vector<float> & frame_out) const
{
  Seconds time = m_video->capture(m_frame);

  Image::scale_by_half (
      2 * height(),
      2 * width(),
      m_frame,
      frame_out);

  return time;
}

//----( video threads )-------------------------------------------------------

CameraThread::CameraThread (VideoSource & video)
  : m_video(video),
    m_frame(video.size()),
    m_running(false)
{}

void CameraThread::start ()
{
  ASSERT(not m_running, "started CameraThread twice");
  m_running = true;
  Thread::start();
}

void CameraThread::stop ()
{
  ASSERT(m_running, "stopped CameraThread twice");
  m_running = false;
  Thread::wait();
}

void CameraThread::run ()
{
  while (m_running) {
    Seconds time = m_video.capture(m_frame);
    process(time, m_frame);
  }
}

