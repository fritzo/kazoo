
#include "images.h"
#include "animate.h"
#include "camera.h"
#include "events.h"
#include "args.h"
#include <algorithm>

const char * infile = "data/test_in.im";
const char * outfile = "data/test_out.im";

using namespace Image;

void normalize_01 (Vector<float> & data)
{
  data -= min(data);
  data /= max(data);
}

//----( testing framework )---------------------------------------------------

//----( operations )----

class Operation
{
protected:

  Screen m_screen;

public:

  const size_t width, height, size;

  Operation (size_t w, size_t h, Args & args)
    : m_screen(Rectangle(w,h), "images test - any key exits", "images_test"),
      width(w),
      height(h),
      size(w*h)
  {}
  virtual ~Operation () {}

  virtual void transform (Vector<float> & image) { ASSERT_SIZE(image, size); }

  virtual void display (Vector<float> & image)
  {
    transform(image);
    normalize_01(image);
    m_screen.draw(image, true);
    m_screen.update();
  }
};

class ColorOperation : public Operation
{
protected:

  Vector<float> red, green, blue;

public:

  ColorOperation (size_t w, size_t h, Args & args)
    : Operation(w,h, args),
      red(size),
      green(size),
      blue(size)
  {
    red.zero();
    green.zero();
    blue.zero();
  }
  virtual ~ColorOperation () {}

  virtual void display (Vector<float> & image)
  {
    transform(image); // this should set red,green,blue
    m_screen.draw(red, green, blue, true);
    m_screen.update();
  }
};

//----( sources )----

class Source
{
public:

  virtual ~Source () {}

  virtual size_t width () const = 0;
  virtual size_t height () const = 0;

  virtual void display (Operation & transform) = 0;
};

class LiveSource : public Source
{
   VideoSource * m_video;

public:

  LiveSource (VideoSource * video) : m_video(video) {}
  virtual ~LiveSource () { delete m_video; }

  virtual size_t width () const { return m_video->width(); }
  virtual size_t height () const { return m_video->height(); }

  virtual void display (Operation & transform)
  {
    Vector<float> temp(m_video->size());
    Vector<float> image(m_video->size());

    while (not key_pressed()) {
      m_video->capture_transpose(image, temp);
      transform.display(image);
    }

    Image::write_image(outfile, width(), height(), image);
  }
};

class FileSource : public Source
{
  size_t m_width;
  size_t m_height;
  float * m_data;

public:

  FileSource (const char * filename)
    : m_data(NULL)
  {
    Image::read_image(filename, m_width, m_height, m_data);
  }
  virtual ~FileSource () { if (m_data) free_float(m_data); }

  virtual size_t width () const { return m_width; }
  virtual size_t height () const { return m_height; }

  virtual void display (Operation & transform)
  {
    Vector<float> image(m_width * m_height, m_data);

    transform.display(image);

    wait_for_keypress();

    Image::write_image(outfile, width(), height(), image);
  }
};

//----( image transforms )----------------------------------------------------

class Transpose : public Operation
{
  Vector<float> temp;

public:

  Transpose (size_t w, size_t h, Args & args)
    : Operation(w, h, args), temp(size)
  {}
  virtual ~Transpose () {}

  virtual void transform (Vector<float> & image)
  {
    transpose_8(width, height, image, temp);
    image = temp;
  }
};

class SquareBlur : public Operation
{
  Vector<float> temp;

public:

  const size_t radius;

  SquareBlur (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      radius(args.pop(2))
  { PRINT(radius); }
  virtual ~SquareBlur () {}

  virtual void transform (Vector<float> & image)
  {
    square_blur_scaled(width, height, radius, image, temp);
  }
};

class SquareBlurWrap : public Operation
{
  Vector<float> temp;

public:

  const size_t radius;

  SquareBlurWrap (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      radius(args.pop(2))
  { PRINT(radius); }
  virtual ~SquareBlurWrap () {}

  virtual void transform (Vector<float> & image)
  {
    square_blur_axis_wrap(width, height, radius, image, temp);
    image = temp;
  }
};

class QuadBlur : public Operation
{
  Vector<float> temp;

public:

  const size_t radius;

  QuadBlur (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      radius(args.pop(2))
  { PRINT(radius); }
  virtual ~QuadBlur () {}

  virtual void transform (Vector<float> & image)
  {
    quadratic_blur_scaled(width, height, radius, image, temp);
  }
};

class ExpBlur : public Operation
{
  Vector<float> temp;

public:

  const float radius;

  ExpBlur (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      radius(args.pop(2.0f))
  { PRINT(radius); }
  virtual ~ExpBlur () {}

  virtual void transform (Vector<float> & image)
  {
    exp_blur_zero(width, height, radius, image, temp);
  }
};

class Dilate : public Operation
{
  Vector<float> temp;

public:

  const size_t radius;
  const float sharpness;

  Dilate (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      radius(args.pop(5)),
      sharpness(args.pop(10.0f))
  { PRINT2(radius, sharpness); }
  virtual ~Dilate () {}

  virtual void transform (Vector<float> & image)
  {
    normalize_01(image);
    dilate(width, height, radius, image, temp, sharpness);
  }
};

class Sharpen : public Operation
{
  Vector<float> sharp;
  Vector<float> temp;

public:

  const size_t radius;

  Sharpen (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      sharp(size),
      temp(size),
      radius(args.pop(4))
  { PRINT(radius); }
  virtual ~Sharpen () {}

  virtual void transform (Vector<float> & image)
  {
    sharpen(width, height, radius, image, sharp, temp);
    affine_to_01(sharp);
    image = sharp;
  }
};

class Hdr : public Operation
{
  Vector<float> temp1;
  Vector<float> temp2;
  Vector<float> temp3;

public:

  const size_t radius;

  Hdr (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp1(size),
      temp2(size),
      temp3(size),
      radius(args.pop(8))
  { PRINT(radius); }
  virtual ~Hdr () {}

  virtual void transform (Vector<float> & image)
  {
    hdr_01(width, height, radius, image, temp1, temp2, temp3);
  }
};

class EnhancePoints : public Operation
{
  Vector<float> temp;

public:

  const size_t radius;

  EnhancePoints (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      radius(args.pop(5))
  { PRINT(radius); }
  virtual ~EnhancePoints () {}

  virtual void transform (Vector<float> & image)
  {
    quadratic_blur_axis(width, height, radius, image, temp);
    transpose_8(width, height, temp, image);
    quadratic_blur_axis(height, width, radius, image, temp);
    transpose_8(height, width, temp, image);
    enhance_points(width, height, radius, image, temp);
    image = temp;
    affine_to_01(image);
  }
};

template<size_t R>
class EnhancePoints_ : public Operation
{
  Vector<float> temp;

public:

  EnhancePoints_ (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size)
  {}
  virtual ~EnhancePoints_ () {}

  virtual void transform (Vector<float> & image)
  {
    quadratic_blur_axis(width, height, R, image, temp);
    transpose_8(width, height, temp, image);
    quadratic_blur_axis(height, width, R, image, temp);
    transpose_8(height, width, temp, image);
    enhance_points_<R>(width, height, image, temp);
    image = temp;

    affine_to_01(image);
  }
};

class EnhancePoints_248 : public Operation
{
  Vector<float> temp;
  Vector<float> sum;

public:

  EnhancePoints_248 (size_t w, size_t h, Args & args)
    : Operation(w, h, args),

      temp(size),
      sum(size)
  {}
  virtual ~EnhancePoints_248 () {}

  virtual void transform (Vector<float> & image)
  {
    {
      enum { R = 2 };
      quadratic_blur_scaled(width, height, R, image, temp);
      enhance_points_<R>(width, height, image, temp);

      sum = temp;
    }

    {
      enum { R = 4 };
      quadratic_blur_scaled(width, height, R, image, temp);
      enhance_points_<R>(width, height, image, temp);

      sum += temp;
    }

    {
      enum { R = 8 };
      quadratic_blur_scaled(width, height, R, image, temp);
      enhance_points_<R>(width, height, image, temp);

      sum += temp;
    }

    image = sum;
    affine_to_01(image);
  }
};

template<size_t R>
class EnhanceFingers_ : public ColorOperation
{
  Vector<float> tips;
  Vector<float> shafts;

public:

  EnhanceFingers_ (size_t w, size_t h, Args & args)
    : ColorOperation(w, h, args),
      tips(red),
      shafts(green)
  {}
  virtual ~EnhanceFingers_ () {}

  virtual void transform (Vector<float> & image)
  {
    blue = image;

    quadratic_blur_axis(width, height, R, image, tips);
    transpose_8(width, height, tips, image);
    quadratic_blur_axis(height, width, R, image, tips);
    transpose_8(height, width, tips, image);

    enhance_fingers_<R>(width, height, image, tips, shafts);

    affine_to_01(red);
    affine_to_01(green);
    green -= red;
    affine_to_01(green);
    affine_to_01(blue);
  }
};

template<size_t R>
class OrientFingers_ : public ColorOperation
{
  Vector<float> tips;
  Vector<float> shafts;
  Vector<float> dx;
  Vector<float> dy;
  Vector<float> temp;

public:

  OrientFingers_ (size_t w, size_t h, Args & args)
    : ColorOperation(w, h, args),
      tips(green),
      shafts(w * h),
      dx(red),
      dy(blue),
      temp(w * h)
  {}
  virtual ~OrientFingers_ () {}

  virtual void transform (Vector<float> & image)
  {
    blue = image;

    quadratic_blur_axis(width, height, R, image, temp);
    transpose_8(width, height, temp, image);
    quadratic_blur_axis(height, width, R, image, temp);

    enhance_fingers_<R>(height, width, temp, tips, shafts);
    transpose_8(height, width, tips, temp);
    tips = temp;

    local_moments_transpose(height, width, 3 * R, shafts, dx, dy, temp);

    affine_to_01(red);
    affine_to_01(green);
    affine_to_01(blue);
  }
};

template<size_t R>
class EnhanceHands_ : public ColorOperation
{
  Vector<float> tips;
  Vector<float> shafts;

  Vector<float> small;
  Vector<float> smaller;

  Vector<float> palms;

public:

  EnhanceHands_ (size_t w, size_t h, Args & args)
    : ColorOperation(w, h, args),

      tips(red),
      shafts(green),

      small(w * h / 4),
      smaller(w * h / 4 / 4),

      palms(w * h / 4 / 4)
  {}
  virtual ~EnhanceHands_ () {}

  virtual void transform (Vector<float> & image)
  {
    quadratic_blur_axis(width, height, R, image, tips);
    transpose_8(width, height, tips, image);
    quadratic_blur_axis(height, width, R, image, tips);
    transpose_8(height, width, tips, image);

    scale_by_half_8(width, height, image, small);
    scale_by_half_8(width/2, height/2, small, smaller);

    enhance_fingers_<R>(width, height, image, tips, shafts);
    enhance_points_<R>(width/4, height/4, smaller, palms);

    // TODO maybe subtract blurred tips from shafts
    // TODO maybe subtract blurred shafts from palms

    scale_smoothly_by_four(width/4, height/4, palms, blue, image);

    affine_to_01(red);
    affine_to_01(green);
    affine_to_01(blue);
  }
};

class EnhanceCrosses : public Operation
{
  Vector<float> feature;
  Vector<float> temp1;
  Vector<float> temp2;

public:

  const size_t radius;

  EnhanceCrosses (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      feature(size),
      temp1(size),
      temp2(size),
      radius(args.pop(3))
  { PRINT(radius); }
  virtual ~EnhanceCrosses () {}

  virtual void transform (Vector<float> & image)
  {
    hdr_real(width, height, radius, image, feature, temp1, temp2);
    quadratic_blur_scaled(width, height, radius, image, feature);
    enhance_crosses(width, height, radius, image, feature);
    image = feature;
    affine_to_01(image);
  }
};

class EnhanceLines : public Operation
{
  Vector<float> temp1;
  Vector<float> temp2;

public:

  const size_t radius;

  EnhanceLines (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp1(size),
      temp2(size),
      radius(args.pop(5))
  { PRINT(radius); }
  virtual ~EnhanceLines () {}

  virtual void transform (Vector<float> & image)
  {
    enhance_lines(width, height, radius, image, temp1, temp2);
    affine_to_01(image);
  }
};

class DetectChanges : public Operation
{
  Vector<float> mean;
  Vector<float> variance;

public:

  const float timescale;

  DetectChanges (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      mean(size),
      variance(size),
      timescale( DEFAULT_VIDEO_FRAMERATE
               * args.pop(DEFAULT_CHANGE_TIMESCALE_SEC))
  { PRINT(timescale); }
  virtual ~DetectChanges () {}

  virtual void transform (Vector<float> & image)
  {
    detect_changes(size, image, mean, variance, 1 / timescale);
    affine_to_01(image);
  }
};

class HorizChange : public ColorOperation
{
  Vector<float> mean;
  Vector<float> variance;
  Vector<float> mass;
  Vector<float> moment;
  Vector<float> mass_old;
  Vector<float> moment_old;
  Vector<float> momentum;

public:

  const float timescale;

  HorizChange (size_t w, size_t h, Args & args)
    : ColorOperation(w, h, args),
      mean(size),
      variance(size),
      mass(width),
      moment(width),
      mass_old(width),
      moment_old(width),
      momentum(width),
      timescale( DEFAULT_VIDEO_FRAMERATE
               * args.pop(DEFAULT_CHANGE_TIMESCALE_SEC))
  { PRINT(timescale); }
  virtual ~HorizChange () {}

  virtual void transform (Vector<float> & image)
  {
    const float dt = 1 / timescale;
    const float tol = 1e-8f;

    detect_change_moment_y(
        width, height,
        image,
        mean, variance,
        mass, moment,
        dt);

    mass += tol;

    update_momentum(
        width,
        mass, moment,
        mass_old, moment_old,
        momentum,
        dt);

    float * restrict im = image;
    float * restrict r = red;
    float * restrict g = green;
    float * restrict b = blue;

    for (size_t x = 0, X = width; x < X; ++x) {

      float m = mass[x];
      float p = momentum[x];

      for (size_t y = 0, Y = height; y < X; ++y) {
        size_t i = Y * x + y;

        float c = im[i];

        g[i] = c;
        r[i] = c / m * max(0.0f, p);
        b[i] = c / m * max(0.0f, -p);
      }
    }

    affine_to_01(red);
    affine_to_01(green);
    affine_to_01(blue);
  }
};

class Sand : public Operation
{
  Vector<float> temp;
  Vector<float> feature;

public:

  const float radius;

  Sand (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      feature(size),
      radius(args.pop(4))
  { PRINT(radius); }
  virtual ~Sand () {}

  virtual void transform (Vector<float> & image)
  {
    quadratic_blur_scaled(width, height, radius, image, temp);
    enhance_points(width, height, radius, image, feature);
    image = feature;
    affine_to_01(image);
  }
};

template<size_t R>
class Sand_ : public Operation
{
  Vector<float> temp;
  Vector<float> feature;

public:

  Sand_ (size_t w, size_t h, Args & args)
    : Operation(w, h, args),
      temp(size),
      feature(size)
  {}
  virtual ~Sand_ () {}

  virtual void transform (Vector<float> & image)
  {
    quadratic_blur_scaled(width, height, R, image, temp);
    enhance_points_<R>(width, height, image, feature);
    image = feature;
    affine_to_01(image);
  }
};

class LocalOpticalFlow : public ColorOperation
{
  Vector<float> highpass0;
  Vector<float> highpass1;
  Vector<float> sx;
  Vector<float> sy;
  Vector<float> temp1;
  Vector<float> temp2;

  float max_flow;

public:

  const float highpass_radius;

  LocalOpticalFlow (size_t w, size_t h, Args & args)
    : ColorOperation(w, h, args),

      highpass0(size),
      highpass1(size),
      sx(size),
      sy(size),
      temp1(size),
      temp2(size),

      max_flow(0),
      highpass_radius(args.pop(8))
  {
    PRINT(highpass_radius);
    highpass1.zero();
  }
  virtual ~LocalOpticalFlow ()
  {
    PRINT(max_flow);
  }

  virtual void transform (Vector<float> & image)
  {
    highpass0 = highpass1;
    highpass(width, height, highpass_radius, image, highpass1, temp1);
    local_optical_flow(
        width, height,
        highpass0, highpass1,
        sx, sy,
        temp1, temp2);

    max_flow *= 0.999f;
    imax(max_flow, sqrtf(1e-8f + max_norm_squared(sx)));
    imax(max_flow, sqrtf(1e-8f + max_norm_squared(sy)));
    float gain = 1 / max_flow;

    float rx = gain * 1;
    float bx = gain * cos(M_PI / 3);
    float by = gain * sin(M_PI / 3);
    float gx = gain * cos(-M_PI / 3);
    float gy = gain * sin(-M_PI / 3);

    float * restrict x = sx;
    float * restrict y = sy;
    float * restrict r = red;
    float * restrict g = green;
    float * restrict b = blue;

    for (size_t ij = 0, IJ = size; ij < IJ; ++ij) {
      r[ij] = max(0.0f, rx * x[ij]);
      g[ij] = max(0.0f, gx * x[ij]) + max(0.0f, gy * y[ij]);
      b[ij] = max(0.0f, bx * x[ij]) + max(0.0f, by * y[ij]);
    }
  }
};

class LocalOpticalFlowPyramid : public ColorOperation
{
  Vector<float> highpass0;
  Vector<float> highpass1;
  Vector<float> im_sum;
  Vector<float> im_diff;
  Vector<float> sx;
  Vector<float> sy;
  Vector<float> temp1;
  Vector<float> temp2;
  Vector<float> temp3;

  float max_flow;

public:

  const float highpass_radius;

  LocalOpticalFlowPyramid (size_t w, size_t h, Args & args)
    : ColorOperation(w, h, args),

      highpass0(size),
      highpass1(size),
      im_sum(size),
      im_diff(size),
      sx(size),
      sy(size),
      temp1(size),
      temp2(size),
      temp3(size),

      max_flow(0),
      highpass_radius(args.pop(8))
  {
    PRINT(highpass_radius);
    highpass1.zero();
  }
  virtual ~LocalOpticalFlowPyramid ()
  {
    PRINT(max_flow);
  }

  virtual void transform (Vector<float> & image)
  {
    highpass0 = highpass1;
    highpass(width, height, highpass_radius, image, highpass1, temp1);
    local_optical_flow_pyramid(
        width, height,
        highpass0, highpass1,
        im_sum, im_diff,
        sx, sy,
        temp1, temp2, temp3);

//#define SMOOTH_OPTICAL_FLOW
#ifdef SMOOTH_OPTICAL_FLOW
    quadratic_blur(width, height, 1, sx, temp1);
    quadratic_blur(width, height, 1, sy, temp1);
#endif // SMOOTH_OPTICAL_FLOW

    max_flow *= 0.999f;
    imax(max_flow, sqrtf(1e-8f + max_norm_squared(sx)));
    imax(max_flow, sqrtf(1e-8f + max_norm_squared(sy)));
    float gain = 1 / max_flow;

    float rx = gain * 1;
    float bx = gain * cos(M_PI / 3);
    float by = gain * sin(M_PI / 3);
    float gx = gain * cos(-M_PI / 3);
    float gy = gain * sin(-M_PI / 3);

    float * restrict x = sx;
    float * restrict y = sy;
    float * restrict r = red;
    float * restrict g = green;
    float * restrict b = blue;

    for (size_t ij = 0, IJ = size; ij < IJ; ++ij) {
      r[ij] = max(0.0f, rx * x[ij]);
      g[ij] = max(0.0f, gx * x[ij]) + max(0.0f, gy * y[ij]);
      b[ij] = max(0.0f, bx * x[ij]) + max(0.0f, by * y[ij]);
    }
  }
};

class OpticalFlow : public ColorOperation
{
  Vector<float> highpass0;
  Vector<float> highpass1;
  Vector<float> sx;
  Vector<float> sy;
  Vector<float> ixx;
  Vector<float> ixy;
  Vector<float> iyy;
  Vector<float> fx;
  Vector<float> fy;

  float max_flow;

public:

  const float krig_radius;
  const float highpass_radius;

  OpticalFlow (size_t w, size_t h, Args & args)
    : ColorOperation(w, h, args),

      highpass0(size),
      highpass1(size),
      sx(size),
      sy(size),
      ixx(size),
      ixy(size),
      iyy(size),
      fx(size),
      fy(size),

      max_flow(0),
      krig_radius(args.pop(8)),
      highpass_radius(krig_radius / 2)
  {
    PRINT2(highpass_radius, krig_radius);
    highpass1.zero();
  }
  virtual ~OpticalFlow ()
  {
    PRINT(max_flow);
  }

  virtual void transform (Vector<float> & image)
  {
    float dt = 1.0f / DEFAULT_VIDEO_FRAMERATE;

    highpass0 = highpass1;
    highpass(width, height, highpass_radius, image, highpass1, iyy);
    local_optical_flow(
        width, height,
        highpass0, highpass1,
        sx, sy,
        ixx, ixy, iyy);
    krig_optical_flow(
        width, height, krig_radius, dt,
        sx, sy,
        ixx, ixy, iyy,
        fx, fy);

    max_flow *= 0.999f;
    imax(max_flow, sqrtf(1e-8f + max_norm_squared(fx)));
    imax(max_flow, sqrtf(1e-8f + max_norm_squared(fy)));
    float gain = 1 / max_flow;

    float rx = gain * 1;
    float bx = gain * cos(M_PI / 3);
    float by = gain * sin(M_PI / 3);
    float gx = gain * cos(-M_PI / 3);
    float gy = gain * sin(-M_PI / 3);

    float * restrict x = fx;
    float * restrict y = fy;
    float * restrict r = red;
    float * restrict g = green;
    float * restrict b = blue;

    for (size_t ij = 0, IJ = size; ij < IJ; ++ij) {
      r[ij] = max(0.0f, rx * x[ij]);
      g[ij] = max(0.0f, gx * x[ij]) + max(0.0f, gy * y[ij]);
      b[ij] = max(0.0f, bx * x[ij]) + max(0.0f, by * y[ij]);
    }
  }
};

//----( main harness )--------------------------------------------------------

const char * help_message =
"Usage: image_test SOURCE TRANSFORM [OPTIONS]"
"\n  Operations image, displays, and saves result to data/test_out.im"
"\n"
"\nSources:"
"\n  camera               Transform live video, save frame 1 to data/test_in.im"
"\n  camera2              1/2 x 1/2 scaled camera"
"\n  camera4              1/4 x 1/4 scaled camera"
"\n  color                Transform color video (yuyv), save first frame"
"\n  color2               1/2 x 1/2 scaled color camera"
"\n  color4               1/4 x 1/4 scaled color camera"
"\n  region               Transform masked region of video, save first frame"
"\n  disk                 Transform disk region of video, save first frame"
"\n  FILENAME             Transform an image file"
"\n"
"\nOperations:"
"\n  wire                 No transform"
"\n  transpose            Transpose image"
"\n  square [RAD]         Square blur at given radius"
"\n  quad [RAD]           Quadratic blur at given radius"
"\n  exp [RAD]            Exponential blur at given radius"
"\n  dilate [SHARPNESS]   Dilate at given sharpness : (-inf,inf)"
"\n  sharpen [RAD]        Highpass filter at given scale"
"\n  hdr [RAD] [RANGE]    High dynamic range transform"
"\n  point [RAD]          Enhance points of given size"
"\n  point(2,4,8)         Enhance points of size 2,4,8"
"\n  cross [RAD]          Enhance crosses of given size"
"\n  line [RAD]           Enhance lines over regions of given size"
"\n  finger(2,4,8)        Enhance finger tips + shafts of size 2,4,8"
"\n  orient(2,4,8)        Orient fingers of size 2,4,8"
"\n  hand(2,4,8)          Enhance tips + shafts + palms of size 2,4,8"
"\n  change [TIMESCALE]   Statistical change detection"
"\n  change-x [TIMESCALE] Vertical momentum of statistical change"
"\n  sand [RAD]           Quad blur; Enhance points"
"\n  sand4                Quad blur; Enhance points"
"\n  loflow               Local ptical flow"
"\n  lofp                 Local ptical flow of image pyramid"
"\n  flow                 Optical flow"
;

Source * g_source;

template<class Op>
void run_source_op (Args & args)
{
  size_t width = g_source->width();
  size_t height = g_source->height();
  PRINT2(width, height);

  Op op(width, height, args);

  g_source->display(op);

  delete g_source;
}

void run_source (Args & args)
{
  args
    .case_("wire", run_source_op<Operation>)
    .case_("transpose", run_source_op<Transpose>)
    .case_("square", run_source_op<SquareBlur>)
    .case_("square_wrap", run_source_op<SquareBlurWrap>)
    .case_("quad", run_source_op<QuadBlur>)
    .case_("exp", run_source_op<ExpBlur>)
    .case_("dilate", run_source_op<Dilate>)
    .case_("sharpen", run_source_op<Sharpen>)
    .case_("hdr", run_source_op<Hdr>)
    .case_("point", run_source_op<EnhancePoints>)
    .case_("point2", run_source_op<EnhancePoints_<2> >)
    .case_("point4", run_source_op<EnhancePoints_<4> >)
    .case_("point8", run_source_op<EnhancePoints_<8> >)
    .case_("point248", run_source_op<EnhancePoints_248>)
    .case_("cross", run_source_op<EnhanceCrosses>)
    .case_("line", run_source_op<EnhanceLines>)
    .case_("finger2", run_source_op<EnhanceFingers_<2> >)
    .case_("finger4", run_source_op<EnhanceFingers_<4> >)
    .case_("finger8", run_source_op<EnhanceFingers_<8> >)
    .case_("orient2", run_source_op<OrientFingers_<2> >)
    .case_("orient4", run_source_op<OrientFingers_<4> >)
    .case_("orient8", run_source_op<OrientFingers_<8> >)
    .case_("hand2", run_source_op<EnhanceHands_<2> >)
    .case_("hand4", run_source_op<EnhanceHands_<4> >)
    .case_("hand8", run_source_op<EnhanceHands_<8> >)
    .case_("change", run_source_op<DetectChanges>)
    .case_("change-x", run_source_op<HorizChange>)
    .case_("sand", run_source_op<Sand>)
    .case_("sand2", run_source_op<Sand_<2> >)
    .case_("sand4", run_source_op<Sand_<4> >)
    .case_("sand8", run_source_op<Sand_<8> >)
    .case_("loflow", run_source_op<LocalOpticalFlow>)
    .case_("lofp", run_source_op<LocalOpticalFlowPyramid>)
    .case_("flow", run_source_op<OpticalFlow>)
    .default_error();
}

template<class Src, class Cam, size_t shrink_exponent>
void run (Args & args)
{
  VideoSource * cam = new Cam();
  for (size_t s = shrink_exponent; s; --s) {
    cam = new ShrinkToHalf(cam);
  }
  g_source = new Src(cam);

  run_source(args);
}

void run_file (Args & args)
{
  g_source = new FileSource(args.pop());

  run_source(args);
}

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  args
    .case_("camera", run<LiveSource, Camera, 0>)
    .case_("camera2", run<LiveSource, Camera, 1>)
    .case_("camera4", run<LiveSource, Camera, 2>)
    .case_("color", run<LiveSource, ColorCamera, 0>)
    .case_("color2", run<LiveSource, ColorCamera, 1>)
    .case_("color4", run<LiveSource, ColorCamera, 2>)
    .case_("region", run<LiveSource, CameraRegion, 2>)
    .case_("disk", run<LiveSource, CameraDisk, 2>)
    .default_(run_file);

  return 0;
}

