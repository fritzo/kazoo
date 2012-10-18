
#include "streaming_video.h"
#include "images.h"
#include <algorithm>

namespace Streaming
{

//----( mosaic )--------------------------------------------------------------

//----( Mono )----

Pushed<MonoImage> & Mosaic::in (size_t i)
{
  ASSERT_LT(i, m_images.size());
  return * m_images[i];
}

Mosaic::Mosaic (Rectangle in, size_t size)
  : m_images(size, NULL),
    shape_in(in),
    shape_out(size * in.width(), in.height())
{
  for (size_t i = 0; i < m_images.size(); ++i) {
    m_images[i] = new Shared<MonoImage, size_t>(in.size());
  }
}

Mosaic::~Mosaic ()
{
  for (size_t i = 0; i < m_images.size(); ++i) {
    delete m_images[i];
  }
}

void Mosaic::pull (Seconds time, MonoImage & image)
{
  ASSERT_SIZE(image, shape_out.size());
  for (size_t i = 0; i < m_images.size(); ++i) {
    MonoImage im = image.block(shape_in.size(), i);
    m_images[i]->pull(time, im);
  }
}

//----( Yuv420p8 )----

Pushed<Yuv420p8Image> & Yuv420p8Mosaic::in (size_t i)
{
  ASSERT_LT(i, m_images.size());
  return * m_images[i];
}

Yuv420p8Mosaic::Yuv420p8Mosaic (Rectangle in, size_t size)
  : m_images(size, NULL),
    m_block(in.size()),
    shape_in(in),
    shape_out(size * in.width(), in.height())
{
  for (size_t i = 0; i < m_images.size(); ++i) {
    m_images[i] = new Shared<Yuv420p8Image, size_t>(in.size());
  }
}

Yuv420p8Mosaic::~Yuv420p8Mosaic ()
{
  for (size_t i = 0; i < m_images.size(); ++i) {
    delete m_images[i];
  }
}

void Yuv420p8Mosaic::pull (Seconds time, Yuv420p8Image & image)
{
  ASSERT_SIZE(image, shape_out.size() * Yuv420p8Image::num_channels);
  for (size_t i = 0; i < m_images.size(); ++i) {

    m_images[i]->pull(time, m_block);

    image.y.block(shape_in.size() / 1, i) = m_block.y;
    image.u.block(shape_in.size() / 4, i) = m_block.u;
    image.v.block(shape_in.size() / 4, i) = m_block.v;
  }
}

//----( filters )-------------------------------------------------------------

void Transpose::filter (
    Seconds,
    const MonoImage & image,
    MonoImage & transposed)
{
  Image::transpose(in.width(), in.height(), image, transposed);
}

//----( projection )----

ProjectAxes::ProjectAxes (Rectangle shape)
  : Rectangle(shape),

    m_image_x(width()),
    m_image_y(height()),

    x_out("ProjectAxes.x_out", width()),
    y_out("ProjectAxes.y_out", height())
{}

void ProjectAxes::push (Seconds time, const MonoImage & image)
{
  ASSERT_SIZE(image, size());

  Image::project_axes_sum(width(), height(), image, m_image_x, m_image_y);

  ASSERT(x_out or y_out, "neither x_out nor y_out were set");
  if (x_out) x_out.push(time, m_image_x);
  if (y_out) y_out.push(time, m_image_y);
}

//----( sliders )----

SliderColumns::SliderColumns (Rectangle shape_in, size_t size_out)
  : Rectangle(shape_in),

    m_spline(shape_in.width(), size_out),
    m_y(shape_in.height()),
    m_sum_m(shape_in.width()),
    m_sum_my(shape_in.width()),
    m_sliders(size_out),

    out("SliderArray.out")
{
  for (size_t i = 0, I = shape_in.height(); i < I; ++i) {
    float y = (i + 0.5f) / I;
    m_y[i] = y;
  }
}

void SliderColumns::push (Seconds time, const MonoImage & image)
{
  ASSERT_SIZE(image, size());

  Image::moments_along_x(width(), height(), image, m_y, m_sum_m, m_sum_my);

  m_spline.transform_fwd(m_sum_m, m_sliders.mass);
  m_spline.transform_fwd(m_sum_my, m_sliders.position);

  float * restrict mass = m_sliders.mass;
  float * restrict pos = m_sliders.position;
  for (size_t i = 0, I = m_sliders.mass.size; i < I; ++i) {
    imax(mass[i], 1e-8f);
    pos[i] /= mass[i];
  }

  out.push(time, m_sliders);
}

//----( resizing )----

void ShrinkBy2::filter (Seconds, const MonoImage & image, MonoImage & shrunk)
{
  Image::scale_by_half(in.width(), in.height(), image, shrunk);
}

void ShrinkBy4::filter (Seconds, const MonoImage & image, MonoImage & shrunk)
{
  Image::scale_by_half(in.width(), in.height(), image, m_shrunk2);
  Image::scale_by_half(in.width() / 2, in.height() / 2, m_shrunk2, shrunk);
}

void Shrink::filter (Seconds time, const MonoImage & image, MonoImage & shrunk)
{
  m_spline.transform_fwd(image, shrunk);
}

void ZoomMono::filter (
    Seconds time,
    const MonoImage & image,
    MonoImage & zoomed)
{
  m_spline.transform_bwd(image, zoomed);
}

void ZoomMono::pull (Seconds time, MonoImage & zoomed)
{
  ASSERT_SIZE(zoomed, out.size());

  if ((out.width() == in.width()) and (out.height() == in.height())) {

    in.pull(time, zoomed);

  } else {

    in.pull(time, m_image);
    m_spline.transform_bwd(m_image, zoomed);
  }
}

void ZoomRgb::filter (Seconds time, const RgbImage & image, RgbImage & zoomed)
{
  m_spline.transform_bwd(image.red, zoomed.red);
  m_spline.transform_bwd(image.green, zoomed.green);
  m_spline.transform_bwd(image.blue, zoomed.blue);
}

void ZoomRgb::pull (Seconds time, RgbImage & zoomed)
{
  ASSERT_SIZE(zoomed.red, out.size());

  if ((out.width() == in.width()) and (out.height() == in.height())) {

    in.pull(time, zoomed);

  } else {

    in.pull(time, m_image);
    m_spline.transform_bwd(m_image.red, zoomed.red);
    m_spline.transform_bwd(m_image.green, zoomed.green);
    m_spline.transform_bwd(m_image.blue, zoomed.blue);
  }
}

void ZoomMono8::filter (
    Seconds time,
    const Mono8Image & image,
    Mono8Image & zoomed)
{
  const size_t X = in.width();
  const size_t Y = in.height();

  const size_t I = out.width();
  const size_t J = out.height();

  const uint8_t * restrict im = image;
  uint8_t * restrict zm = zoomed;

  for (size_t i = 0; i < I; ++i) {
    size_t x = i * X / I;

    for (size_t j = 0; j < J; ++j) {
      size_t y = j * Y / J;

      zm[J * i + j] = im[Y * x + y];
    }
  }
}

void ZoomRgb8::filter (
    Seconds time,
    const Rgb8Image & image,
    Rgb8Image & zoomed)
{
  const size_t X = in.width();
  const size_t Y = in.height();

  const size_t I = out.width();
  const size_t J = out.height();

  const uint8_t * restrict ir = image.red;
  const uint8_t * restrict ig = image.green;
  const uint8_t * restrict ib = image.blue;
  uint8_t * restrict zr = zoomed.red;
  uint8_t * restrict zg = zoomed.green;
  uint8_t * restrict zb = zoomed.blue;

  for (size_t i = 0; i < I; ++i) {
    size_t x = i * X / I;

    for (size_t j = 0; j < J; ++j) {
      size_t y = j * Y / J;

      size_t ij = J * i + j;
      size_t xy = Y * x + y;

      zr[ij] = ir[xy];
      zg[ij] = ig[xy];
      zb[ij] = ib[xy];
    }
  }
}

//----( blurring )----

SquareBlur::SquareBlur (Rectangle shape_in, size_t blur_radius)
  : Filter<MonoImage>(
      string("SquareBlur"),
      shape_in,
      shape_in.transposed()),
    m_blur_radius(blur_radius)
{}

void SquareBlur::filter (
    Seconds time,
    const MonoImage & const_image,
    MonoImage & blurred)
{
  const size_t R = m_blur_radius;
  const size_t X = out.width();
  const size_t Y = out.height();

  // WARNING HACK this uses input as working memory
  MonoImage & image = const_cast<MonoImage &>(const_image);

  Image::square_blur_axis(Y,X,R, image, blurred);
  Image::transpose(Y,X, blurred, image);
  Image::square_blur_axis(X,Y,R, image, blurred);
}

QuadraticBlur::QuadraticBlur (Rectangle shape_in, size_t blur_radius)
  : Filter<MonoImage>(
      string("QuadraticBlur"),
      shape_in,
      shape_in.transposed()),
    m_blur_radius(blur_radius)
{}

void QuadraticBlur::filter (
    Seconds time,
    const MonoImage & const_image,
    MonoImage & blurred)
{
  const size_t R = m_blur_radius;
  const size_t X = out.width();
  const size_t Y = out.height();

  // WARNING HACK this uses input as working memory
  MonoImage & image = const_cast<MonoImage &>(const_image);

  Image::quadratic_blur_axis(Y,X,R, image, blurred);
  Image::transpose(Y,X, blurred, image);
  Image::quadratic_blur_axis(X,Y,R, image, blurred);
}

ImageHighpass::ImageHighpass (Rectangle shape, size_t blur_radius)
  : Filter<MonoImage>(string("ImageHighpass"), shape),
    m_blur_radius(blur_radius),
    m_temp(shape.size())
{}

void ImageHighpass::filter (
    Seconds time,
    const MonoImage & image,
    MonoImage & highpass)
{
  const size_t R = m_blur_radius;
  const size_t X = in.width();
  const size_t Y = in.height();

  highpass = image;
  Image::quadratic_blur(X,Y,R, highpass, m_temp);

  const float * restrict im = image;
  float * restrict hi = highpass;
  const float scale = powf(R+1+R, -6);

  for (size_t xy = 0; xy < X * Y; ++xy) {
    hi[xy] = im[xy] - scale * hi[xy];
  }
}

//----( masking )----

DiskMask::DiskMask (Rectangle shape)
  : Rectangle(shape),
    m_mask(size()),
    out("DiskMask.out", shape)
{
  const size_t I = width();
  const size_t J = height();
  float * restrict mask = m_mask;

  for (size_t i = 0; i < I; ++i) {
    float x = (i + 0.5f) / I * 2 - 1;

    for (size_t j = 0; j < J; ++j) {
      float y = (j + 0.5f) / J * 2 - 1;
      float r2 = sqr(x) + sqr(y);

      mask[J * i + j] = bound_to(0.0f, 1.0f, 1 - r2);
    }
  }
}

void DiskMask::push (Seconds time, const MonoImage & const_image)
{
  // WARNING HACK this uses input as working memory
  MonoImage & image = const_cast<MonoImage &>(const_image);

  image *= m_mask;

  out.push(time, image);
}

//----( feature enhancement )----

EnhancePoints::EnhancePoints (Rectangle shape_in, size_t blur_radius)
  : Filter<MonoImage>(
      string("EnhancePoints"),
      shape_in,
      shape_in.transposed()),

    m_blur_radius(blur_radius),
    m_timescale(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_VIDEO_FRAMERATE),
    m_gain(m_timescale)
{}

void EnhancePoints::filter (
    Seconds time,
    const MonoImage & const_image,
    MonoImage & points)
{
  const size_t R = m_blur_radius;
  const size_t X = out.width();
  const size_t Y = out.height();

  // WARNING HACK this uses input as working memory
  MonoImage & image = const_cast<MonoImage &>(const_image);

  Image::quadratic_blur_axis(Y,X,R, image, points);
  Image::transpose(Y,X, points, image);
  Image::quadratic_blur_axis(X,Y,R, image, points);
  Image::enhance_points(X,Y,R, points, image);

  float gain = m_gain.update(max(image));
  multiply(gain, image, points);
}

EnhanceFingers::EnhanceFingers (Rectangle shape, size_t blur_radius)
  : Filter<MonoImage, MomentImage>(string("EnhanceFingers"), shape),

    m_blur_radius(blur_radius),
    m_timescale(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_VIDEO_FRAMERATE),

    m_tips_gain(m_timescale),
    m_grad_gain(m_timescale),

    m_temp(in.size())
{}

void EnhanceFingers::filter (
    Seconds time,
    const MonoImage & const_image,
    MomentImage & moments)
{
  TODO("git show c21413e:src/controllers.o | grep -A60 PointerThread::process");
  /*
  const size_t R = m_blur_radius;
  const size_t X = width();
  const size_t Y = height();

  float * restrict dx = moments.dx;
  float * restrict dy = moments.dy;
  float * restrict tips = moments.mass;

  // WARNING HACK this uses input as working memory
  MonoImage & image = const_cast<MonoImage &>(const_image);

  LOG1("blur");
  MonoImage & temp0 = m_dx;
  Image::quadratic_blur_axis(Y,X,R, image, temp0);
  Image::transpose(Y,X, temp0, image);
  Image::quadratic_blur_axis(X,Y,R, image, temp0);

  LOG1("find fingers & orientation");
  Image::enhance_fingers_<R>(X,Y, temp0, m_tips, m_shafts);

  MonoImage & temp1 = image;
  Image::local_moments_transpose(X, Y, 3*R, m_shafts, temp_dx, temp_dy, temp1);

  LOG1("deal with transposes");
  MonoImage & temp2 = m_shafts;
  Image::transpose(Y,X, temp_dx, dy);
  Image::transpose(Y,X, temp_dy, dx);

  LOG1("update gain");
  float max_tips = 0;
  float max_norm = 0;

  for (size_t ij = 0; ij < X*Y; ++ij) {
    imax(max_tips, tips[ij]);
    imax(max_norm, sqr(dx[ij]) + sqr(dy[ij]));
  }

  float tips_gain = m_tips_gain.update(max_tips);
  float grad_gain = m_grad_gain.update(sqrt(max_norm));

  for (size_t ij = 0; ij < X*Y; ++ij) {
    tips[ij] *= tips_gain;
    dx[ij] *= grad_gain;
    dy[ij] *= grad_gain;
  }
  */
}

EnhanceHands::EnhanceHands (
    Rectangle shape_in,
    size_t blur_radius,
    size_t hand_radius)

  : Filter<MonoImage, HandImage>(
      string("EnhanceHands"),
      shape_in,
      shape_in.transposed()),

    m_blur_radius(blur_radius),
    m_hand_radius(hand_radius),
    m_timescale(DEFAULT_GAIN_TIMESCALE_SEC * DEFAULT_VIDEO_FRAMERATE),

    m_tip_gain(m_timescale),
    m_shaft_gain(m_timescale),
    m_palm_gain(m_timescale),

    m_blurred(in.size())
{}

EnhanceHands::~EnhanceHands ()
{
  float tip_gain = m_tip_gain;
  float shaft_gain = m_shaft_gain;
  float palm_gain = m_palm_gain;
  PRINT3(tip_gain, shaft_gain, palm_gain);
}

void EnhanceHands::filter (
    Seconds time,
    const MonoImage & const_image,
    HandImage & hands)
{
  const size_t R = m_blur_radius;
  const size_t H = m_hand_radius;
  const size_t I = out.width();
  const size_t J = out.height();

  // WARNING HACK this uses input as working memory
  MonoImage & image = const_cast<MonoImage &>(const_image);

  Image::quadratic_blur_axis(J,I,R, image, m_blurred);
  Image::transpose(J,I, m_blurred, image);
  Image::quadratic_blur_axis(I,J,R, image, m_blurred);

  Vector<float> & temp1 = hands.tip;
  Vector<float> & temp2 = hands.shaft;
  Image::scale_by_half(I, J, m_blurred, temp1);
  Image::scale_by_half(I/2, J/2, temp1, temp2);
  Image::enhance_points(I/4, J/4, H, temp2, hands.palm);
  ipow(hands.palm, 0.25f);
  float palm_gain = m_palm_gain.update(max(hands.palm));
  palm_gain *= pow(1+1+1, -4); // to account for scale_smoothly_by_four_8
  hands.palm *= palm_gain;
  Image::scale_smoothly_by_four(I/4, J/4, hands.palm, hands.palm, temp2);

  Image::enhance_fingers(I, J, R, m_blurred, hands.tip, hands.shaft);
  hands.tip *= m_tip_gain.update(max(hands.tip));
  hands.shaft *= m_shaft_gain.update(max(hands.shaft));
}

void HandsToColor::push (Seconds time, const HandImage & hands)
{
  out.push(time, reinterpret_cast<const RgbImage &>(hands));
}

void HandsToColor::pull (Seconds time, RgbImage & rgb)
{
  in.pull(time, reinterpret_cast<HandImage &>(rgb));
}

void HandsToColor::filter (
    Seconds time,
    const HandImage & hands,
    RgbImage & rgb)
{
  rgb = hands;
}

//----( moments )----

void ExtractMoments::push (Seconds time, const MonoImage & image)
{
  ASSERT_SIZE(image, size());

  Image::extract_moments(
      width(),
      height(),
      image,
      m_moments.z,
      m_moments.x,
      m_moments.y);

  m_moments.z *= m_gain.update(m_moments.z);

  out.push(time, m_moments);
}

//----( reassignment )----

ReassignAccum::ReassignAccum (Rectangle shape, float timescale)
  : Filter<MonoImage>("ReassignAccum", shape),
    m_decay(expf(-1 / timescale)),
    m_accum(in.size())
{
  ASSERT_LT(0, timescale);

  m_accum.set(1e-20f);
}

void ReassignAccum::filter (
    Seconds time,
    const MonoImage & dmass,
    MonoImage & reas)
{
  m_accum += dmass;

  Image::reassign_wrap_repeat(in.width(), in.height(), m_accum, reas, m_decay);

  m_accum = reas;
}

AttractRepelAccum::AttractRepelAccum (Rectangle shape, float timescale)
  : Filter<MonoImage>("AttractRepelAccum", shape),
    m_decay(expf(-1 / timescale)),
    m_accum(in.size()),
    m_dx(in.size()),
    m_dy(in.size()),
    m_blur(in.size())
{
  ASSERT_LT(0, timescale);

  m_accum.set(1e-20f);
}

void AttractRepelAccum::filter (
    Seconds time,
    const MonoImage & dmass,
    MonoImage & reas)
{
  enum { R1 = 2, R2 = 8 };
  const size_t I = in.width();
  const size_t J = in.height();
  const float decay = m_decay;

  const float * restrict dm = dmass;
  float * restrict accum = m_accum;
  float * restrict dx = m_dx;
  float * restrict dy = m_dy;
  float * restrict blur = m_blur;
  float * restrict re = reas;

  const float blur_scale = powf(R2 + 1 + R2, -4);
  for (size_t ij = 0; ij < I*J; ++ij) {
    float a = decay * accum[ij] + dm[ij];
    accum[ij] = a;
    blur[ij] = -blur_scale * a;
  }

  // this is a mexican hat
  Image::linear_blur_axis_wrap(I, J, R2, blur, dx);
  Image::transpose(I, J, blur, dx);
  Image::linear_blur_axis(J, I, R2, dx, blur);
  m_dx += m_accum;
  Image::linear_blur(J, I, R1, dx, blur);
  Image::transpose(J, I, dx, blur);
  Image::linear_blur_axis_wrap(I, J, R1, blur, dx);
  m_blur *= powf(max(1e-16f, max_norm_squared(m_blur)), -0.5f);

  Image::gradient_wrap_repeat(I, J, blur, dx, dy);
  // bounding is unnecessary after normalization
  //for (size_t ij = 0; ij < I*J; ++ij) {
  //  dx[ij] = bound_to(-1.0f, 1.0f, dx[ij]);
  //  dy[ij] = bound_to(-1.0f, 1.0f, dy[ij]);
  //}

  Image::reassign_flow_wrap_repeat(I, J, dx, dy, accum, re);

  m_accum = reas;
}

//----( optical flow )----

OpticalFlow::OpticalFlow (Rectangle shape, size_t highpass_radius)

  : Filter<MonoImage, FlowImage>("OpticalFlow", shape),

    m_highpass_radius(highpass_radius),

    m_highpass0(shape.size()),
    m_highpass1(shape.size()),
    m_temp1(shape.size()),
    m_temp2(shape.size())
{
  m_highpass1.zero();
}

void OpticalFlow::filter (
    Seconds time,
    const MonoImage & image,
    FlowImage & flow)
{
  m_highpass0 = m_highpass1;

  Image::highpass(
      in.width(),
      in.height(),
      m_highpass_radius,
      image,
      m_highpass1,
      m_temp1);

  Image::local_optical_flow(
      in.width(), in.height(),
      m_highpass0, m_highpass1,
      flow.dx, flow.dy,
      m_temp1, m_temp2);
}

KrigOpticalFlow::KrigOpticalFlow (
    Rectangle shape_in,
    float spacescale,
    size_t highpass_radius,
    float prior)

  : Filter<MonoImage, FlowImage>(
      "KrigOpticalFlow",
      shape_in,
      shape_in.scaled(0.5)),

    m_highpass_radius(highpass_radius),
    m_spacescale(spacescale),
    m_prior(prior),

    m_highpass0(in.size()),
    m_highpass1(in.size()),
    m_flow_full(in.size()),
    m_flow_half(out.size()),

    m_time(Seconds::now())
{
  m_highpass1.zero();
}

KrigOpticalFlow::~KrigOpticalFlow ()
{
  PRINT2(rms(m_highpass0), rms(m_highpass1));
  FlowInfoImage & flow = m_flow_half;
  PRINT2(rms(flow.surprise_x), rms(flow.surprise_x));
  PRINT3(rms(flow.info_xx), rms(flow.info_xy), rms(flow.info_yy));
}

void KrigOpticalFlow::filter (
    Seconds time,
    const MonoImage & image,
    FlowImage & flow)
{
  const size_t I = in.width();
  const size_t J = in.height();
  float prior_info = sqr(m_prior);

  float dt = max(1e-8f, time - m_time);
  m_time = time;

  m_highpass0 = m_highpass1;
  Vector<float> & temp_full = m_flow_full.surprise_x;
  Image::highpass(I, J, m_highpass_radius, image, m_highpass1, temp_full);
  m_highpass1 *= 1.0f / 255;

  Image::local_optical_flow(
      I, J,
      m_highpass0,
      m_highpass1,
      m_flow_full.surprise_x,
      m_flow_full.surprise_y,
      m_flow_full.info_xx,
      m_flow_full.info_xy,
      m_flow_full.info_yy);

  size_t I_flow = I * FlowInfoImage::num_channels;
  Image::scale_by_half(I_flow, J, m_flow_full.data, m_flow_half.data);
  prior_info *= 4; // to account for scaling

  Image::krig_optical_flow(
      I/2, J/2,
      m_spacescale,
      dt,
      m_flow_half.surprise_x,
      m_flow_half.surprise_y,
      m_flow_half.info_xx,
      m_flow_half.info_xy,
      m_flow_half.info_yy,
      flow.dx,
      flow.dy,
      prior_info);
}

GlovesFlow::GlovesFlow (
    Rectangle shape_in,
    float spacescale,
    float lowpass_radius,
    float prior)

  : Filter<MonoImage, GlovesImage>(
      "GlovesFlow",
      shape_in,
      shape_in),

    m_lowpass_radius(lowpass_radius),
    m_spacescale(spacescale),
    m_prior(prior),

    m_lowpass0(in.size()),
    m_lowpass1(in.size()),
    m_flow_full(in.size()),
    m_flow_half(out.size()),

    m_time(Seconds::now())
{
  m_lowpass1.zero();
}

GlovesFlow::~GlovesFlow ()
{
  PRINT2(rms(m_lowpass0), rms(m_lowpass1));
  FlowInfoImage & flow = m_flow_half;
  PRINT2(rms(flow.surprise_x), rms(flow.surprise_x));
  PRINT3(rms(flow.info_xx), rms(flow.info_xy), rms(flow.info_yy));
}

void GlovesFlow::filter (
    Seconds time,
    const MonoImage & image,
    GlovesImage & yuv)
{
  const size_t I = in.width();
  const size_t J = in.height();
  float prior_info = sqr(m_prior);

  multiply(1.0f / 255, image, yuv.y);

  float dt = max(1e-8f, time - m_time);
  m_time = time;

  // we could save a transpose here, at elegance cost
  m_lowpass0 = m_lowpass1;
  m_lowpass1 = yuv.y;
  Vector<float> & temp_full = m_flow_full.surprise_x;
  Image::exp_blur_zero(I, J, m_lowpass_radius, m_lowpass1, temp_full);

  Image::local_optical_flow(
      I, J,
      m_lowpass0,
      m_lowpass1,
      m_flow_full.surprise_x,
      m_flow_full.surprise_y,
      m_flow_full.info_xx,
      m_flow_full.info_xy,
      m_flow_full.info_yy);

  size_t I_flow = I * FlowInfoImage::num_channels;
  Image::scale_by_half(I_flow, J, m_flow_full.data, m_flow_half.data);
  prior_info *= 4; // to account for scaling

  Image::krig_optical_flow(
      I/2, J/2,
      m_spacescale,
      dt,
      m_flow_half.surprise_x,
      m_flow_half.surprise_y,
      m_flow_half.info_xx,
      m_flow_half.info_xy,
      m_flow_half.info_yy,
      yuv.u,
      yuv.v,
      prior_info);
}

FilterOpticalFlow::FilterOpticalFlow (
    Rectangle shape_in,
    float process_noise,
    size_t highpass_radius,
    float prior)

  : Filter<MonoImage, FlowImage>(
      "FilterOpticalFlow",
      shape_in,
      shape_in.scaled(0.5)),

    m_highpass_radius(highpass_radius),
    m_process_noise(process_noise),
    m_prior(prior),

    m_highpass0(in.size()),
    m_highpass1(in.size()),
    m_flow_full(in.size()),
    m_flow_half(out.size()),
    m_flow_old(out.size()),

    m_time(Seconds::now())
{
  ASSERT_LT(0, process_noise);

  m_highpass1.zero();

  m_flow_half.surprise_x.zero();
  m_flow_half.surprise_y.zero();
  m_flow_half.info_xx.set(0.1f / process_noise);
  m_flow_half.info_xy.zero();
  m_flow_half.info_yy.set(0.1f / process_noise);

  m_filtered.zero();
}

FilterOpticalFlow::~FilterOpticalFlow ()
{
  PRINT2(rms(m_highpass0), rms(m_highpass1));
  FlowInfoImage & flow = m_flow_half;
  PRINT2(rms(flow.surprise_x), rms(flow.surprise_x));
  PRINT3(rms(flow.info_xx), rms(flow.info_xy), rms(flow.info_yy));
}

void FilterOpticalFlow::filter (
    Seconds time,
    const MonoImage & image,
    FlowImage & flow)
{
  const size_t I = in.width();
  const size_t J = in.height();
  float prior_info = sqr(m_prior);

  float dt = max(1e-8f, time - m_time);
  m_time = time;

  Image::advance_optical_flow(
      I/2, J/2,
      dt,
      flow.dx,
      flow.dy,
      m_flow_half.surprise_x,
      m_flow_half.surprise_y,
      m_flow_half.info_xx,
      m_flow_half.info_xy,
      m_flow_half.info_yy,
      m_flow_old.surprise_x,
      m_flow_old.surprise_y,
      m_flow_old.info_xx,
      m_flow_old.info_xy,
      m_flow_old.info_yy);

  m_highpass0 = m_highpass1;
  Vector<float> & temp_full = m_flow_full.surprise_x;
  Image::highpass(I, J, m_highpass_radius, image, m_highpass1, temp_full);
  m_highpass1 *= 1.0f / 255;

  Image::local_optical_flow(
      I, J,
      m_highpass0,
      m_highpass1,
      m_flow_full.surprise_x,
      m_flow_full.surprise_y,
      m_flow_full.info_xx,
      m_flow_full.info_xy,
      m_flow_full.info_yy);

  Image::scale_by_half(5 * I, J, m_flow_full.data, m_flow_half.data);
  prior_info *= 4; // to account for scaling

  Image::fuse_optical_flow(
      I * J / 4,
      dt,
      m_process_noise,
      prior_info,
      m_flow_old.surprise_x,
      m_flow_old.surprise_y,
      m_flow_old.info_xx,
      m_flow_old.info_xy,
      m_flow_old.info_yy,
      m_flow_half.surprise_x,
      m_flow_half.surprise_y,
      m_flow_half.info_xx,
      m_flow_half.info_xy,
      m_flow_half.info_yy,
      flow.dx,
      flow.dy);
}

FlowToColor::FlowToColor (Rectangle shape)
  : Filter<FlowImage, RgbImage>("FlowToColor", shape),
    m_time(Seconds::now()),
    m_max_flow(1e-20f)
{}

void FlowToColor::filter (Seconds time, const FlowImage & flow, RgbImage & rgb)
{
  float dt = max(1e-8f, time - m_time);
  m_time = time;

  m_max_flow *= expf(-dt / DEFAULT_GAIN_TIMESCALE_SEC);

  imax(m_max_flow, sqrtf(1e-8f + max_norm_squared(flow.dx)));
  imax(m_max_flow, sqrtf(1e-8f + max_norm_squared(flow.dy)));
  float gain = 1 / m_max_flow;

  float rx = gain * 1;
  float bx = gain * cos(2 * M_PI / 3);
  float by = gain * sin(2 * M_PI / 3);
  float gx = gain * cos(-2 * M_PI / 3);
  float gy = gain * sin(-2 * M_PI / 3);

  const float * restrict x = flow.dx;
  const float * restrict y = flow.dy;
  float * restrict r = rgb.red;
  float * restrict g = rgb.green;
  float * restrict b = rgb.blue;

  for (size_t ij = 0, IJ = flow.dx.size; ij < IJ; ++ij) {
    r[ij] = max(0.0f, rx * x[ij]);
    g[ij] = max(0.0f, gx * x[ij] + gy * y[ij]);
    b[ij] = max(0.0f, bx * x[ij] + by * y[ij]);
  }
}

GlovesToColor::GlovesToColor (Rectangle shape)
  : Filter<GlovesImage, RgbImage>("GlovesToColor", shape),
    m_flow_to_color(shape.scaled(0.5)),
    m_flow_half(m_flow_to_color.in.size()),
    m_rgb_half(m_flow_to_color.in.size())
{}

void GlovesToColor::filter (
    Seconds time,
    const GlovesImage & gloves,
    RgbImage & rgb)
{
  const size_t I = in.width();
  const size_t J = in.height();

  m_flow_half.dx = gloves.u;
  m_flow_half.dy = gloves.v;
  m_flow_to_color.filter(time, m_flow_half, m_rgb_half);

  Image::scale_by_two(I/2, J/2, m_rgb_half.red, rgb.red);
  Image::scale_by_two(I/2, J/2, m_rgb_half.green, rgb.green);
  Image::scale_by_two(I/2, J/2, m_rgb_half.blue, rgb.blue);

  imax(rgb.red, gloves.y);
  imax(rgb.green, gloves.y);
  imax(rgb.blue, gloves.y);
}

//----( change detector )----

ChangeFilter::ChangeFilter (Rectangle shape)
  : Rectangle(shape),

    m_uv_scale(3.0f),

    m_change_timescale(DEFAULT_CHANGE_TIMESCALE_SEC),
    m_lowpass_timescale(2.0f / DEFAULT_VIDEO_FRAMERATE),

    m_time(Seconds::now()),

    m_mean(size() * 2),
    m_variance(size() * 2),
    m_lowpass(size()),

    out("ChangeFilter.out", shape)
{
  m_mean.set(127.0f);
  m_variance.set(127.0f);
  m_lowpass.zero();
}

ChangeFilter::~ChangeFilter ()
{
  PRINT3(rms(m_mean), rms(m_variance), mean(m_lowpass));

  Vector<float> y_variance(size(), m_variance.begin());
  Vector<float> u_variance(size() / 2, y_variance.end());
  Vector<float> v_variance(size() / 2, u_variance.end());
  PRINT3(mean(y_variance), mean(u_variance), mean(v_variance));
}

void ChangeFilter::push (Seconds time, const YyuvImage & const_image)
{
  ASSERT_SIZE(const_image.yy, size());

  // WARNING HACK this uses input as working memory
  YyuvImage & image = const_cast<YyuvImage &>(const_image);

  float dt = time - m_time;
  m_time = time;

  Image::detect_changes(
      image.size,
      image.data,
      m_mean,
      m_variance,
      dt / m_change_timescale);

  const size_t I = size();
  const size_t I2 = I / 2;

  float * restrict yy = image.yy;
  float * restrict u = image.u;
  float * restrict v = image.v;
  float uv_scale = m_uv_scale;

  // note that image is transposed here
  for (size_t i = 0; i < I2; ++i) {
    float uv = uv_scale * (u[i] + v[i]);

    yy[2 * i + 1] += uv;
    yy[2 * i + 0] += uv;
  }

  float decay = exp(-dt / m_lowpass_timescale);
  float * restrict lowpass = m_lowpass;

  for (size_t i = 0; i < I; ++i) {
    lowpass[i] = max(decay * lowpass[i], yy[i]);
  }

  out.push(time, m_lowpass);
}

/** TODO adapt this:
HorizChangeThread::HorizChangeThread (
    VideoSource & video,
    Instruments::VectorComponent & instrument)

  : CameraThread(video),
    m_video(video),
    m_instrument(instrument),

    m_timescale(DEFAULT_CHANGE_TIMESCALE_SEC * DEFAULT_VIDEO_FRAMERATE),
    m_time(Seconds::now()),

    m_mean(video.size()),
    m_variance(video.size()),

    m_mass(video.width()),
    m_moment(video.width()),
    m_mass_old(video.width()),
    m_moment_old(video.width()),
    m_momentum(video.width())
{
  m_mean = 0.5f;
  m_variance = 0.5f;
  m_moment_old = 1e-8f;
  m_moment_old.zero();
  m_momentum.zero();
}

void HorizChangeThread::process (Seconds time, MonoImage & image)
{
  const size_t X = m_video.width();
  const size_t Y = m_video.height();

  const float tol = 1e-8f;
  const float dt = max(tol, time - m_time) / m_timescale;
  m_time = time;

  LOG1("detect change momentum along vertical lines");
  Image::detect_change_moment_x(
      Y,X, image, m_mean, m_variance, m_mass, m_moment, dt);
  Image::update_momentum(
      X, m_mass, m_moment, m_mass_old, m_moment_old, m_momentum, dt);

  m_instrument.update(time, m_momentum);

  PROGRESS_TICKER('|');
}
*/

//----( peak detection )----

PeakDetector::PeakDetector (
    Rectangle shape,
    size_t peak_capacity,
    float power,
    float init_gain,
    float min_ratio,
    float gain_timescale)

  : Rectangle(shape),

    m_peak_capacity(peak_capacity),
    m_power(power),
    m_min_ratio(min_ratio),
    m_gain(gain_timescale, init_gain),
    m_peaks(),

    out("PeakDetector.out")
{
  ASSERT_LT(0, power);
}

PeakDetector::~PeakDetector ()
{
  float peak_detector_gain = m_gain;
  PRINT(peak_detector_gain);
}

void PeakDetector::push (Seconds time, const MonoImage & image)
{
  float min_value = (m_gain > 0) ? pow(m_min_ratio / m_gain, 1 / m_power) : 0;

  m_peaks.clear();
  Image::find_peaks(
      width(),
      height(),
      m_peak_capacity,
      min_value,
      image,
      m_peaks);

  if (not m_peaks.empty()) {

    std::sort(m_peaks.begin(), m_peaks.end());
    m_gain.update(pow(m_peaks[0].z, m_power));
    float scale = m_gain;

    for (size_t i = 0; i < m_peaks.size(); ++i) {
      m_peaks[i].z = scale * pow(m_peaks[i].z, m_power);
    }

    PROGRESS_TICKER(' ' << m_peaks.size() << ' ');
  }

  out.push(time, m_peaks);
}

void PeakTransform::push (Seconds time, const Image::Peaks & const_peaks)
{
  // WARNING HACK this uses input as working memory
  Image::Peaks & peaks = const_cast<Image::Peaks &>(const_peaks);

  typedef Image::Peaks::iterator Auto;
  for (Auto i = peaks.begin(); i != peaks.end(); ++i) {
    m_transform(*i);
  }

  out.push(time, const_peaks);
}

MomentsToFinger::MomentsToFinger ()
  : m_time0(Seconds::now()),
    m_time1(Seconds::now()),

    impact_in("MomentsToFinger.impact_in")
{
  m_finger0.clear();
  m_finger1.clear();
}

MomentsToFinger::~MomentsToFinger ()
{
  PRINT(m_finger1.get_energy());
  PRINT3(m_finger1.x(), m_finger1.y(), m_finger1.z());
}

void MomentsToFinger::push (Seconds time, const Image::Peak & moment)
{
  m_mutex.lock();

  float tol = 1e-8f;
  float dt = max(tol, time - m_time1);

  m_time0 = m_time1;
  m_time1 = time;

  float3 pos0 = m_finger0.get_pos();
  float3 pos1 = m_finger1.get_pos();
  float3 vel = (pos1 - pos0) / dt;

  m_finger0 = m_finger1;

  m_finger1.set_vel(vel);
  m_finger1.set_z(moment.z);
  m_finger1.set_x((0.5f - moment.x) * GRID_SIZE_X);
  m_finger1.set_y((0.5f - moment.y) * GRID_SIZE_Y);

  m_mutex.unlock();
}

void MomentsToFinger::pull (Seconds time, Gestures::Finger & finger)
{
  m_mutex.lock();

  float tol = 1e-8f;
  float dt = max(tol, time - m_time1);
  finger = m_finger1.extrapolate(dt);

  if (impact_in) {
    impact_in.pull(time, finger.energy());
  } else {
    finger.set_energy(finger.get_impact());
  }

  m_mutex.unlock();
}

//----( pulled )--------------------------------------------------------------

//----( history )----

void RgbHistory::pull (Seconds, RgbImage & history)
{
  ASSERT_SIZE(history.red, size());

  m_mutex.lock();
  m_history.get(m_transposed);
  m_mutex.unlock();

  const size_t I = width();
  const size_t J = height();
  for (size_t i = 0; i < I; ++i) {
    copy_float(m_transposed + J * (3 * i + 0), history.red + J * i, J);
    copy_float(m_transposed + J * (3 * i + 1), history.green + J * i, J);
    copy_float(m_transposed + J * (3 * i + 2), history.blue + J * i, J);
  }
}

//----( scaling )----

CombineRgb::CombineRgb (Rectangle shape)
  : Rectangle(shape),
    red_in("CombineRgb.red_in", shape),
    green_in("CombineRgb.green_in", shape),
    blue_in("CombineRgb.blue_in", shape)
{}

void CombineRgb::pull (Seconds time, RgbImage & rgb)
{
  if (red_in) {
    MonoImage red(rgb.red);
    red_in.pull(time, red);
  } else {
    rgb.red.zero();
  }

  if (green_in) {
    MonoImage green(rgb.green);
    green_in.pull(time, green);
  } else {
    rgb.green.zero();
  }

  if (blue_in) {
    MonoImage blue(rgb.blue);
    blue_in.pull(time, blue);
  } else {
    rgb.blue.zero();
  }
}

NormalizeTo01::NormalizeTo01 (Rectangle shape)
  : Rectangle(shape),
    m_time(Seconds::now()),

    m_LB(INFINITY),
    m_UB(-INFINITY),

    in("NormalizeTo01.in", shape)
{
}

void NormalizeTo01::pull (Seconds time, MonoImage & image)
{
  ASSERT_SIZE(image, size());

  in.pull(time, image);

  float dt = max(1e-8f, time - m_time);
  m_time = time;

  float decay = exp(-dt / DEFAULT_GAIN_TIMESCALE_SEC);
  float LB = m_LB * decay;
  float UB = m_UB * decay;

  float * restrict f = image;
  const size_t I = image.size;

  for (size_t i = 0; i < I; ++i) {
    imin(LB, f[i]);
    imax(UB, f[i]);
  }
  m_LB = LB;
  m_UB = UB;

  float scale = 1 / (UB - LB + 1e-20f);
  float shift = -LB * scale;
  for (size_t i = 0; i < I; ++i) {
    f[i] = scale * f[i] + shift;
  }
}

//----( projection )----

LiftAxes::LiftAxes (Rectangle shape)
  : Rectangle(shape),

    m_image_x(width()),
    m_image_y(height()),

    x_in("LiftAxes.x_in", width()),
    y_in("LiftAxes.y_in", height())
{}

void LiftAxes::pull (Seconds time, MonoImage & image_yx)
{
  ASSERT_SIZE(image_yx, size());

  x_in.pull(time, m_image_x);
  y_in.pull(time, m_image_y);

  Image::lift_axes_sum(height(), width(), m_image_y, m_image_x, image_yx);
}

//----( oscilloscope )----

Oscilloscope::Oscilloscope (
    Rectangle shape,
    float radius,
    float timescale_sec)
  : Rectangle(shape),
    m_radius(radius),
    m_timescale(timescale_sec),
    m_image(size()),
    m_time(Seconds::now())
{
  ASSERT_LT(0, timescale_sec);
  m_image.zero();
}

void Oscilloscope::push (Seconds time, const complex & signal)
{
  const size_t X = width();
  const size_t Y = height();

  float signal_x = X * (1 + signal.real() / m_radius) / 2;
  float signal_y = Y * (1 + signal.imag() / m_radius) / 2;

  m_mutex.lock();
  BilinearInterpolate(signal_x, X, signal_y, Y).imax(m_image, 1);
  m_mutex.unlock();
}

void Oscilloscope::push (Seconds time, const std::vector<complex> & signals)
{
  const size_t X = width();
  const size_t Y = height();

  m_mutex.lock();

  for (size_t i = 0; i < signals.size(); ++i) {
    complex signal = signals[i] / m_radius;
    float signal_x = X * (1 + signal.real()) / 2;
    float signal_y = Y * (1 + signal.imag()) / 2;

    BilinearInterpolate(signal_x, X, signal_y, Y).imax(m_image, 1);
  }

  m_mutex.unlock();
}

void Oscilloscope::pull (Seconds time, MonoImage & image)
{
  float dt = max(1e-8f, time - m_time);
  m_time = time;
  float decay = expf(-dt / m_timescale);

  m_mutex.lock();

  m_image *= decay;

  image = m_image;

  m_mutex.unlock();
}

//----( features )----

void ImpactDistributor::pull (
    Seconds time,
    BoundedMap<Id, Gestures::Finger> & fingers)
{
  fingers_in.pull(time, fingers);

  float total_impact = 1e-8f;
  for (size_t i = 0; i < fingers.size; ++i) {
    Gestures::Finger & finger = fingers.values[i];

    clip(finger.z(), 0, 1);

    total_impact += finger.get_impact();
  }

  float power;
  if (impact_in) {
    impact_in.pull(time, power);
    power /= total_impact;
  } else {
    power = 1.0f;
  }

  for (size_t i = 0; i < fingers.size; ++i) {
    Gestures::Finger & finger = fingers.values[i];
    finger.set_energy(power * finger.get_impact());
  }
}

void PixToPolar::pull (Seconds time, BoundedMap<Id, Gestures::Finger> & fingers)
{
  float x_center = 0.5f * width();
  float y_center = 0.5f * height();
  float scale = (m_mask_to_disk ? 2.0 : M_SQRT2) / max(width(), height());

  in.pull(time, fingers);
  for (size_t i = 0; i < fingers.size; ++i) {
    Gestures::Finger & finger = fingers.values[i];

    float x = scale * (finger.get_x() - x_center);
    float y = scale * (finger.get_y() - y_center);
    float x_t = scale * finger.get_x_t();
    float y_t = scale * finger.get_y_t();

    float phase = wrap(atan2f(y, x) / (2 * M_PI));
    float radius = sqrtf(sqr(x) + sqr(y) + 1e-8f);
    float phase_t = (y_t * x - x_t * y) / sqr(radius) / (2 * M_PI);
    float radius_t = (x_t * x + y_t * y) / radius;
    imin(radius, 1.0f);

    finger.set_x(radius);
    finger.set_y(phase);
    finger.set_x_t(radius_t);
    finger.set_y_t(phase_t);
  }
}

RelativizeFingers::RelativizeFingers (size_t hand_capacity)
  : m_hands(hand_capacity),
    hands_in("RelativizeFingers.hands_in", hand_capacity),
    fingers_in("RelativizeFingers.fingers_in",
        FINGERS_PER_HAND * hand_capacity)
{}

void RelativizeFingers::pull (
    Seconds time,
    BoundedMap<Id, Gestures::Finger> & fingers)
{
  ASSERT_LE(fingers_in.size(), fingers.capacity);

  hands_in.pull(time, m_hands);
  fingers_in.pull(time, fingers);
  Gestures::relativize_fingers(m_hands, fingers);
}

//----( threads )-------------------------------------------------------------

TimedVideo::TimedVideo (Rectangle shape, float framerate)
  : Rectangle(shape),
    m_timestep(1 / framerate),
    m_time(Seconds::now()),
    m_image(size()),

    out("TimedVideo", shape)
{
  ASSERT_LT(0, framerate);
}

void TimedVideo::run ()
{
  m_time = Seconds::now();

  while (m_running) {
    if (Seconds::now() < m_time) {
      usleep(10000);
    } else {

      step();
      out.push(m_time, m_image);

      m_time += m_timestep;

      PROGRESS_TICKER('|');
    }
  }
}

void Random01Video::step ()
{
  const size_t I = m_image.size;
  float * restrict im = m_image;

  for (size_t i = 0; i < I; ++i) {
    im[i] = random_01();
  }
}

void RandomStdVideo::step ()
{
  const size_t I = m_image.size;
  float * restrict im = m_image;

  for (size_t i = 0; i < I; ++i) {
    im[i] = random_std();
  }
}

} // namespace Streaming

