
#include "gloves.h"

#define GLOVES_FILTER_LOWPASS_RADIUS_PX (1.0f)
#define GLOVES_FILTER_KRIG_RADIUS_PX    (1.0f)
#define GLOVES_FILTER_SPEED_RAD_PER_SEC (4.0f)
#define GLOVES_FILTER_IMAGE_SNR         (64.0f)

namespace Streaming
{

//----( filtering )-----------------------------------------------------------

GlovesFilter::GlovesFilter (Rectangle shape_in, const char * config_filename)
  : Rectangle(shape_in),

    m_config(config_filename),
    m_lowpass_radius_px(m_config(
          "filter_lowpass_radius_px",
          GLOVES_FILTER_LOWPASS_RADIUS_PX)),
    m_krig_radius_px(m_config(
          "filter_krig_radius_px",
          GLOVES_FILTER_KRIG_RADIUS_PX)),
    m_speed_rad_per_sec(m_config(
          "filter_speed_rad_per_sec",
          GLOVES_FILTER_SPEED_RAD_PER_SEC)),
    m_image_snr(m_config(
          "filter_image_snr",
          GLOVES_FILTER_IMAGE_SNR)),

    m_lowpass0(shape_in.size()),
    m_lowpass1(shape_in.size()),
    m_info_full(shape_in.size()),
    m_info_half(shape_in.size() / 4),
    m_info_quart(shape_in.size() / 16),
    m_flow_quart(shape_in.size() / 16),
    m_gloves(shape_in.size() / 4),

    m_time(Seconds::now()),

    m_initialized(false),

    out("GlovesFilter.out", shape_in.scaled(0.5))
{
  ASSERT_DIVIDES(4, width());
  ASSERT_DIVIDES(4, height());
}

GlovesFilter::~GlovesFilter ()
{
  PRINT2(rms(m_lowpass0), rms(m_lowpass1));

  FlowInfoImage & flow = m_info_full;
  PRINT2(rms(flow.surprise_x), rms(flow.surprise_x));
  PRINT3(rms(flow.info_xx), rms(flow.info_xy), rms(flow.info_yy));

  PRINT(m_flow_stats);
  float flow_std_dev = sqrtf(m_flow_stats.variance());
  float good_std_dev = 255/4.0f;
  ASSERTW_LT(good_std_dev / M_SQRT2, flow_std_dev);
  ASSERTW_LT(flow_std_dev, good_std_dev * M_SQRT2);
}

void GlovesFilter::push (Seconds time, const Mono8Image & image)
{
  ASSERT_SIZE(image, size());

  if (not m_initialized) {

    m_time = time;
    uchar_to_01(image, m_lowpass1);
    m_initialized = true;

    return;
  }

  const size_t I = width();
  const size_t J = height();

  float dt = max(1e-8f, time - m_time);
  m_time = time;

  float radius_px = sqrtf(sqr(I) + sqr(J)) / 2;
  float speed_px_per_frame = m_speed_rad_per_sec * radius_px * dt;
  float prior_info = 1 / sqr(m_image_snr * speed_px_per_frame);

  m_lowpass0 = m_lowpass1;
  uchar_to_01(image, m_lowpass1);

  Vector<float> & temp_full = m_info_full.surprise_x;
  Image::exp_blur_zero(I, J, m_lowpass_radius_px, m_lowpass1, temp_full);

  Image::local_optical_flow(
      I, J,
      m_lowpass0,
      m_lowpass1,
      m_info_full.surprise_x,
      m_info_full.surprise_y,
      m_info_full.info_xx,
      m_info_full.info_xy,
      m_info_full.info_yy);

  // shrink to half size
  size_t I_flow = I * FlowInfoImage::num_channels;
  Image::scale_by_half(I_flow, J, m_info_full.data, m_info_half.data);
  prior_info *= 4; // to account for scaling

  // shrink to quarter size
  Image::scale_by_half(I_flow/2, J/2, m_info_half.data, m_info_quart.data);
  prior_info *= 4; // to account for scaling

  float normalized_dt = 1.0f / speed_px_per_frame;

  Image::krig_optical_flow(
      I/4, J/4,
      m_krig_radius_px,
      normalized_dt,
      m_info_quart.surprise_x,
      m_info_quart.surprise_y,
      m_info_quart.info_xx,
      m_info_quart.info_xy,
      m_info_quart.info_yy,
      m_flow_quart.dx,
      m_flow_quart.dy,
      prior_info);

  // convert & scale to uint8_t image
  {
    // this does both dx & dy in a single pass
    const size_t IJ8 = I * J / 8;
    Vector<float> dx(IJ8, m_flow_quart.dx.data);
    Vector<uint8_t> u(IJ8, m_gloves.u.data);

    real_to_uchar(dx, u);

    m_flow_stats.add(u.data, u.size);
  }
  {
    const uint8_t * restrict y = image;
    uint8_t * restrict y_half = m_gloves.y;

    for (size_t i = 0, I2 = I/2; i < I2; ++i)
    for (size_t j = 0, J2 = J/2; j < J2; ++j) {
      y_half[i * J2 + j] = ( 2
                           + int(y[(2 * i + 0) * J + 2 * j + 0])
                           + int(y[(2 * i + 0) * J + 2 * j + 1])
                           + int(y[(2 * i + 1) * J + 2 * j + 0])
                           + int(y[(2 * i + 1) * J + 2 * j + 1])
                           ) / 4;
    }
  }

  //PROGRESS_TICKER('f');

  out.push(m_time, m_gloves);
}

//----( visualization )-------------------------------------------------------

Gloves8ToColor::Gloves8ToColor (Rectangle shape)
  : Filter<Gloves8Image, Rgb8Image>("Gloves8ToColor", shape)
{}

void Gloves8ToColor::filter (
    Seconds time,
    const Gloves8Image & gloves,
    Rgb8Image & rgb)
{
  const uint8_t * restrict y = gloves.y;
  const uint8_t * restrict u = gloves.u;
  const uint8_t * restrict v = gloves.v;

  uint8_t * restrict red = rgb.red;
  uint8_t * restrict green = rgb.green;
  uint8_t * restrict blue = rgb.blue;

  const size_t I = in.width();
  const size_t J = in.height();
  const size_t I2 = in.width() / 2;
  const size_t J2 = in.height() / 2;

  for (size_t i2 = 0; i2 < I2; ++i2)
  for (size_t j2 = 0; j2 < J2; ++j2) {
    size_t i2j2 = i2 * J2 + j2;

    int pos_u = int(u[i2j2]);
    int pos_v = int(v[i2j2]);
    int neg_u = 255 - pos_u;
    int neg_v = 255 - pos_v;

    // approximate angles separated by 120degrees
    uint8_t r = pos_u / 2;
    uint8_t g = (neg_u + 2 * pos_v) / 6;
    uint8_t b = (neg_u + 2 * neg_v) / 6;

    size_t ij = i2*2 * J + j2*2;

    red[ij+0+0] = r;  green[ij+0+0] = g;  blue[ij+0+0] = b;
    red[ij+0+1] = r;  green[ij+0+1] = g;  blue[ij+0+1] = b;
    red[ij+J+0] = r;  green[ij+J+0] = g;  blue[ij+J+0] = b;
    red[ij+J+1] = r;  green[ij+J+1] = g;  blue[ij+J+1] = b;
  }

  for (size_t ij = 0; ij < I*J; ++ij) {
    uint8_t y2 = y[ij] / 2;

    red[ij] += y2;
    green[ij] += y2;
    blue[ij] += y2;
  }
}

} // namespace Streaming

