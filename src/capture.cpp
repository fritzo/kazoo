
#include "capture.h"
#include <set>
#include <algorithm>

//----( capture )-------------------------------------------------------------

/** TODO set frame size with
  cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_WIDTH, 320 );
  cvSetCaptureProperty( capture, CV_CAP_PROP_FRAME_HEIGHT, 240 );
*/

Capture::Capture (bool visible)
  : m_capture(cvCaptureFromCAM(0)),
    m_width( m_capture
           ? cvGetCaptureProperty(m_capture, CV_CAP_PROP_FRAME_WIDTH)
           : 0),
    m_height( m_capture
            ? cvGetCaptureProperty(m_capture, CV_CAP_PROP_FRAME_HEIGHT)
            : 0),
    m_visible(visible)
{
  ASSERT(m_capture, "cannot initialize webcam");

  LOG("capturing " << m_width << "x" << m_height << " video");

  if (m_visible) cvNamedWindow("Capture", CV_WINDOW_AUTOSIZE );
}

Capture::~Capture ()
{
  cvReleaseCapture(&m_capture);

  if (m_visible) cvDestroyWindow("Capture");
}

IplImage * Capture::capture (bool visible)
{
  IplImage * frame = cvQueryFrame(m_capture);
  if (m_visible and visible and frame) draw(frame);
  return frame;
}

void Capture::draw (IplImage * frame)
{
  if (m_visible) cvShowImage("Capture", frame);
}

//----( real capture )--------------------------------------------------------

RealCapture::RealCapture ()
  : m_capture(true),

    m_frame_rgb(cvCreateMat(m_capture.height(), m_capture.width(), CV_32FC3)),
    m_frame_out(cvCreateMat(m_capture.height(), m_capture.width(), CV_32F)),
    m_frame_in(cvCreateMat(m_capture.height(), m_capture.width(), CV_32F)),

    m_view(cvCreateImage(
          cvSize(m_capture.width(), m_capture.height()),
          IPL_DEPTH_8U, 1)),

    in(m_capture.width() * m_capture.height(), m_frame_in->data.fl),
    out(m_capture.width() * m_capture.height(), m_frame_out->data.fl)
{
  out.zero();
  in.zero();
}

RealCapture::~RealCapture ()
{
  cvReleaseMat(&m_frame_rgb);
  cvReleaseMat(&m_frame_out);
  cvReleaseMat(&m_frame_in);

  cvReleaseImage(&m_view);
}

bool RealCapture::capture ()
{
  if (IplImage * frame = m_capture.capture(false)) {

    cvConvertScale(frame, m_frame_rgb, 2.0 / 255.0, -1.0);
    cvCvtColor(m_frame_rgb, m_frame_out, CV_RGB2GRAY);

    return true;
  } else {
    return false;
  }
}

void RealCapture::draw ()
{
  cvConvertScale(m_frame_in, m_view, 127,127);
  m_capture.draw(m_view);
}

//----( sparse capture )------------------------------------------------------

SparseCapture::SparseCapture (size_t size, float time_scale, bool visible)
  : m_size(size),
    m_mean_part(pow(0.5, 1.0 / time_scale)),
    m_new_part(1 - m_mean_part),
    m_capture(visible),

    m_frame_rgb(cvCreateMat(m_capture.height(), m_capture.width(), CV_32FC3)),
    m_frame_gray(cvCreateMat(m_capture.height(), m_capture.width(), CV_32F)),
    m_frame_smooth(cvCreateMat(m_capture.height(), m_capture.width(), CV_32F)),
    m_frame_mean(cvCreateMat(m_capture.height(), m_capture.width(), CV_32F)),
    m_frame_max(cvCreateMat(m_capture.height(), m_capture.width(), CV_32F)),

    m_view(cvCreateImage(
          cvSize(m_capture.width(), m_capture.height()),
          IPL_DEPTH_8U, 1))
{
  PRINT(m_mean_part);
  PRINT(m_new_part);

  for (int i = 0, I = m_frame_mean->rows; i < I; ++i) {
    for (int j = 0, J = m_frame_mean->cols; j < J; ++j) {
      int ij = J * i + j;
      m_frame_mean->data.fl[ij] = 0;
    }
  }
}

SparseCapture::~SparseCapture ()
{
  cvReleaseMat(&m_frame_rgb);
  cvReleaseMat(&m_frame_gray);
  cvReleaseMat(&m_frame_smooth);
  cvReleaseMat(&m_frame_mean);
  cvReleaseMat(&m_frame_max);

  cvReleaseImage(&m_view);
}

struct Peak
{
  float energy;
  int i;
  int j;

  Peak (float a_energy, int a_i, int a_j) : energy(a_energy), i(a_i), j(a_j) {}

  bool operator< (const Peak & other) const { return energy < other.energy; }
};

class Peaks
{
  const size_t m_max_size;

  std::set<Peak> m_peaks;
  bool m_full;

public:

  Peaks (size_t size) : m_max_size(size), m_full(false) {}

  size_t size () const { return m_peaks.size(); }

  void add (float energy, int i, int j)
  {
    if (m_full) {
      std::set<Peak>::iterator smallest = m_peaks.begin();
      if (energy > smallest->energy) {
        m_peaks.erase(smallest);
        m_peaks.insert(Peak(energy, i, j));
      }
    } else {
      m_peaks.insert(Peak(energy, i, j));
      m_full = (size() == m_max_size);
    }
  }

  // iteration
  typedef std::set<Peak>::const_iterator iterator;
  iterator begin () const { return m_peaks.begin(); }
  iterator end () const { return m_peaks.end(); }
};

bool SparseCapture::capture (Vector<float> & e_out,
                             Vector<float> & x_out,
                             Vector<float> & y_out)
{
  ASSERT_SIZE(x_out, m_size);
  ASSERT_SIZE(y_out, m_size);
  ASSERT_SIZE(e_out, m_size);

  if (IplImage * frame = m_capture.capture(false)) {

    cvConvertScale(frame, m_frame_rgb, -1.0f / 255.0);
    cvCvtColor(m_frame_rgb, m_frame_gray, CV_RGB2GRAY);
    cvSmooth(m_frame_gray, m_frame_smooth, CV_GAUSSIAN, 7,7, 0,0);

    float * restrict data = m_frame_smooth->data.fl;
    float * restrict mean = m_frame_mean->data.fl;
    float * restrict local_max = m_frame_max->data.fl;

    // meanumulate running average and subtract
    for (int i = 1, I = m_frame_smooth->rows; i < I - 1; ++i) {
      for (int j = 1, J = m_frame_smooth->cols; j < J - 1; ++j) {
        int ij = J * i + j;

        float & mean_ij = mean[ij];
        float & data_ij = data[ij];

        mean_ij = m_mean_part * mean_ij
                + m_new_part * data_ij;
        data_ij -= mean_ij;
      }
    }
    if (m_capture.visible()) cvConvertScale(m_frame_smooth, m_view, 127,127);

    // find peaks
    const size_t num_dilate_iters = 3;
    cvDilate(m_frame_smooth, m_frame_max, NULL, num_dilate_iters);
    Peaks peaks(m_size);
    for (int i = 1, I = m_frame_smooth->rows; i < I - 1; ++i) {
      for (int j = 1, J = m_frame_smooth->cols; j < J - 1; ++j) {
        int ij = J * i + j;

        float energy = data[ij];
        if ((energy >= 0) and (energy >= local_max[ij])) {
          peaks.add(energy, i, j);
        }
      }
    }
    ASSERTW(peaks.size(), "no peaks found");

    // collect peaks
    float i_scale = 1.0f / m_frame_smooth->rows;
    float j_scale = 1.0f / m_frame_smooth->cols;
    int radius_scale = 100;

    size_t rank = m_size - 1;
    for (Peaks::iterator p = peaks.begin(); p != peaks.end(); ++p) {

      x_out[rank] = i_scale * p->i;
      y_out[rank] = j_scale * p->j;
      e_out[rank] = p->energy;

      if (m_capture.visible()) {
        CvPoint point;
        point.x = p->j;
        point.y = p->i;
        cvCircle(m_view, point, radius_scale * p->energy, CV_RGB(255,0,0));
      }

      --rank;
    }

    if (m_capture.visible()) m_capture.draw(m_view);

    return true;

  } else {

    x_out.zero();
    y_out.zero();
    e_out.zero();

    return false;
  }
}

