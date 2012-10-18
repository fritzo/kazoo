#ifndef KAZOO_CAPTURE_H
#define KAZOO_CAPTURE_H

/** OpenCV wrappers

  (R1) http://opencv.willowgarage.com/documentation/index.html
  (R1) file:///usr/share/doc/opencv-doc/index.htm
  (R3) http://www.cs.iit.edu/~agam/cs512/lect-notes/opencv-intro/opencv-intro.html
  (R4) http://www.seas.upenn.edu/~bensapp/opencvdocs/ref/opencvref_cv.htm
*/

#include "common.h"
#include "vectors.h"
#include <opencv/cv.h>
#include <opencv/highgui.h>

//----( capture )-------------------------------------------------------------

class Capture
{
  CvCapture * m_capture;
  const size_t m_width;
  const size_t m_height;
  const bool m_visible;

public:

  Capture (bool visible = true);
  ~Capture ();

  size_t width () const { return m_width; }
  size_t height () const { return m_height; }
  bool visible () const { return m_visible; }

  // returns a frame pointer, possibly NULL
  IplImage * capture (bool draw = true);
  void draw (IplImage * frame);
};

//----( real capture )--------------------------------------------------------

class RealCapture
{
  Capture m_capture;

  CvMat * m_frame_rgb;
  CvMat * m_frame_out;
  CvMat * m_frame_in;

  IplImage * m_view;

public:

  Vector<float> in, out;

  RealCapture ();
  ~RealCapture ();

  size_t width () const { return m_capture.width(); }
  size_t height () const { return m_capture.height(); }

  // returns m_size most active locations
  bool capture ();
  void draw ();
};

//----( sparse capture )------------------------------------------------------

class SparseCapture
{
  const size_t m_size;
  const float m_mean_part;
  const float m_new_part;

  Capture m_capture;

  CvMat * m_frame_rgb;
  CvMat * m_frame_gray;
  CvMat * m_frame_smooth;
  CvMat * m_frame_mean;
  CvMat * m_frame_max;

  IplImage * m_view;

public:

  SparseCapture (size_t size, float time_scale = 8, bool visible = true);
  ~SparseCapture ();

  size_t size () const { return m_size; }

  // returns m_size most active locations
  bool capture (
      Vector<float> & energy_out,
      Vector<float> & x_out,
      Vector<float> & y_out);
};

//----( testing )-------------------------------------------------------------

void test_capture ();

#endif // KAZOO_CAPTURE_H
