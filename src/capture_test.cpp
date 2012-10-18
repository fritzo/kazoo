
#include "capture.h"
#include "transforms.h"

void test_capture1 ()
{
  Capture capture;

  while (true) {

    if (not capture.capture()) break;
    if (cvWaitKey(5) >= 0) break;

    cout << '.' << flush;
  }
}

void test_capture2 (size_t size = 32)
{
  SparseCapture capture(size);
  Vector<float> e(size);
  Vector<float> x(size);
  Vector<float> y(size);

  while (true) {

    if (not capture.capture(e,x,y)) break;
    if (cvWaitKey(5) >= 0) break;

    cout << '.' << flush;
  }
}

void test_capture3 (float scale = 8)
{
  RealCapture capture;

  while (true) {

    if (not capture.capture()) break;
    if (cvWaitKey(5) >= 0) break;

    //TODO test blurring
    capture.in = capture.out;
    capture.draw();

    cout << '.' << flush;
  }
}

int main ()
{
  test_capture3();
  //test_capture1();
  //test_capture2();
  test_capture3();

  return 0;
}

