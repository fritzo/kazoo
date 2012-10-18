
#include "calibration.h"
#include "images.h"
#include "args.h"

using namespace Calibration;

void test_calibrate (
    size_t radius,
    const char * bg_name = "data/background.im",
    const char * mask_name = "data/mask.im")
{
  size_t I,J;
  float * bg;
  Image::read_image(bg_name, I,J, bg);
  Rectangle shape(I,J);

  float * mask = NULL;
  if (mask_name) {
    size_t I_mask,J_mask;
    Image::read_image(mask_name, I_mask,J_mask, mask);
    ASSERT_EQ(I_mask, I);
    ASSERT_EQ(J_mask, J);
  }

  Calibrate calibrate(shape, true);
  calibrate.fit_grid(bg, mask, false, radius);
}

const char * help_message =
"Usage: ./calibration_test [radius]"
"\nFiles: config/default.calibrate.conf"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  size_t radius = args.pop(3);

  test_calibrate(radius);

  return 0;
}

