
#include "common.h"
#include "psycho.h"
#include "images.h"
#include "args.h"
#include <algorithm>

using namespace Psycho;

void test_history (Args & args)
{
  size_t size = args.pop(64);
  size_t length = args.pop(32);
  size_t density = args.pop(16);

  LOG("Constructing History("<< size <<','<< length <<','<< density <<')');
  History history(size, length, density);

  LOG("Adding to history");
  Vector<float> data(size);
  data.zero();
  while (not history.full()) {
    cout << '.' << flush;
    history.add(data);
  }
  cout << endl;

  LOG("Getting from history");
  Vector<float> image(size * length);
  history.get(image);

  LOG("Done!");
}

void test_exp_blur (Args & args)
{
  const size_t size = args.pop(64);
  const float radius = args.pop(1.0f);

  Vector<float> a1(size);
  Vector<float> a2(size);
  Vector<float> b1(size);
  Vector<float> b2(size);

  for (size_t i = 0; i < size; ++i) {
    a1[i] = a2[i] = random_01();
  }
  std::reverse(a2.begin(), a2.end());

  Image::exp_blur_1d_zero(size, radius, a1, b1);
  Image::exp_blur_1d_zero(size, radius, a2, b2);
  std::reverse(b2.begin(), b2.end());

  float rms_error = sqrt(dist_squared(b1, b2) / size);
  PRINT(rms_error);
  ASSERT_LT(rms_error, 1e-8f * size);
}

void test_exp_blur2 (Args & args)
{
  const size_t size = args.pop(64);
  const float radius_fwd = args.pop(1.0f);
  const float radius_bwd = args.pop(2.0f);

  Vector<float> a1(size);
  Vector<float> a2(size);
  Vector<float> b1(size);
  Vector<float> b2(size);

  for (size_t i = 0; i < size; ++i) {
    a1[i] = a2[i] = random_01();
  }
  std::reverse(a2.begin(), a2.end());

  Image::exp_blur_1d_zero(size, radius_fwd, radius_bwd, a1, b1);
  Image::exp_blur_1d_zero(size, radius_bwd, radius_fwd, a2, b2);
  std::reverse(b2.begin(), b2.end());

  float rms_error = sqrt(dist_squared(b1, b2) / size);
  PRINT(rms_error);
  ASSERT_LT(rms_error, 1e-8f * size);
}

const char * help_message =
"Usage: psycho_test [COMMAND = all] [OPTIONS]"
"\nCommands:"
"\n  history [SIZE] [LENGTH] [DURATION]"
"\n  exp_blur [SIZE]"
"\n  exp_blur2 [SIZE]"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  args
    .case_("history", test_history)
    .case_("exp_blur", test_exp_blur)
    .case_("exp_blur2", test_exp_blur2)
    .default_all();

  return 0;
}

