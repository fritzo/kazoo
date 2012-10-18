
#include "common.h"
#include "camera.h"
#include "images.h"
#include "animate.h"
#include "filters.h"
#include "events.h"
#include "args.h"
#include <fstream>
#include <algorithm>

void test_camera ()
{
  Camera camera;
  Screen screen(camera);
  Vector<float> frame(camera.size());

  while (not key_pressed()) {
    camera.capture(frame);
    frame *= 1.0f / 255;
    screen.draw(frame);
    screen.update();
  }
}

void test_dilate (float sharpness, size_t radius)
{
  Camera camera;
  Screen screen(camera);
  Vector<float> frame(camera.size());
  Vector<float> temp(camera.size());

  while (not key_pressed()) {
    camera.capture(frame);
    frame /= max(frame);
    Image::dilate(
        camera.height(),
        camera.width(),
        radius,
        frame,
        temp,
        sharpness);
    screen.draw(frame);
    screen.update();
  }
}

void test_crop ()
{
  CameraRegion camera;
  Screen screen(camera);
  Vector<float> frame(camera.size());

  while (not key_pressed()) {
    camera.capture_crop(frame);
    frame *= 1.0f / 255;
    screen.draw(frame);
    screen.update();
  }
}

void test_mask ()
{
  CameraRegion camera;
  size_t W = camera.width();
  size_t H = camera.height();
  Vector<float> temp(camera.size());

  Image::transpose_8(H,W, camera.background(), temp);
  Image::write_image("data/background.im", W, H, temp);
  Image::transpose_8(H,W, camera.mask(), temp);
  Image::write_image("data/mask.im", W, H, temp);

  Screen screen(camera);
  Vector<float> frame(camera.size());

  while (not key_pressed()) {
    camera.capture_crop_mask(frame);
    frame *= 1.0f / 255;
    screen.draw(frame);
    screen.update();
  }
}

void test_diff ()
{
  CameraRegion camera;
  Screen screen(camera);
  Vector<float> frame(camera.size());

  while (not key_pressed()) {
    camera.capture(frame);
    frame *= 1.0f / 255;
    frame += 0.5f;
    screen.draw(frame);
    screen.update();
  }
}

void test_disk ()
{
  CameraRegion camera(new CameraRegion::Disk());
  Screen screen(camera);
  Vector<float> frame(camera.size());

  while (not key_pressed()) {
    camera.capture(frame);
    frame *= 1.0f / 255;
    frame += 0.5f;
    screen.draw(frame);
    screen.update();
  }
}

void test_disk2 ()
{
  CameraRegion camera1(new CameraRegion::Disk());
  CameraRegion camera2(new CameraRegion::Disk());

  ASSERT_EQ(camera1.width(), camera2.width());
  ASSERT_EQ(camera1.height(), camera2.height());

  size_t w = camera1.width();
  size_t h = camera1.height();

  Screen screen(Rectangle(w, 2 * h));
  Vector<float> frame(2 * w * h);
  Vector<float> frame1(w * h, frame.begin());
  Vector<float> frame2(w * h, frame1.end());

  while (not key_pressed()) {
    camera1.capture(frame1);
    camera2.capture(frame2);
    frame *= 1.0f / 255;
    frame += 0.5f;
    screen.draw(frame);
    screen.update();
  }
}

void test_read (const char * filename)
{
  size_t I,J;
  float * image_data;
  Image::read_image(filename, I,J, image_data);
  Vector<float> image(I * J, image_data);

  Screen screen(Rectangle(I,J));
  screen.draw(image, true);
  screen.update();

  while (not key_pressed()) {}
}

const char * help_message =
"Usage: camera_test [OPTIONS] COMMAND [ARGS]"
"\nOptions:"
"\n  exposure _ Sets camera exposure"
"\nCommands:"
"\n  help       Prints this message"
"\n  full       Full camera view"
"\n  crop       Crop to bounding box"
"\n  mask       Crop; mask screen (save bg & mask)"
"\n  diff       Crop; mask screen; background-subtract"
"\n  disk       Crop & mask to disk; background-subtract"
"\n  disk2      Disks from two cameras"
"\n  read FILE  Reads & displays raw image file"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  string arg = args.pop();

  if (arg == "exposure") {
    int exposure = atoi(args.pop());
    Camera::set_exposure(exposure);
    arg = args.pop();
  }

  if (arg == "help")  { LOG(help_message); } else
  if (arg == "full")  { test_camera(); } else
  if (arg == "crop")  { test_crop(); } else
  if (arg == "mask")  { test_mask(); } else
  if (arg == "diff")  { test_diff(); } else
  if (arg == "disk")  { test_disk(); } else
  if (arg == "disk2") { test_disk2(); } else
  if (arg == "read")  { test_read(args.pop()); } else

  ERROR("unknown command: " << arg << '\n' << help_message);

  return 0;
}

