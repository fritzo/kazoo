
#include "compress.h"
#include "streaming_camera.h"
#include "streaming_devices.h"
#include "args.h"

namespace Streaming
{

template<class Camera>
void run_encode (Args & args)
{
  const char * filename = args.pop("data/test.avi");

  Camera camera;
  VideoEncoder encoder(filename, camera.out);

  camera.out - encoder;

  run();
}

void run_decode (Args & args)
{
  const char * filename = args.pop("data/test.avi");
  float speed = args.pop(1.0f);

  VideoFile file(filename);
  VideoPlayer decoder(file, speed);
  Shared<Mono8Image, size_t> image(decoder.mono_out.size());
  ShowMono8 screen(decoder.mono_out);

  decoder.mono_out - image;
  screen.in - image;

  run();
}

} // namespace Streaming

using namespace Streaming;

//----( options )-------------------------------------------------------------

void option_config (Args & args)
{
  Camera::set_config(args.pop());
}

//----( harness )-------------------------------------------------------------

const char * help_message =
"Usage: compress_test [OPTIONS] COMMAND [ARGS]"
"\nOptions:"
"\n  config FILENAME"
"\nCommands:"
"\n  encode [FILENAME = data/test.avi]"
"\n  encode_mono [FILENAME = data/test.avi]"
"\n  encode_fifth [FILENAME = data/test.avi]"
"\n  encode_mono_fifth [FILENAME = data/test.avi]"
"\n  decode [FILENAME = data/test.avi] [SPEED = 1]"
;

int main (int argc, char ** argv)
{
  LOG(kazoo_logo);

  Args args(argc, argv, help_message);

  args
    .case_("config", option_config)
    .default_break_else_repeat();

  args
    .case_("encode", run_encode<Yuv420p8CameraThread>)
    .case_("encode_mono", run_encode<Mono8CameraThread>)
    .case_("encode_fifth", run_encode<FifthYuv420p8CameraThread>)
    .case_("encode_mono_fifth", run_encode<FifthMono8CameraThread>)
    .case_("decode", run_decode)
    .default_error();

  return 0;
}

