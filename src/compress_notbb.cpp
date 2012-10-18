
#include "compress.h"

namespace Streaming
{

void VideoSequence::add_files (std::vector<string> filenames)
{
  for (size_t i = 0; i < filenames.size(); ++i) {
    add_file(filenames[i]);
  }
}

} // namespace Video

