
#ifndef KAZOO_SCRIPTING_H
#define KAZOO_SCRIPTING_H

#include "common.h"
#include <algorithm>
#include <unistd.h>
#include <sys/wait.h>

inline string strip_path (string filename)
{
  size_t pos = filename.rfind("/");
  if (pos == filename.npos) {
    return filename;
  } else {
    return filename.substr(pos + 1);
  }
}

inline string strip_extension (string filename)
{
  size_t pos = filename.rfind(".");
  ASSERT(pos != filename.npos, "found no extension on filename: " << filename);
  return filename.substr(0, pos);
}

inline string strip_to_stem (string filename)
{
  return strip_path(strip_extension(filename));
}

inline string prepend_to_filename (string filename, string prefix)
{
  size_t pos = filename.rfind("/");
  if (pos == filename.npos) {
    return prefix + filename;
  } else {
    string result = filename;
    result.insert(pos + 1, prefix);
    return result;
  }
}

inline char to_lower (char c) { return std::tolower(c); }
inline string get_filetype (string filename)
{
  size_t pos = filename.rfind(".");
  ASSERT(pos != filename.npos, "could not determine filetype of " << filename);
  string type = filename.substr(pos + 1);
  std::transform(type.begin(), type.end(), type.begin(), to_lower);
  return type;
}

class DecompressAudio
{
  string m_tempfile;

public:

  DecompressAudio (string & filename, bool stereo)
  {
    string type = get_filetype(filename);

    std::ostringstream sample_rate;
    sample_rate << "--sample-rate=" << DEFAULT_SAMPLE_RATE;

    if (type == "raw") {

      return;

    } else if (type == "mp3") {

      m_tempfile = "/tmp/" + strip_to_stem(filename) + ".raw";
      LOG("decompressing " << (stereo ? " stereo " : " mono ")
          << filename << " -> " << m_tempfile);

      int info = 0;
      switch (fork()) {
        case -1:
          ERROR("failed to fork madplay");
          break;

        case 0:
          info = execlp("madplay", "madplay",
              stereo ? "--stereo" : "--mono",
              "-o", m_tempfile.c_str(),
              sample_rate.str().c_str(),
              filename.c_str(),
              NULL);
          exit(info);
          break;

        default:
          wait(&info);
          ASSERT(info == 0, "madplay failed with info " << info);
      }

      filename = m_tempfile;

    } else {

      ERROR("unsupported audio file type: " << type);
    }
  }

  ~DecompressAudio ()
  {
    if (m_tempfile.size()) {
      LOG("removing " << m_tempfile);

      int info = 0;
      switch (fork()) {
        case -1:
          ERROR("failed to fork rm");
          break;

        case 0:
          info = execlp("rm", "rm", m_tempfile.c_str(), NULL);
          exit(info);
          break;

        default:
          wait(&info);
          ASSERT(info == 0, "rm failed with info " << info);
      }
    }
  }
};

#endif // KAZOO_SCRIPTING_H

