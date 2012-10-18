
#ifndef KAZOO_AUDIO_H
#define KAZOO_AUDIO_H

/** A simple C++ wrapper object for portaudio streams

  TODO implement mp3 file reader calling system("madplay...") and remove("...")
*/

#include "common.h"
#include "vectors.h"
#include "audio_types.h"
#include <fstream>
#include <portaudio.h>

//----( audio device )--------------------------------------------------------

class Audio
{
  // params
  const size_t m_frames_per_buffer;
  const size_t m_sample_rate;
  const bool m_reading;
  const bool m_writing;
  const bool m_stereo;
  const size_t m_num_channels;

  // portaudio structs
  PaStreamParameters m_input_parameters;
  PaStreamParameters m_output_parameters;
  PaStream * m_stream;

  bool m_active;

public:
  Audio (
      size_t frames_per_buffer = DEFAULT_FRAMES_PER_BUFFER,
      size_t sample_rate = DEFAULT_SAMPLE_RATE,
      bool reading = true,
      bool writing = true,
      bool stereo = true,
      bool verbose = false);
  virtual ~Audio ();

  // diagnostics
  size_t size () const { return m_frames_per_buffer; }
  size_t rate () const { return m_sample_rate; }
  bool reading () const { return m_reading; }
  bool writing () const { return m_writing; }
  bool stereo () const { return m_stereo; }

  // blocking i/o
  void start ();
  void read (Vector<float> & samples);
  void read (Vector<complex> & samples);
  void write (const Vector<float> & samples);
  void write (const Vector<complex> & samples);
  void stop ();
};

//----( audio device thread )-------------------------------------------------

extern "C" PaStreamCallback AudioThread_callback;

class AudioThread
{
  // params
  const size_t m_frames_per_buffer;
  const size_t m_sample_rate;
  const bool m_reading;
  const bool m_writing;
  const bool m_stereo;
  const size_t m_num_channels;

  // portaudio structs
  PaStreamParameters m_input_parameters;
  PaStreamParameters m_output_parameters;
  PaStream * m_stream;

  bool m_active;

public:
  AudioThread (
      size_t frames_per_buffer = DEFAULT_FRAMES_PER_BUFFER,
      size_t sample_rate = DEFAULT_SAMPLE_RATE,
      bool reading = true, // listening
      bool writing = true, // speaking
      bool stereo = true,
      bool verbose = false);
  virtual ~AudioThread ();

  // diagnostics
  size_t size () const { return m_frames_per_buffer; }
  size_t rate () const { return m_sample_rate; }
  bool reading () const { return m_reading; }
  bool writing () const { return m_writing; }
  bool stereo () const { return m_stereo; }
  bool active () const { return m_active; }

  // callback-based i/o
  void start ();
  void stop ();

protected:

  friend PaStreamCallback AudioThread_callback;

  virtual void process (
      const float * restrict samples_in,
      float * restrict samples_out,
      size_t size)
  { ERROR("AudioThread::process was not implemented in base class"); }

  virtual void process (
      const complex * restrict samples_in,
      complex * restrict samples_out,
      size_t size)
  { ERROR("AudioThread::process was not implemented in base class"); }
};

//----( audio file )----------------------------------------------------------

/** Reads audio files.
  Input: raw 16-bit host-endian stereo-interleaved audio files
  Output: finitely many fixed-size zero-padded frames

  Note: to convert test.mp3 to a suitable test.raw, use
    madplay --stereo --output=raw:test.raw test.mp3
  TODO implement mono audio file
  For mono, use
    madplay --mono --output=raw:test.raw test.mp3
*/
class AudioFile
{
  FILE * m_file;
  const bool m_stereo;
  const size_t m_num_channels;
  const size_t m_frame_size;
  Vector<int16_t> m_buffer;
  bool m_done;

public:

  AudioFile (
      string filename,
      bool stereo = true,
      size_t frame_size = DEFAULT_FRAMES_PER_BUFFER);
  ~AudioFile ();

  bool stereo () const { return m_stereo; }
  size_t frame_size () const { return m_frame_size; }

  void skip (size_t num_frames);

  bool done () const { return m_done; }
  void read_frame (Vector<float> & samples);
  void read_frame (Vector<complex> & samples);

private:

  void read_frame (float * restrict samples);
};

//----( simple function api )-------------------------------------------------

// returns true if sound was completely filled from file
bool read_audio_sample (
    const char * filename,
    Vector<float> & sound,
    size_t begin_frame = 0);

// returns true if sound was completely filled from file
bool read_audio_sample (
    const char * filename,
    Vector<complex> & sound,
    size_t begin_frame = 0);

#endif // KAZOO_AUDIO_H

