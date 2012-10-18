#ifndef KAZOO_COMPRESS_H
#define KAZOO_COMPRESS_H

// Lossless video compression

#include "common.h"
#include "streaming.h"
#include <list>
#include <vector>

namespace Video
{

class EncoderGuts;

} // namespace Video

//----------------------------------------------------------------------------

namespace Streaming
{

class VideoEncoder
  : public Pushed<Yuv420p8Image>,
    public Pushed<Mono8Image>
{
public:

  VideoEncoder (
      string filename,
      Rectangle shape,
      size_t framerate = DEFAULT_VIDEO_FRAMERATE);
  ~VideoEncoder ();

  virtual void push (Seconds time, const Yuv420p8Image & image);
  virtual void push (Seconds time, const Mono8Image & image);

  void push (const Yuv420p8Image & image);
  void push (const Mono8Image & image);

private:

  Video::EncoderGuts * m_guts;
};

class VideoSequence;

class VideoFile
{
  Rectangle m_shape;
  float m_framerate;

  struct Frame
  {
    Seconds time;
    const Yuv420p8Image * image;
    Frame (Seconds t, const Yuv420p8Image * im) : time(t), image(im) {}
  };
  typedef std::list<Frame> Frames;
  Frames m_frames;

public:

  explicit VideoFile (string filename);
  ~VideoFile () { clear(); }

  Rectangle shape () const { return m_shape; }
  size_t size () const { return m_frames.size(); }
  float framerate () const { return m_framerate; }
  const Frames & frames () const { return m_frames; }

  typedef Frames::const_iterator iterator;
  iterator begin () const { return m_frames.begin(); }
  iterator end () const { return m_frames.end(); }

  void clear ();
  void dump_to (VideoSequence & seq);
};

class VideoSequence
{
  struct Frame
  {
    size_t file, time;
    const Yuv420p8Image * image;

    Frame () {}
    Frame (size_t f, size_t t, const Yuv420p8Image * i)
      : file(f), time(t), image(i)
    {}

    bool operator< (const Frame & other) const
    {
      return file == other.file ? time < other.time : file < other.file;
    }
  };
  typedef std::vector<Frame> Frames;

  Rectangle m_shape;
  Frames m_frames;

  size_t m_file;
  size_t m_time;

public:

  VideoSequence (Rectangle shape = Rectangle(0,0));
  VideoSequence (string filename);
  ~VideoSequence () { clear(); }

  Rectangle shape () const { return m_shape; }
  size_t size () const { return m_frames.size(); }

  const Yuv420p8Image * get (size_t i) const;
  const Yuv420p8Image * maybe_prev (size_t i) const;
  const Yuv420p8Image * maybe_next (size_t i) const;

  typedef Frames::const_iterator iterator;
  iterator begin () const { return m_frames.begin(); }
  iterator end () const { return m_frames.end(); }

  void add (const Yuv420p8Image * image);
  void add_file (VideoFile & filename);
  void add_file (string filename) { VideoFile f(filename); add_file(f); }
  void add_files (std::vector<string> filenames);
  void shuffle ();
  void sort ();
  void clear ();
};

class VideoPlayer : public TimedThread
{
  const VideoFile & m_file;
  VideoFile::iterator m_pos;
  Seconds m_time;
  const float m_timestep;

public:

  RectangularPort<Pushed<Yuv420p8Image> > out;
  RectangularPort<Pushed<Mono8Image> > mono_out;

  VideoPlayer (const VideoFile & file, float speed = 1.0f);

  virtual void step ();
};

} // namespace Streaming

#endif // KAZOO_COMPRESS_H
