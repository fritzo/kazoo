
#ifndef KAZOO_PLAYBACK_H
#define KAZOO_PLAYBACK_H

#include "common.h"
#include "image_types.h"
#include "streaming.h"

namespace Streaming
{

template<class Data>
class Sequence
  : public Thread,
    public Pushed<Data>,
    public Pulled<Data>
{
public:

  struct Frame
  {
    Seconds time;
    Data data;
  };

private:

  typedef std::vector<Frame> Frames;
  Frames m_frames;
  size_t m_pull_frame;
  const size_t m_sleep_usec;

public:

  Port<Pushed<Data> > out;

  Sequence (size_t sleep_usec = 10000)
    : m_frames(),
      m_pull_frame(0),
      m_sleep_usec(10000),
      out("Sequence<...>.out")
  {}
  virtual ~Sequence () {}

  // frame access
  size_t size () const { return m_frames.size(); }
  Seconds time (size_t i) const { return m_frames[i].time; }
  Data & data (size_t i) { return m_frames[i].data; }
  const Data & data (size_t i) const { return m_frames[i].data; }

  // pushing
  Data & add (Seconds time)
  {
    m_frames.resize(1 + m_frames.size());
    Frame & frame = m_frames.back();
    frame.time = time;
    return frame.data;
  }
  virtual void push (Seconds time, const Data & data) { add(time) = data; }

  // pulling
  void reset_pull_frame () { m_pull_frame = 0; }
  virtual void pull (Seconds time, Data & data);

  // persistence
  void save (const char * filename = "data/test.seq") const;
  void load (const char * filename = "data/test.seq");
  void playback () const;

  // threading
  virtual void step () {}
  virtual void run ();
};

// TODO specialize to handle Vectors

//----( peaks )---------------------------------------------------------------

class PeaksSequence : public Sequence<Image::Peaks>
{
public:

  typedef Image::Peak Peak;
  typedef Image::Peaks Peaks;

  // statistics
  size_t peak_capacity () const;
  Peak extent () const;
};

} // namespace Streaming

#endif // KAZOO_PLAYBACK_H

