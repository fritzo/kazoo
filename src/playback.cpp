
#include "playback.h"

//----( file i/o )------------------------------------------------------------

inline void safe_fwrite (const void * ptr, size_t size, FILE * file)
{
  if (size) {
    size_t info = fwrite(ptr, size, 1, file);
    ASSERT(info, "fwrite failed to write " << size << " bytes");
  }
}

inline void safe_fread (void * ptr, size_t size, FILE * file)
{
  if (size) {
    size_t info = fread(ptr, size, 1, file);
    ASSERT(info, "fread failed to read " << size << " bytes");
  }
}

template<class T>
inline void safe_fwrite (const T & t, FILE *file)
{
  safe_fwrite(&t, sizeof(T), file);
}

template<class T>
inline void safe_fread (T & t, FILE *file)
{
  safe_fread(&t, sizeof(T), file);
}

template<class T>
inline void safe_fwrite (const std::vector<T> & ts, FILE *file)
{
  size_t size = ts.size();
  safe_fwrite(size, file);

  safe_fwrite(& ts.front(), size * sizeof(T), file);
}

template<class T>
inline void safe_fread (std::vector<T> & ts, FILE *file)
{
  size_t size;
  safe_fread(size, file);
  ts.resize(size);

  safe_fread(& ts.front(), size * sizeof(T), file);
}

template<class T>
inline void safe_fwrite (const Vector<T> & ts, FILE *file)
{
  size_t size = ts.size();
  safe_fwrite(size, file);

  safe_fwrite(ts.data, ts.size * sizeof(T), file);
}

template<class T>
inline void safe_fread (Vector<T> & ts, FILE *file)
{
  size_t size;
  safe_fread(size, file);
  ASSERT_EQ(size, ts.size);

  safe_fread(ts.data, size * sizeof(T), file);
}

namespace Streaming
{

//----( sequences )-----------------------------------------------------------

//----( streaming )----

template<class Data>
void Sequence<Data>::pull (Seconds, Data & data)
{
  while (m_pull_frame >= m_frames.size()) usleep(m_sleep_usec);

  data = m_frames[m_pull_frame++].data;
}

template<class Data>
void Sequence<Data>::playback () const
{
  if (m_frames.empty()) return;

  float offset = Seconds::now() - m_frames.front().time;
  typedef typename Frames::const_iterator Auto;
  for (Auto i = m_frames.begin(); i != m_frames.end(); ++i) {
    out.push(i->time + offset, i->data);
  }
}

template<class Data>
void Sequence<Data>::run ()
{
  typedef typename Frames::iterator Auto;
  Auto frame = m_frames.begin();
  if (frame == m_frames.end()) return;

  float offset = Seconds::now() - frame->time;
  while (m_running and frame != m_frames.end()) {
    Seconds time = frame->time + offset;
    if (time > Seconds::now()) {
      usleep(m_sleep_usec);
    } else {
      out.push(time, frame->data);
      ++frame;
    }
  }
}

//----( persistence )----

/**
  Format:
    num_frames
    frame = [ time data ]
    ...
    frame = [ time data ]
*/

template<class Data>
void Sequence<Data>::save (const char * filename) const
{
  FILE * file = fopen(filename, "wb");
  ASSERT(file, "failed to open " << filename << " for writing");
  LOG("saving sequence of " << size() << " frames to " << filename);

  size_t num_frames = m_frames.size();
  safe_fwrite(num_frames, file);

  for (size_t i = 0; i < num_frames; ++i) {
    const Frame & frame = m_frames[i];

    safe_fwrite(frame.time, file);
    safe_fwrite(frame.data, file);
  }

  fclose(file);
}

template<class Data>
void Sequence<Data>::load (const char * filename)
{
  FILE * file = fopen(filename, "rb");
  ASSERT(file, "failed to open " << filename << " for reading");
  LOG("loading sequence from " << filename);

  size_t num_frames;
  safe_fread(num_frames, file);
  m_frames.clear();

  for (size_t i = 0; i < num_frames; ++i) {
    Seconds time;
    safe_fread(time, file);
    Data & data = add(time);

    safe_fread(data, file);
  }

  fclose(file);

  LOG(" loaded " << size() << " frames");
}

//----( peaks )---------------------------------------------------------------

//----( statistics )----

size_t PeaksSequence::peak_capacity () const
{
  size_t capacity = 0;
  for (size_t i = 0; i < size(); ++i) {
    imax(capacity, data(i).size());
  }
  return capacity;
}

PeaksSequence::Peak PeaksSequence::extent () const
{
  float x = 0;
  float y = 0;
  float z = 0;
  for (size_t i = 0; i < size(); ++i) {
    const Peaks & peaks = data(i);

    for (size_t j = 0; j < peaks.size(); ++j) {
      const Peak & peak = peaks[j];
      imax(x, fabsf(peak.x));
      imax(y, fabsf(peak.y));
      imax(z, fabsf(peak.z));
    }
  }
  return Peak(x,y,z);
}

//----( explicit template instantiations )------------------------------------

#define INSTANTIATE_TEMPLATES(Data) \
  template void Sequence<Data>::pull(Seconds, Data &); \
  template void Sequence<Data>::playback () const; \
  template void Sequence<Data>::save(const char *) const; \
  template void Sequence<Data>::load(const char *); \
  template void Sequence<Data>::run();

INSTANTIATE_TEMPLATES(Image::Peaks)
INSTANTIATE_TEMPLATES(float)

} // namespace Streaming

