#ifndef KAZOO_CLOUD_VIDEO_H
#define KAZOO_CLOUD_VIDEO_H

#include "common.h"
#include "cloud_points.h"
#include "compress.h"

namespace Cloud
{

//----( video formats )-------------------------------------------------------

inline VideoFormat detect_video_format (Rectangle shape, size_t dim)
{
  const size_t yuv_dim = shape.size() * 3/2;
  const size_t mono_dim = shape.size();
  const size_t mono_batch_dim = shape.height();

  if (dim == yuv_dim) {
    LOG(" assuming 1 point = 1 YUV video frame");
    return YUV_SINGLE;
  }

  if (dim == mono_dim) {
    LOG(" assuming 1 point = 1 mono video frame");
    return MONO_SINGLE;
  }

  if (dim == mono_batch_dim) {
    size_t points_per_frame = shape.width();
    LOG(" assuming " << points_per_frame << " points = 1 mono frame");
    return MONO_BATCH;
  }

  ERROR(" incompatible shape,dim: " << shape << ", " << dim);
}

//----( iteration )-----------------------------------------------------------

struct PointSequence
{
  const VideoSequence & video;
  const VideoFormat format;
  const Rectangle shape;
  const size_t dim;
  const size_t size;

  PointSequence (const VideoSequence & seq, size_t d)
    : video(seq),
      format(detect_video_format(seq.shape(), d)),
      shape(seq.shape()),
      dim(d),
      size(format == MONO_BATCH ? seq.size() * shape.width() : seq.size())
  {}

  //void shuffle () { const_cast<VideoSequence&>(video).shuffle(); }

  class Iterator
  {
    const VideoSequence & m_video;
    const VideoFormat m_format;
    const size_t m_dim;
    const size_t m_num_frames;
    const size_t m_points_per_frame;
    size_t m_frame;
    size_t m_line;

  public:

    Iterator (PointSequence & seq)
      : m_video(seq.video),
        m_format(seq.format),
        m_dim(seq.dim),
        m_num_frames(m_video.size()),
        m_points_per_frame(m_format == MONO_BATCH ? seq.shape.width() : 1),
        m_frame(0),
        m_line(0)
    {}

    operator bool () { return m_frame < m_num_frames; }
    void next ()
    {
      ++m_line;
      if (m_line >= m_points_per_frame) {
        ++m_frame;
        m_line = 0;
      }
    }
    void operator++ () { next(); }

    uint8_t * data () { return m_video.get(m_frame)->data + m_dim * m_line; }
    Point point () { return Point(m_dim, data()); }
    bool has_prev () { return m_line or m_video.maybe_prev(m_frame); }
  };

  class BatchIterator
  {
    Iterator m_iter;
    const size_t m_dim;
    const size_t m_capacity;
    Point m_buffer;
    std::vector<bool> m_has_prev;
    size_t m_size;

  public:

    BatchIterator (PointSequence & seq, size_t capacity)
      : m_iter(seq),
        m_dim(seq.dim),
        m_capacity(capacity),
        m_buffer(seq.dim * capacity),
        m_has_prev(capacity)
    {
      next();
    }

    operator bool () { return m_size; }
    void next ()
    {
      for (m_size = 0; m_iter and m_size < m_capacity; ++m_iter, ++m_size) {
        memcpy(m_buffer + m_dim * m_size, m_iter.data(), m_dim);
        m_has_prev[m_size] = m_iter.has_prev();
      }
    }
    void operator++ () { next(); }

    size_t buffer_size () { return m_size; }
    Point points () { return Point(m_dim * m_size, m_buffer.data); }
    Point point (size_t i) const
    {
      ASSERT_LT(i, m_size);
      return m_buffer.block(m_dim, i);
    }
    bool has_prev (size_t i) const
    {
      ASSERT_LT(i, m_size);
      return m_has_prev[i];
    }
  };
};

} // namespace Cloud

#endif // KAZOO_CLOUD_VIDEO_H
