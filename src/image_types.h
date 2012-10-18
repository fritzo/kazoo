
#ifndef KAZOO_IMAGE_TYPES_H
#define KAZOO_IMAGE_TYPES_H

#include "common.h"
#include <vector>

namespace Image
{

//----( data structures )-----------------------------------------------------

struct Point
{
  float x,y;

  Point () {}
  Point (float a_x, float a_y) : x(a_x), y(a_y) {}
};
typedef std::vector<Point> Points;

struct Transform { virtual void operator () (Point &) const = 0; };

struct Peak : public Point
{
  float z; // = intensity

  Peak () {}
  Peak (float a_x, float a_y, float a_z) : Point(a_x, a_y), z(a_z) {}

  bool operator < (const Peak & other) const { return z > other.z; }
};
typedef std::vector<Peak> Peaks;

struct Blob : public Point
{
  float xx,xy,yy;

  Blob () {}
  Blob (float a_x, float a_y, float a_xx, float a_xy, float a_yy)
    : Point(a_x,a_y), xx(a_xx), xy(a_xy), yy(a_yy)
  {}
  Blob (const Point & p, float a_xx, float a_xy, float a_yy)
    : Point(p), xx(a_xx), xy(a_xy), yy(a_yy)
  {}

  void validate ()
  {
    ASSERTW_LT(0, xx);
    ASSERTW_LT(0, yy);
    ASSERTW_LT(0, xx * yy - sqr(xy));
  }
};
typedef std::vector<Blob> Blobs;

//----( pixel coordinate transforms )-----------------------------------------

class PixToRadial : public Transform
{
  float shift_x, shift_y, scale;

public:

  PixToRadial (Rectangle shape)
    : shift_x(-0.5f * shape.width()),
      shift_y(-0.5f * shape.height()),
      scale(2 / sqrtf(sqr(shape.width()) + sqr(shape.height())))
  {}

  PixToRadial (float width, float height)
    : shift_x(-0.5f * width),
      shift_y(-0.5f * height),
      scale(2 / sqrtf(sqr(width) + sqr(height)))
  {}

  virtual void operator() (Point & point) const
  {
    point.x = (point.x + shift_x) * scale;
    point.y = (point.y + shift_y) * scale;
  }

  void scale_input (float factor)
  {
    shift_x *= factor;
    shift_y *= factor;
    scale /= factor;
  }
};

} // namespace Image

//----( i/o )-----------------------------------------------------------------

inline ostream & operator<< (ostream & o, const Image::Point & p)
{
  return o << p.x << ' ' << p.y;
}
inline istream & operator>> (istream & i, Image::Point & p)
{
  return i >> p.x >> p.y;
}

inline ostream & operator<< (ostream & o, const Image::Points & points)
{
  o << points.size() << '\n';
  for (size_t p = 0; p < points.size(); ++p) {
    o << points[p] << '\n';
  }
  return o;
}
inline istream & operator>> (istream & i, Image::Points & points)
{
  size_t size;
  i >> size;
  points.resize(size);
  for (size_t p = 0; p < size; ++p) {
    i >> points[p];
  }
  return i;
}

inline ostream & operator<< (ostream & o, const Image::Peak & p)
{
  return o << p.x << ' ' << p.y << ' ' << p.z;
}
inline istream & operator>> (istream & i, Image::Peak & p)
{
  return i >> p.x >> p.y >> p.z;
}

inline ostream & operator<< (ostream & o, const Image::Peaks & peaks)
{
  o << peaks.size() << '\n';
  for (size_t p = 0; p < peaks.size(); ++p) {
    o << peaks[p] << '\n';
  }
  return o;
}
inline istream & operator>> (istream & i, Image::Peaks & peaks)
{
  size_t size;
  i >> size;
  peaks.resize(size);
  for (size_t p = 0; p < size; ++p) {
    i >> peaks[p];
  }
  return i;
}

inline ostream & operator<< (ostream & o, const Image::Blob & b)
{
  return o << "Blob(" << b.x << ", "
                      << b.y << ", "
                      << b.xx << ", "
                      << b.xy << ", "
                      << b.yy << ")";
}
inline ostream & operator<< (ostream & o, const Image::Blobs & b)
{
  o << "[\n";
  for (size_t i = 0; i < b.size(); ++i) {
    o << "  " << b[i] << ",";
  }
  return o << "]";
}

#endif // KAZOO_IMAGE_TYPES_H

