#ifndef KAZOO_ARRAY_H
#define KAZOO_ARRAY_H

/** Constant size vector math.

  By keeping these vectors memory-aligned,
  we can get gcc to optimize to use SSE operations.

  TODO check whether round_size is required for sse optimization
*/

#include "common.h"

#define for_i for (size_t i = 0; i < size; ++i)
#define round_i for (size_t i = 0; i < round_size; ++i)

template<class T, size_t size>
class Array : public Aligned<Array<T,size> >
{
  typedef Array<T, size> This;

protected:

  T data[size ? size : 1] __attribute__ ((aligned (16)));
  enum { round_size = (size * sizeof(T) + 15) / 16 * 16 / sizeof(T) };

public:

  Array () {}
  explicit Array (T a) { for_i data[i] = a; }

  Array (T a, T b)
  {
    static_assert(size == 2, "Array constructed with wrong number of arguments (2)");
    data[0] = a;
    data[1] = b;
  }
  Array (T a, T b, T c)
  {
    static_assert(size == 3, "Array constructed with wrong number of arguments (3)");
    data[0] = a;
    data[1] = b;
    data[2] = c;
  }
  Array (T a, T b, T c, T d)
  {
    static_assert(size == 4, "Array constructed with wrong number of arguments (4)");
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
  }
  Array (T a, T b, T c, T d, T e, T f)
  {
    static_assert(size == 6, "Array constructed with wrong number of arguments (6)");
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
    data[4] = e;
    data[5] = f;
  }
  Array (T a, T b, T c, T d, T e, T f, T g, T h)
  {
    static_assert(size == 8, "Array constructed with wrong number of arguments (8)");
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
    data[4] = e;
    data[5] = f;
    data[6] = g;
    data[7] = h;
  }
  Array (T a, T b, T c, T d, T e, T f, T g, T h, T i, T j, T k, T l)
  {
    static_assert(size == 12, "Array constructed with wrong number of arguments (12)");
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
    data[4] = e;
    data[5] = f;
    data[6] = g;
    data[7] = h;
    data[8] = i;
    data[9] = j;
    data[10] = k;
    data[11] = l;
  }

  const This & operator = (T a)
  {
    for_i data[i] = a;
    return * this;
  }

  operator T * () { return data; }
  operator const T * () const { return data; }

  //----( math operations )----

  This operator + () const { return * this; }
  This operator - () const { return * this * -1; }

  const This & operator += (const This & other) { for_i data[i] += other[i]; return * this; }
  const This & operator -= (const This & other) { for_i data[i] -= other[i]; return * this; }
  const This & operator *= (const This & other) { for_i data[i] *= other[i]; return * this; }
  const This & operator /= (const This & other) { for_i data[i] /= other[i]; return * this; }

  This operator + (const This & other) const { This result = * this; result += other; return result; }
  This operator - (const This & other) const { This result = * this; result -= other; return result; }
  This operator * (const This & other) const { This result = * this; result *= other; return result; }
  This operator / (const This & other) const { This result = * this; result /= other; return result; }

  const This & operator *= (T scale) { for_i data[i] *= scale; return * this; }
  const This & operator /= (T scale) { operator *= (1 / scale); return * this; }

  This operator * (T scale) const { This result = * this; result *= scale; return result; }
  This operator / (T scale) const { return operator * (1 / scale); }

  friend inline This operator * (T scale, const This & vect) { return vect * scale; }
  friend inline This operator / (T scale, const This & vect)
  {
    This result;
    for_i result[i] = scale / vect[i];
    return result;
  }
};

typedef Array<float,2> float2;
typedef Array<float,3> float3;
typedef Array<float,4> float4;
typedef Array<float,6> float6;
typedef Array<float,8> float8;
typedef Array<float,12> float12;

template<class T, size_t size>
inline T sum (const Array<T,size> & x) { T result = 0; for_i result += x[i]; return result; }

template<class T, size_t size>
inline ostream & operator << (ostream & s, const Array<T,size> & x) { for_i s << x[i] << ' '; return s; }

template<class T, size_t size>
inline istream & operator >> (istream & s, Array<T,size> & x) { for_i s >> x[i]; return s; }

template<class T, size_t size>
inline const Array<float,size> * vectorize (const T * x)
{
  ASSERT_ALIGNED(x);
  return reinterpret_cast<const Array<float,size> *>(x);
}

template<class T, size_t size>
inline Array<float,size> * vectorize (T * x)
{
  ASSERT_ALIGNED(x);
  return reinterpret_cast<Array<float,size> *>(x);
}

#undef round_i
#undef for_i

#endif // KAZOO_ARRAY_H
