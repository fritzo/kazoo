
#ifndef KAZOO_STREAMING_VECTORS_H
#define KAZOO_STREAMING_VECTORS_H

#include "common.h"
#include "streaming.h"
#include "vectors.h"

namespace Streaming
{

template<class In, class Out = In>
class VectorFilter
  : public Pushed<Vector<In> >,
    public Pulled<Vector<Out> >
{
  Vector<In> m_input;
  Vector<Out> m_output;

public:

  SizedPort<Pulled<Vector<In> > > in;
  SizedPort<Pushed<Vector<Out> > > out;

  VectorFilter (string name, size_t size)
    : m_input(size),
      m_output(size),
      in(name + string(".in"), size),
      out(name + string(".out"), size)
  {}

  VectorFilter (string name, size_t size_in, size_t size_out)
    : m_input(size_in),
      m_output(size_out),
      in(name + string(".in"), size_in),
      out(name + string(".out"), size_out)
  {}

  virtual ~VectorFilter () {}

  virtual void filter (
      Seconds time,
      const Vector<In> & input,
      Vector<Out> & output) = 0;

  virtual void push (Seconds time, const Vector<In> & input)
  {
    ASSERT_SIZE(input, in.size());
    filter(time, input, m_output);
    out.push(time, m_output);
  }

  virtual void pull (Seconds time, Vector<Out> & output)
  {
    ASSERT_SIZE(output, out.size());
    in.pull(time, m_input);
    filter(time, m_input, output);
  }
};

template<class Function>
class VectorMap
  : public VectorFilter<
        typename Function::value_type,
        typename Function::result_type>
{
  Function m_function;

  typedef typename Function::value_type value_type;
  typedef typename Function::result_type result_type;

public:

  VectorMap (Function function, size_t size)
    : VectorFilter<value_type, result_type>("VectorMap", size),
      m_function(function)
  {}
  virtual ~VectorMap () {}

  virtual void filter (
      Seconds time,
      const Vector<value_type> & input,
      Vector<result_type> & output)
  {
    const value_type * restrict in = input;
    result_type * restrict out = output;

    for (size_t i = 0, I = input.size; i < I; ++i) {
      out[i] = m_function(in[i]);
    }
  }
};

namespace VectorOperations
{
struct Sqrtf
{
  typedef float value_type;
  typedef float result_type;
  inline float operator() (float x) { return sqrtf(x); }
};
struct Powf
{
  const float exponent;
  typedef float value_type;
  typedef float result_type;
  Powf (float e) : exponent(e) {}
  inline float operator() (float x) const { return powf(x, exponent); }
};
} // namespace VectorOperations

class VectorMaxGain : public VectorFilter<float>
{
  Filters::MaxGain m_gain;

public:

  VectorMaxGain (size_t size, float timescale)
    : VectorFilter<float>("VectorMaxGain", size),
      m_gain(timescale)
  {
  }
  virtual ~VectorMaxGain () {}

  virtual void filter (
      Seconds time,
      const Vector<float> & input,
      Vector<float> & output)
  {
    float scale = m_gain.update(max(input));
    multiply(scale, input, output);
  }
};

} // namespace Streaming

#endif // KAZOO_STREAMING_VECTORS_H

