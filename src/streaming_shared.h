
#ifndef KAZOO_STREAMING_SHARED_H
#define KAZOO_STREAMING_SHARED_H

#include "common.h"
#include "streaming.h"

namespace Streaming
{

class SharedMaxLowpass
  : public Pushed<float>,
    public Pulled<float>
{
  const float m_timescale;

  Seconds m_time;
  float m_lowpass;

  Mutex m_mutex;

public:

  SharedMaxLowpass (float timescale, float initial_value = 0);
  virtual ~SharedMaxLowpass () {}

  virtual void push (Seconds time, const float & data);
  virtual void pull (Seconds time, float & data);
};

template<class T>
class SharedLowpass
  : public Pushed<T>,
    public Pulled<T>
{
  const float m_timescale;
  Seconds m_time;
  T m_value;
  Mutex m_mutex;

public:

  SharedLowpass (float timescale)
    : m_timescale(timescale),
      m_time(Seconds::now()),
      m_value(0)
  {}
  virtual ~SharedLowpass () {}

  virtual void push (Seconds time, const T & data)
  {
    float dt = max(1e-8f, time - m_time);
    m_time = time;
    float decay = exp(-dt / m_timescale);

    m_mutex.lock();
    m_value = decay * m_value + (1-decay) * data;
    m_mutex.unlock();
  }

  virtual void pull (Seconds, T & data)
  {
    m_mutex.lock();
    data = m_value;
    m_mutex.unlock();
  }
};

template<class T>
class SharedLowpass<Vector<T> >
  : public Pushed<Vector<T> >,
    public Pulled<Vector<T> >
{
  const float m_timescale;
  Seconds m_time;
  Vector<T> m_value;
  Mutex m_mutex;

public:

  SharedLowpass (float timescale, size_t size)
    : m_timescale(timescale),
      m_time(Seconds::now()),
      m_value(size)
  {
    m_value.zero();
  }
  virtual ~SharedLowpass () {}

  virtual void push (Seconds time, const Vector<T> & data)
  {
    float dt = max(1e-8f, time - m_time);
    m_time = time;
    float decay = exp(-dt / m_timescale);

    m_mutex.lock();
    accumulate_step(decay, m_value, data);
    m_mutex.unlock();
  }

  virtual void pull (Seconds, Vector<T> & data)
  {
    m_mutex.lock();
    data = m_value;
    m_mutex.unlock();
  }
};

class SharedEvent
  : public Pushed<bool>,
    public Pulled<bool>
{
  bool m_event;

public:

  SharedEvent () : m_event(false) {}
  virtual ~SharedEvent () {}

  virtual void push (Seconds, const bool & event) { m_event |= event; }
  virtual void pull (Seconds, bool & event)
  {
    event = m_event;
    m_event = false;
  }
};

class SharedCounter
  : public Pushed<size_t>,
    public Pulled<size_t>
{
  bool m_total;

public:

  SharedCounter () : m_total(0) {}
  virtual ~SharedCounter () {}

  virtual void push (Seconds, const size_t & value) { m_total += value; }
  virtual void pull (Seconds, bool & value) { value = m_total; m_total = 0; }
};

} // namespace Streaming

#endif // KAZOO_STREAMING_SHARED_H

