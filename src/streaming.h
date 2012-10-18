
#ifndef KAZOO_STREAMING_H
#define KAZOO_STREAMING_H

#include "common.h"
#include "vectors.h"
#include "cyclic_time.h"
#include "audio_types.h"
#include "threads.h"
#include <vector>
#include <utility>

//#define PROGRESS_TICKER(mess) {}
#define PROGRESS_TICKER(mess) { cout << mess << flush; }

namespace Streaming
{

void run (); // the main running command

//----( threaded systems )----------------------------------------------------

class Thread : private ::Thread
{
protected:

  bool m_running;

public:

  Thread ();
  virtual ~Thread ();

  virtual void start ();
  virtual void stop ();
  virtual void wait () { ::Thread::wait(); }

protected:

  virtual void step () = 0;

  virtual void run () { while (m_running) step(); }
};

class TimedThread : public Thread
{
protected:

  const float m_rate;

public:

  TimedThread (float rate) : m_rate(rate) { ASSERT_LT(0, rate); }
  virtual ~TimedThread () {}

protected:

  virtual void run ();
};

//----( pushing & pulling )---------------------------------------------------

template<class In>
struct Pushed
{
  virtual ~Pushed () {}
  virtual void push (Seconds time, const In & in) = 0;
};

template<class Out>
struct Pulled
{
  virtual ~Pulled () {}
  virtual void pull (Seconds time, Out & out) = 0;
};

template<class In, class Out = In, class Init = In>
class Bounced;

template<class In, class Out, class Init>
class Bounced
  : public Pushed<In>,
    public Pulled<Out>
{
  In m_in;
  Seconds m_time;

public:

  Bounced () : m_in() {}
  Bounced (Init init) : m_in(init) {}
  virtual ~Bounced () {}

  virtual void bounce (Seconds time, const In & in, Out & out) = 0;

  virtual void push (Seconds time, const In & in)
  {
    m_in = in;
    m_time = time;
  }

  virtual void pull (Seconds time, Out & out)
  {
    ASSERT(time == m_time, "Bounced::push,pull were called out of order");
    bounce(time, m_in, out);
  }
};

// some datatypes prevent efficient emulation of bounce as (push; pull)
template<class In, class Out>
struct Bounced<In, Out, void>
{
  virtual ~Bounced () {}
  virtual void bounce (Seconds time, const In & in, Out & out) = 0;
};

//----( ports )---------------------------------------------------------------

template<class P> class Port;

template<class In>
class Port<Pushed<In> >
{
protected:

  mutable Pushed<In> * m_port;
  const string m_name;

public:

  Port (string name) : m_port(NULL), m_name(name) {}

  operator bool () const { return m_port; }
  void operator- (Pushed<In> & p) { m_port = & p; }
  void insert (Pushed<In> & pushed, Port<Pushed<In> > & port)
  {
    ASSERT(m_port, "pushed port " << m_name << " not set before inserting");
    port.m_port = m_port;
    m_port = pushed;
  }

  void push (Seconds time, const In & in) const
  {
    ASSERT(m_port, "pushed port " << m_name << " not set befure pushing");
    m_port->push(time, in);
  }
};

template<class Out>
class Port<Pulled<Out> >
{
protected:

  mutable Pulled<Out> * m_port;
  const string m_name;

public:

  Port (string name) : m_port(NULL), m_name(name) {}

  operator bool () const { return m_port; }
  void operator- (Pulled<Out> & p) { m_port = & p; }
  void insert (Pulled<Out> & pulled, Port<Pulled<Out> > & port)
  {
    ASSERT(m_port, "pulled port " << m_name << " not set before inserting");
    port.m_port = m_port;
    m_port = pulled;
  }

  void pull (Seconds time, Out & out) const
  {
    ASSERT(m_port, "pulled port " << m_name << " not set before pulling");
    return m_port->pull(time, out);
  }
};

template<class In, class Out, class Init>
class Port<Bounced<In, Out, Init> >
{
protected:

  mutable Bounced<In, Out, Init> * m_port;
  const string m_name;

public:

  Port (string name) : m_port(NULL), m_name(name) {}

  operator bool () const { return m_port; }
  void operator- (Bounced<In, Out, Init> & p) { m_port = & p; }

  void bounce (Seconds time, const In & in, Out & out) const
  {
    ASSERT(m_port, "bounced port " << m_name << " not set");
    return m_port->bounce(time, in, out);
  }
};

template<class P>
class SizedPort : public Port<P>
{
  const size_t m_size;

public:

  SizedPort (string name, size_t size) : Port<P>(name), m_size(size) {}

  size_t size () const { return m_size; }

  operator bool () const { return Port<P>::operator bool(); }
  void operator- (P & p) { Port<P>::operator-(p); }
};

template<class P>
class RectangularPort
  : public Port<P>,
    public Rectangle
{
public:

  RectangularPort (string name, Rectangle shape)
    : Port<P>(name),
      Rectangle(shape)
  {}

  operator bool () const { return Port<P>::operator bool(); }
  void operator- (P & p) { Port<P>::operator-(p); }
};

//----( adapters )------------------------------------------------------------

/** Push-Pull adapters

             Pulled out Pushed out
            +----------+----------+
  Pulled in |  Puller  |  Relay   |
            +----------+----------+
  Pushed in |  Shared  |  Pusher  |
            +----------+----------+

  Puller & Pusher act as buffers.
*/

template<class T, class Init = T>
class Shared
  : public Pushed<T>,
    public Pulled<T>
{
  T m_data;
  Mutex m_mutex;

public:

  Shared (Init init) : m_data(init) {}
  virtual ~Shared () {}

  virtual void push (Seconds, const T & data)
  {
    m_mutex.lock();
    m_data = data;
    m_mutex.unlock();
  }

  virtual void pull (Seconds, T & data)
  {
    m_mutex.lock();
    data = m_data;
    m_mutex.unlock();
  }

  T & unsafe_access () { return m_data; }
};

template<class T, class Init = T>
class Relay : public TimedThread
{
  T m_data;

public:

  Port<Pulled<T> > in;
  Port<Pushed<T> > out;

  Relay (float rate, Init init)
    : TimedThread(rate),
      m_data(init),

      in("Relay.in"),
      out("Relay.out")
  {}
  virtual ~Relay () {}

  virtual void step ()
  {
    Seconds time = Seconds::now();
    in.pull(time, m_data);
    out.push(time, m_data);
    PROGRESS_TICKER('R')
  }
};

template<class T, class Init = T>
class Pusher
  : public Thread,
    public Pushed<T>
{
  Seconds m_time;
  T m_data;

  bool m_full;
  Mutex m_mutex;
  ConditionVariable m_cond;

public:

  Port<Pushed<T> > out;

  Pusher (Init init)
    : m_time(),
      m_data(init),
      m_full(false),
      out("Pusher.out")
  {}
  virtual ~Pusher () {}

  virtual void start () { Thread::start(); m_cond.notify_all(); }
  virtual void stop  () { Thread::stop();  m_cond.notify_all(); }

  virtual void step () { out.push(m_time, m_data); PROGRESS_TICKER('<'); }

  virtual void push (Seconds time, const T & data)
  {
    UniqueLock lock(m_mutex);
    while (m_running) {
      if (m_full) {
        m_cond.wait(lock);
      } else {

        m_time = time;
        m_data = data;

        m_full = true;
        m_cond.notify_all();
        break;
      }
    }
  }

  virtual void run ()
  {
    UniqueLock lock(m_mutex);
    while (m_running) {
      if (not m_full) {
        m_cond.wait(lock);
      } else {

        step();

        m_full = false;
        m_cond.notify_all();
      }
    }
  }
};

template<class T, class Init = T>
class Puller
  : public Thread,
    public Pulled<T>
{
  Seconds m_time;
  T m_data;

  bool m_full;
  Mutex m_mutex;
  ConditionVariable m_cond;

public:

  Port<Pulled<T> > in;

  Puller (Init init)
    : m_time(),
      m_data(init),
      m_full(false),
      in("Puller.in")
  {}
  virtual ~Puller () {}

  virtual void start () { m_time = Seconds::now(); Thread::start(); }
  virtual void stop () { Thread::stop(); m_cond.notify_all(); }

  virtual void step () { in.pull(m_time, m_data);  PROGRESS_TICKER('>'); }

  virtual void run ()
  {
    UniqueLock lock(m_mutex);
    while (m_running) {
      if (m_full) {
        m_cond.wait(lock);
      } else {

        step();

        m_full = true;
        m_cond.notify_all();
      }
    }
  }

  // WARNING the actual data pulled is one timestep previous
  // WARNING initial pulled data may be garbage
  virtual void pull (Seconds time, T & data)
  {
    UniqueLock lock(m_mutex);
    while (m_running) {
      if (not m_full) {
        m_cond.wait(lock);
      } else {

        m_time = time;
        data = m_data;

        m_full = false;
        m_cond.notify_all();
        break;
      }
    }
  }
};

// Vectors should only be initialized with size_t.
// These undefined template prevents errors like Shared<Vector<float> >.
template<class T> class Shared<Vector<T>, Vector<T> > {};
template<class T> class Relay<Vector<T>, Vector<T> > {};
template<class T> class Pusher<Vector<T>, Vector<T> > {};
template<class T> class Puller<Vector<T>, Vector<T> > {};

template<class T>
class DevNull : public Pushed<T>
{
public:

  DevNull () {}
  virtual ~DevNull () {}

  virtual void push (Seconds time, const T & data) {}
};

template<class T>
class Splitter : public Pushed<T>
{
public:

  Port<Pushed<T> > out1;
  Port<Pushed<T> > out2;

  Splitter ()
    : out1("Splitter.out1"),
      out2("Splitter.out2")
  {}
  virtual ~Splitter () {}

  virtual void push (Seconds time, const T & data)
  {
    out1.push(time, data);
    out2.push(time, data);
  }
};

template<class T>
class PullSplitter : public Pulled<T>
{
public:

  Port<Pulled<T> > in;
  Port<Pushed<T> > out;

  PullSplitter ()
    : in("PullSplitter.in"),
      out("PullSplitter.out")
  {}
  virtual ~PullSplitter () {}

  virtual void pull (Seconds time, T & data)
  {
    in.pull(time, data);
    out.push(time, data);
  }
};

template<class Fst, class Snd>
class PushedPair : public Pushed<Snd>
{
  Fst m_fst;

public:

  Shared<Fst> in_fst;
  Port<Pushed<std::pair<Fst,Snd> > > out;

  PushedPair ()
    : in_fst(Fst()),
      out("PushedPair.out")
  {}
  virtual ~PushedPair () {}

  virtual void push (Seconds time, const Snd & snd)
  {
    std::pair<Fst,Snd> data(m_fst, snd);
    out.push(time, data);
  }
};

template<class From, class To>
struct PushedCast : public Pushed<From>
{
  Port<Pushed<To> > out;

  PushedCast () : out("PushedCast.out") {}
  virtual ~PushedCast () {}

  virtual void push (Seconds time, const From & data)
  {
    out.push(time, static_cast<const To &>(data));
  }
};

template<class From, class To>
struct PulledCast : public Pulled<To>
{
  Port<Pulled<From> > in;

  PulledCast () : in("PulledCast.in") {}
  virtual ~PulledCast () {}

  virtual void pull (Seconds time, To & data)
  {
    in.pull(time, static_cast<From &>(data));
  }
};

//----( basic data types )----------------------------------------------------

struct MonoImage : public Vector<float>
{
  enum { num_channels = 1 };

  MonoImage (size_t size) : Vector<float>(size * num_channels) {}
  MonoImage (const Vector<float> & other) : Vector<float>(other) {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct RgbImage : public Vector<float>
{
  enum { num_channels = 3 };

  Vector<float> red;
  Vector<float> green;
  Vector<float> blue;

  RgbImage (size_t size)
    : Vector<float>(size * num_channels),
      red(size, begin() + size * 0),
      green(size, begin() + size * 1),
      blue(size, begin() + size * 2)
  {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct YyuvImage : public Vector<float>
{
  enum { num_channels = 3 };

  Vector<float> yy;
  Vector<float> u;
  Vector<float> v;

  YyuvImage (size_t size)
    : Vector<float>(size * num_channels),
      yy(size, begin() + size * 0),
      u(size / 2, begin() + size * 1),
      v(size / 2, begin() + size * 3 / 2)
  {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct GlovesImage : public Vector<float>
{
  static const float num_channels;

  Vector<float> y;
  Vector<float> u;
  Vector<float> v;

  GlovesImage (size_t size)
    : Vector<float>(size * 3/2),
      y(size, begin() + size * 0),
      u(size / 4, begin() + size * 1),
      v(size / 4, begin() + size * 5 / 4)
  {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct MomentImage : public Vector<float>
{
  enum { num_channels = 3 };

  Vector<float> mass;
  Vector<float> dx;
  Vector<float> dy;

  MomentImage (size_t size)
    : Vector<float>(size * num_channels),
      mass(size, begin() + size * 0),
      dx(size, begin() + size * 1),
      dy(size, begin() + size * 2)
  {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct HandImage : public Vector<float>
{
  enum { num_channels = 3 };

  Vector<float> tip;
  Vector<float> shaft;
  Vector<float> palm;

  HandImage (size_t size)
    : Vector<float>(size * num_channels),
      tip(size, begin() + size * 0),
      shaft(size, begin() + size * 1),
      palm(size, begin() + size * 2)
  {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct FlowImage : public Vector<float>
{
  enum { num_channels = 2 };

  Vector<float> dx;
  Vector<float> dy;

  FlowImage (size_t size)
    : Vector<float>(size * num_channels),
      dx(size, begin() + size * 0),
      dy(size, begin() + size * 1)
  {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct FlowInfoImage : public Vector<float>
{
  enum { num_channels = 5 };

  Vector<float> surprise_x;
  Vector<float> surprise_y;
  Vector<float> info_xx;
  Vector<float> info_xy;
  Vector<float> info_yy;

  FlowInfoImage (size_t size)
    : Vector<float>(size * num_channels),
      surprise_x(size, begin() + size * 0),
      surprise_y(size, begin() + size * 1),
      info_xx(size, begin() + size * 2),
      info_xy(size, begin() + size * 3),
      info_yy(size, begin() + size * 4)
  {}

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct Sliders : public Vector<float>
{
  enum { num_channels = 2 };

  Vector<float> mass, position;

  Sliders (size_t mass_size)
    : Vector<float>(num_channels * mass_size),
      mass      (mass_size, data + mass_size * 0),
      position  (mass_size, data + mass_size * 1)
  {
    zero();
  }

  void operator= (const Vector<float> & other)
  {
    Vector<float>::operator=(other);
  }
};

struct Mono8Image : public Vector<uint8_t>
{
  enum { num_channels = 1 };

  Mono8Image (size_t size, uint8_t * data = NULL)
    : Vector<uint8_t>(size * num_channels, data)
  {}
  Mono8Image (const Vector<uint8_t> & other) : Vector<uint8_t>(other) {}

  void operator= (const Vector<uint8_t> & other)
  {
    Vector<uint8_t>::operator=(other);
  }
};

struct Rgb8Image : public Vector<uint8_t>
{
  enum { num_channels = 3 };

  Vector<uint8_t> red;
  Vector<uint8_t> green;
  Vector<uint8_t> blue;

  Rgb8Image (size_t size)
    : Vector<uint8_t>(size * num_channels),
      red(size, begin() + size * 0),
      green(size, begin() + size * 1),
      blue(size, begin() + size * 2)
  {}

  void operator= (const Vector<uint8_t> & other)
  {
    Vector<uint8_t>::operator=(other);
  }
};

struct Yuv420p8Image : public Vector<uint8_t>
{
  static const float num_channels;

  Vector<uint8_t> y;
  Vector<uint8_t> u;
  Vector<uint8_t> v;

  Yuv420p8Image (size_t size, uint8_t * data = NULL)
    : Vector<uint8_t>(size * 3/2, data),
      y(size, begin() + size * 0),
      u(size / 4, begin() + size * 1),
      v(size / 4, begin() + size * 5 / 4)
  {}

  void operator= (const Vector<uint8_t> & other)
  {
    Vector<uint8_t>::operator=(other);
  }
};

typedef Yuv420p8Image Gloves8Image;

//----( image casting )-------------------------------------------------------

template<class Image>
struct VectorAsImage : public Pushed<Vector<typename Image::value_type> >
{
  RectangularPort<Pushed<Image> > out;

  VectorAsImage (Rectangle shape) : out("VectorAsImage.out", shape) {}
  virtual ~VectorAsImage () {}

  virtual void push (
      Seconds time,
      const Vector<typename Image::value_type> & data)
  {
    ASSERT_SIZE(data, out.size() * Image::num_channels);
    Image image(out.size(), data.data);
    out.push(time, image);
  }
};

} // namespace Streaming

#endif // KAZOO_STREAMING_H

