#ifndef KAZOO_CYCLIC_TIME_H
#define KAZOO_CYCLIC_TIME_H

#include "common.h"
#include <sys/time.h>

/** A cyclic clock with microsecond resolution and 1-minute cycles
*/
class Seconds
{
  enum {
    PERIOD = 60000000,
    SECOND = 1000000
  };
  static int ticks (timeval t) { return (t.tv_sec % 60) * SECOND + t.tv_usec; }

  typedef uint64_t jack_time_t; // should mach typedef in jack/jack.h
  static int ticks (jack_time_t t) { return t % PERIOD; }

  static int ticks (float t) { return roundi(t * SECOND); }
  static int wrap (int t) { return t >= 0 ? t % PERIOD
                                          : PERIOD - 1 - (-t-1) % PERIOD; }

  static long ticks (double t) { return roundi(t * SECOND); }
  static int wrap (long t) { return t >= 0 ? t % PERIOD
                                           : PERIOD - 1 - (-t-1) % PERIOD; }

  int m_phase;

  explicit Seconds (int phase) : m_phase(wrap(phase)) {}
public:
  explicit Seconds (float t) : m_phase(wrap(ticks(t))) {}
  explicit Seconds (double t) : m_phase(wrap(ticks(t))) {}
  explicit Seconds (timeval t) : m_phase(wrap(ticks(t))) {}
  explicit Seconds (jack_time_t t) : m_phase(wrap(ticks(t))) {}
  Seconds () {}

  // current time
  static Seconds now () { timeval t; gettimeofday(&t,NULL); return Seconds(t); }

  // time shifts
  Seconds operator+ (float dt) const { return Seconds(m_phase + ticks(dt)); }
  Seconds operator- (float dt) const { return Seconds(m_phase - ticks(dt)); }
  void operator+= (float dt) { m_phase = wrap(m_phase + ticks(dt)); }
  void operator-= (float dt) { m_phase = wrap(m_phase - ticks(dt)); }

  // time differences
  float operator- (const Seconds & t) const
  {
    int dt = (3 * PERIOD / 2 + m_phase - t.m_phase) % PERIOD - PERIOD / 2;
    return dt * 1e-6f;
  }

  // time ordering (only locally transitive)
  bool operator== (const Seconds & t) const { return m_phase == t.m_phase; }
  bool operator!= (const Seconds & t) const { return m_phase != t.m_phase; }
  bool operator<  (const Seconds & t) const { return (*this - t) <  0; }
  bool operator<= (const Seconds & t) const { return (*this - t) <= 0; }
  bool operator>  (const Seconds & t) const { return (*this - t) >  0; }
  bool operator>= (const Seconds & t) const { return (*this - t) >= 0; }
};

#endif // KAZOO_CYCLIC_TIME_H
