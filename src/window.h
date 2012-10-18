#ifndef KAZOO_WINDOW_H
#define KAZOO_WINDOW_H

/** Window functions map [0,1] --> RR, and may be negative

                        ^  w(t)              Satisfying:
                    __--|--__                  w(0) = "max value"
                 _-~    |    ~-_               w(-1) = w(1) = 0
               _/       |       \_             dw(-1) = dw(0) = dw(1) = 0
             _/         |         \_           w(x) = w(-x)
      ___--~~           |           ~~--___
   -+-------------------+-------------------+------> t
   -1                   0                   1
*/

#include "common.h"

//----------------------------------------------------------------------------
/** Hann window.

  The hann window also satisfies
    w(x) = 1 - w(1-x) for x in [0,1]
*/

inline float window_Hann  (float t)
{
  ASSERT((-1 <= t) and (t <= 1), "window argument out of range: " << t);
  return   (1 + cosf(M_PI * t)) / 2;
}
inline float dwindow_Hann (float t)
{
  ASSERT((-1 <= t) and (t <= 1), "window argument out of range: " << t);
  return -M_PI * sinf(M_PI * t) / 2;
}

struct HannWindow : public FunctionAndDeriv
{
  virtual ~HannWindow () {}
  virtual float value (float t) const { return window_Hann(t); }
  virtual float deriv (float t) const { return dwindow_Hann(t); }
};

//----------------------------------------------------------------------------
/** Hann window to a power
*/

inline float window_Hann (float t, float p)
{
  return pow(window_Hann(t), p);
}
inline float dwindow_Hann (float t, float p)
{
  return dwindow_Hann(t) * p * window_Hann(t, p-1);
}

struct HannPowerWindow : public FunctionAndDeriv
{
  const float power;

  HannPowerWindow (float a_power) : power(a_power) {}
  virtual ~HannPowerWindow () {}

  virtual float value (float t) const { return window_Hann(t, power); }
  virtual float deriv (float t) const { return dwindow_Hann(t, power); }
};

//----------------------------------------------------------------------------
/** Blackman Nuttal window.
  see http://en.wikipedia.org/wiki/Window_function
*/

inline float window_BlackmanNuttall (float t)
{
  t *= M_PI;
  return 0.3635819f
       + 0.4891775f * cos(t)
       + 0.1365995f * cos(2 * t)
       + 0.0106411f * cos(3 * t);
}
inline float dwindow_BlackmanNuttall (float t)
{
  t *= M_PI;
  return - M_PI * ( 0.4891775f * sin(t)
                  + 0.1365995f * sin(2 * t) * 2
                  + 0.0106411f * sin(3 * t) * 3 );
}

struct BlackmanNuttallWindow : public FunctionAndDeriv
{
  virtual ~BlackmanNuttallWindow () {}

  virtual float value (float t) const { return window_BlackmanNuttall(t); }
  virtual float deriv (float t) const { return dwindow_BlackmanNuttall(t); }
};

//----------------------------------------------------------------------------
/** A parametric narrow window.

  We define a family of windows with "width" 2/2^w,

    h(t,w) for t:[-1,1], w:{0,1,...}

  each of the simple polynomial form

    h(t,w) = ((t+1)(t-1))^n, for some s,n.

  h(t,w) is defined by the induction:

    h(t,0) is the constant function 1 on [-1,1]

    Variance[h(-,w+1)]
    ------------------ = 1/4
     Variance[h(-,w)]

  To derive n as a function of w,
  consider first the simpler class of functions f(t,w) on [0,1],
  of the form

    f(t,w) = 2 (t (1-t))^(n/2)   for some n : real

  These functions have raw squared moments

                                   (n!)^2
    M0 = int t:[0,1]. f(t,w)^2 = 4 -------
                                   (2n+1)!

                                     n! (n+1)!
    M1 = int t:[0,1]. t f(t,w)^2 = 4 ---------
                                      (2n+2)!

                                       n! (n+2)!
    M2 = int t:[0,1]. t^2 f(t,w)^2 = 4 ---------
                                        (2n+3)!

  The normalized moments are thus

    m0 = 1

    m1 = M1/M0  = 1/2
                          (n+1)(n+2)               1
    m2 = M2/M0 - m1^2  = ------------ - 1/4  = --------
                         (2n+2)(2n+3)          8 n + 12

  For w = 0, the constant function requires n = 0, which hase base variance

    V(0) = (n:=0. m2)  = 1/12

  Now setting

             1           1
    V(w) = ------  = --------
           12 4^w    8 n + 12

  and solving for n, we find

    n = 3/2 (4^w - 1)

  For w = 0, n = 0; for all other cases w > 0, n is a half-integer.

  In this implementation, width = 2, when w = 0, so

    power = n/2
    width = 2^(1-w)

  whence

    2^w = 2 / width
    power = 3/4 (4^w - 1)
          = 3/4 (4/width^2 - 1)
          = 3 width^2 - 0.75
*/
struct NarrowWindow : public FunctionAndDeriv
{
  const float power;

  NarrowWindow (double width) : power(3 / sqr(width) - 0.75)
  {
    ASSERT_LE(width, 1);
  }
  virtual ~NarrowWindow () {}

  virtual float value (float t) const
  {
    return pow((t + 1.0f) * (1.0f - t), power);
  }
  virtual float deriv (float t) const
  {
    return -2 * power * t * pow((t + 1.0f) * (1.0f - t), power - 1);
  }
};

#endif // KAZOO_WINDOW_H
