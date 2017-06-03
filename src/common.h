
#ifndef KAZOO_COMMON_H
#define KAZOO_COMMON_H

#include "cxx0x.h"
#include <cstdlib>  // for exit() & abort();
#include <iostream>
#include <string>
#include <cmath>
#include <complex>
#include <cstdint>
#include <unistd.h>  // for usleep

using std::cin;
using std::cout;
using std::cerr;
using std::flush;
using std::endl;
using std::ostream;
using std::istream;
using std::string;

extern const char * kazoo_logo;
void chdir_kazoo ();

//----( global parameters )---------------------------------------------------

// TODO switch macros to global constants
#define DEFAULT_SAMPLE_RATE             (48000)
#define DEFAULT_FRAMES_PER_BUFFER       (64)
#define DEFAULT_AUDIO_FRAMERATE         ( DEFAULT_SAMPLE_RATE \
                                        / DEFAULT_FRAMES_PER_BUFFER )
#define DEFAULT_VIDEO_FRAMERATE         (125)
#define DEFAULT_VIDEO_WIDTH             (320)
#define DEFAULT_VIDEO_HEIGHT            (240)
#define DEFAULT_SCREEN_FRAMERATE        (30)
#define DEFAULT_MIDI_MESSAGE_RATE       (DEFAULT_SAMPLE_RATE / 64.0f)
#define DEFAULT_GAIN_TIMESCALE_SEC      (120.0f)
#define DEFAULT_CHANGE_TIMESCALE_SEC    (0.5f)

// table
#define GRID_SIZE_X                     (6.5f)
#define GRID_SIZE_Y                     (6.5f)
#define GRID_SPACING_X_INCH             (7.0f)
#define GRID_SPACING_Y_INCH             (3.0f)

// music
#define MIDDLE_C_HZ                     (261.626f)
#define CONCERT_A_HZ                    (440.0f)
#define SEMITONE_INTERVAL               (logf(2) / 12)
#define GOLDEN_RATIO                    ((1 + sqrtf(5)) / 2)

// psychoacoustics
#define CRITICAL_BAND_WIDTH             (0.2f)
#define MIN_CHROMATIC_FREQ_HZ           (30.0f)
#define MAX_CHROMATIC_FREQ_HZ           (5e3f)
#define MAX_AUDIBLE_FREQ_HZ             (18e3f)
// this is the acuity at freq < 500Hz
// (see PFaM, Fastl & Zwicker, pp. 183)
#define LOW_FREQ_FM_JND_HZ              (3.6f)
#define MIN_PULSE_BPM                   (60.0f)
#define MAX_PULSE_BPM                   (150.0f)

#define DEG_TO_RAD                      (M_PI / 180.0)
#define RAD_TO_DEG                      (180.0 / M_PI)

//----( compiler-specific )---------------------------------------------------

#ifndef __STDC_VERSION__
  #define __STDC_VERSION__ 199901L
#endif // __STDC_VERSION__

#ifdef __GNUG__
  #define restrict __restrict__
  #define no_inline __attribute__ ((noinline))
  #define DEPRECATED __attribute__ ((deprecated))
#else // __GNUG__
  #warning keyword 'restrict' ignored
  #define restrict
  #define no_inline
  #define DEPRECATED
#endif // __GNUG__

//----( logging )-------------------------------------------------------------

#define QUOTE(str) # str

#define LOG(mess) { cout << mess << endl; }
#define DEBUG(mess) LOG("\033[1;33mDEBUG\033[0m " << mess)
#define PRINT(arg) LOG(#arg " = " << (arg))
#define PRINT2(arg1,arg2) LOG(#arg1 " = " << (arg1) << ", " \
                              #arg2 " = " << (arg2))
#define PRINT3(arg1,arg2,arg3) LOG(#arg1 " = " << (arg1) << ", " \
                                   #arg2 " = " << (arg2) << ", " \
                                   #arg3 " = " << (arg3))
#define PRINT4(arg1,arg2,arg3,arg4) LOG(#arg1 " = " << (arg1) << ", " \
                                        #arg2 " = " << (arg2) << ", " \
                                        #arg3 " = " << (arg3) << ", " \
                                        #arg4 " = " << (arg4))
#define PRINT5(arg1,arg2,arg3,arg4,arg5) LOG(#arg1 " = " << (arg1) << ", " \
                                             #arg2 " = " << (arg2) << ", " \
                                             #arg3 " = " << (arg3) << ", " \
                                             #arg4 " = " << (arg4) << ", " \
                                             #arg5 " = " << (arg5))
#define SAVE(key, value) LOG("SAVE " << (key) << " = " << (value))
#define SAVE_TO_PYTHON(name) { save_to_python(name, "data/temp." #name ".py"); }

#define ERROR(mess) {\
    cerr << "ERROR "\
         << mess << "\n\t"\
         << __FILE__ << " : " << __LINE__ << "\n\t"\
         << __PRETTY_FUNCTION__ << endl; \
    abort(); }

#define WARN(mess) {\
    cerr << "WARNING "\
         << mess << "\n\t"\
         << __FILE__ << " : " << __LINE__ << "\n\t"\
         << __PRETTY_FUNCTION__ << endl; }

#define ASSERT(cond, mess) { if (!(cond)) ERROR(mess); }
#define ASSERTW(cond, mess) { if (!(cond)) WARN(mess); }

#define ASSERT_NULL(x) ASSERT((x) == NULL, \
    "expected NULL " #x ",\n\tactual: " << (x))
#define ASSERT_EQ(x,y) ASSERT((x) == (y), \
    "expected " #x " = " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERT_NE(x,y) ASSERT((x) != (y), \
    "expected " #x " != " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERT_LT(x,y) ASSERT((x) < (y), \
    "expected " #x " < " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERT_LE(x,y) ASSERT((x) <= (y), \
    "expected " #x " <= " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERT_DIVIDES(x,y) ASSERT((y) % (x) == 0, \
    "expected " #y " to be a multiple of " << (x) << ",\n\tactual " << (y))
#define ASSERT_NONNEG(x) ASSERT(0 <= x, \
    "expected " #x " nonnegative,\n\tactual: " << (x))
#define ASSERT_FINITE(x) ASSERT(safe_isfinite(x), \
    "expected " #x " finite,\n\tactual: " << (x))

#define ASSERT_SIZE(vect, vect_size) { \
  ASSERT(vect.size==static_cast<size_t>(vect_size), \
      "vector '" # vect "' has wrong size " \
      << vect.size << ",\n\tshould be " << (vect_size)); }

#define ASSERTW_NULL(x) ASSERTW((x) == NULL, \
    "expected NULL " #x ",\n\tactual: " << (x))
#define ASSERTW_EQ(x,y) ASSERTW((x) == (y), \
    "expected " #x " = " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERTW_NE(x,y) ASSERTW((x) != (y), \
    "expected " #x " != " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERTW_LT(x,y) ASSERTW((x) < (y), \
    "expected " #x " < " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERTW_LE(x,y) ASSERTW((x) <= (y), \
    "expected " #x " <= " #y ",\n\tactual: " << (x) << " vs " << (y))
#define ASSERTW_DIVIDES(x,y) ASSERTW((y) % (x) == 0, \
    "expected " #y " to be a multiple of " << (x) << ",\n\tactual " << (y))
#define ASSERTW_NONNEG(x) ASSERTW(0 <= x, \
    "expected " #x " nonnegative,\n\tactual: " << (x))
#define ASSERTW_FINITE(x) ASSERTW(safe_isfinite(x), \
    "expected " #x " finite,\n\tactual: " << (x))

#ifndef KAZOO_NDEBUG
  #define ASSERT1(cond, mess) ASSERT(cond, mess)
  #define ASSERT1_EQ(x,y) ASSERT_EQ(x,y)
  #define ASSERT1_NE(x,y) ASSERT_NE(x,y)
  #define ASSERT1_LT(x,y) ASSERT_LT(x,y)
  #define ASSERT1_LE(x,y) ASSERT_LE(x,y)
  #define ASSERT1_DIVIDES(x) ASSERT_DIVIDES(x)
  #define ASSERT1_NONNEG(x) ASSERT_NONNEG(x)
  #define ASSERT1_FINITE(x) ASSERT_FINITE(x)
#else // KAZOO_NDEBUG
  #define ASSERT1(cond, mess)
  #define ASSERT1_EQ(x,y)
  #define ASSERT1_NE(x,y)
  #define ASSERT1_LT(x,y)
  #define ASSERT1_LE(x,y)
  #define ASSERT1_DIVIDES(x)
  #define ASSERT1_NONNEG(x)
  #define ASSERT1_FINITE(x)
#endif // KAZOO_NDEBUG

#define TODO(mess) { ERROR("\033[1;33mTODO\033[0m " << mess); }

// time
double get_elapsed_time ();
string get_date (bool hour = true);

class Timer
{
  double m_time, m_paused;
public:
  Timer () { reset(); }
  void reset () { m_time = m_paused = get_elapsed_time(); }
  void pause () { m_paused = get_elapsed_time(); }
  void resume () { m_time -= get_elapsed_time() - m_paused; }
  double elapsed () const { return get_elapsed_time() - m_time; }
};

//----( abstract objects )----------------------------------------------------

struct Named { const string name; Named (string n) : name(n) {} };

//----( datatypes )-----------------------------------------------------------

// TODO eliminate typedefs float and complex.
//typedef std::complex<float> complex;

typedef size_t              Id;
typedef std::complex<float> complex;

/** resolve over-encapsulation defect in c++0x complex<T> specification
  http://gcc.gnu.org/onlinedocs/libstdc++/manual/bk01pt01ch01s03.html
  http://www.open-std.org/jtc1/sc22/wg21/docs/lwg-defects.html#387
*/
inline float & real_ref (complex & z)
{
  return reinterpret_cast<float(&)[2]>(z)[0];
}
inline float & imag_ref (complex & z)
{
  return reinterpret_cast<float(&)[2]>(z)[1];
}

// these make finiteness testing safe even after optimization

inline bool safe_isfinite (float x)
{
  return (-HUGE_VALF < x) and (x < HUGE_VALF);
}
inline bool safe_isfinite (complex z)
{
  return safe_isfinite(z.real()) and safe_isfinite(z.imag());
}

//----( abstract functions )--------------------------------------------------

struct Function
{
  virtual ~Function () {}
  virtual float value (float t) const = 0;
  inline float operator() (float t) const { return value(t); }
};

struct FunctionAndDeriv : public Function
{
  virtual ~FunctionAndDeriv () {}
  virtual float deriv (float t) const = 0;
};

struct FunctionAndInverse : public Function
{
  virtual ~FunctionAndInverse () {}
  virtual float inverse (float t) const = 0;
};

struct FunctionDerivAndInverse : public FunctionAndInverse
{
  virtual ~FunctionDerivAndInverse () {}
  virtual float deriv (float t) const = 0;
};

class AffineFunction : public FunctionAndInverse
{
  const float m_scale;
  const float m_shift;
public:
  AffineFunction (float scale, float shift = 0)
    : m_scale(scale),
      m_shift(shift)
  {
    ASSERT(scale != 0, "AffineFunction is singular: scale = 0");
  }
  virtual ~AffineFunction () {}
  virtual float value (float t) const { return t * m_scale + m_shift; }
  virtual float inverse (float t) const { return (t - m_shift) / m_scale; }
};

class ConjugateFunction : public Function
{
  const FunctionAndInverse & m_conj;
  const Function & m_fun;
public:
  ConjugateFunction (const FunctionAndInverse & conj,
                     const Function & fun)
    : m_conj(conj),
      m_fun(fun)
  {}
  virtual ~ConjugateFunction () {}
  virtual float value (float t) const
  {
    return m_conj.inverse(m_fun(m_conj(t)));
  }
};

//----( memory tools )--------------------------------------------------------

//#define CHECK_ALIGNMENT
#ifdef CHECK_ALIGNMENT
#define ASSERT_ALIGNED(ptr) \
  ASSERT((reinterpret_cast<size_t>(ptr) % 16) == 0, \
         "pointer " # ptr " is not aligned, offset = " \
      << (reinterpret_cast<size_t>(ptr) % 16));
#else // CHECK_ALIGNMENT
#define ASSERT_ALIGNED(ptr)
#endif // CHECK_ALIGNMENT

void * malloc_aligned (size_t size, size_t alignment = 16)
  __attribute__ ((malloc));
void free_aligned (void * pointer); // just calls free

// To avoid padding of classes whos first members also derive from Aligned,
// each class T will derive from a unique template-instance class Aligned<T>.
// For example,
//   class float4 : public Aligned<float4> {...};
// See http://www.cantrip.org/emptyopt.html for explanation of standard.
// WARNING this functionality relies on the empty-base-class optimization.
template<class Derived>
struct Aligned
{
  void * operator new   (size_t size) { return malloc_aligned(size); }
  void * operator new[] (size_t size) { return malloc_aligned(size); }
  void operator delete   (void * ptr) { free_aligned(ptr); }
  void operator delete[] (void * ptr) { free_aligned(ptr); }

  Aligned () { ASSERT_ALIGNED(this); }
} __attribute__ ((aligned (16)));

inline size_t ceil4 (size_t size) { return (size + 3) / 4 * 4; }

inline float * malloc_float (size_t size, size_t alignment = 16)
{
  return (float *) malloc_aligned(sizeof(float) * size, alignment);
}
inline void free_float (float* pointer) { free_aligned(pointer); }

inline complex * malloc_complex (size_t size, size_t alignment = 16)
{
  return (complex *) malloc_aligned(sizeof(complex) * size, alignment);
}

inline void free_complex (complex* pointer) { free_aligned(pointer); }

void copy_float (
    const float * restrict source,
    float * restrict dest,
    size_t size);
void copy_complex (
    const complex * restrict source,
    complex * restrict dest,
    size_t size);

void zero_float (float * x, size_t size);
void zero_complex (complex * x, size_t size);
void zero_bytes (void * x, size_t size);

void print_float (const float * data, size_t size);
void print_complex (const complex * data, size_t size);

template<class Iter>
inline void delete_all (Iter i, Iter end) { while (i != end) delete *i++; }

//----( math )----------------------------------------------------------------

template<class T> inline T min (T x, T y) { return (x < y) ? x : y; }
template<class T> inline T max (T x, T y) { return (x > y) ? x : y; }
template<class T> inline void imax (T & x, const T & y) { if (y > x) x = y; }
template<class T> inline void imin (T & x, const T & y) { if (y < x) x = y; }

// clipping
template<class T> inline T bound_to (T LB, T UB, T x)
{
  return max(LB, min(UB, x));
}
inline void clip (float & x, float LB, float UB)
{
  if (not (x >= LB)) x = LB; else
  if (not (x <= UB)) x = UB;
}
inline float clipped (float x, float LB = 0, float UB = 1)
{
  if (x >= LB) {
    if (x <= UB) {
      return x;
    } else {
      return UB;
    }
  } else {
    return LB;
  }
}

inline int roundi (float x) { return lrintf(x); }
inline int roundi (double x) { return lrint(x); }
inline size_t roundu (float x) { return max(0l, lrintf(x)); }
inline size_t roundu (double x) { return max(0l, lrint(x)); }
template<class T> inline T frac_part (T x) { return x - floor(x); }

// complex phase
inline complex exp_2_pi_i (float t)
{
  t *= 2 * M_PI;
  return complex(cosf(t), sinf(t));
}

inline float dot (complex u, complex v)
{
  return u.real() * v.real() + u.imag() * v.imag();
}
// cross(u,v) = dot(i u, v)
inline float cross (complex u, complex v)
{
  return u.real() * v.imag() - u.imag() * v.real();
}

// wraps x to the closed interval [0, modulus]
inline float wrap (float x)
{
  return x - floor(x);
}
inline float wrap (float x, float modulus)
{
  return x - floor(x / modulus) * modulus;
}
inline float wrap (float x, float modulus, float offset)
{
  return wrap(x - offset, modulus) + offset;
}

// safely normalizes
inline complex phase_part (complex z)
{
  float norm_z = norm(z);
  return norm_z > 0 ? z / sqrtf(norm_z) : complex(1, 0);
}

template <class T> inline int cmp(const T& lhs, const T& rhs)
{ return int(lhs > rhs) - int(lhs < rhs); }

template <class T> inline T safe_div (T num, T denom)
{ return denom == 0 ? 0.0 : num / denom; }

template <class T> inline T sqr (const T& x) { return x*x; }
inline size_t choose_2 (size_t n) { return n * (n - 1) / 2; }

int log2i (int x);

template<int x> struct static_log2i
{
  enum { value = 1 + static_log2i<x/2>::value };
};
template<> struct static_log2i<1>
{
  enum { value = 0 };
};

inline float affine_sum (float x0, float x1, float t)
{
  return x0 + (x1 - x0) * t;
}
inline float affine_prod (float x0, float x1, float t)
{
  return x0 * powf(x1 / x0, t);
}

inline float sigmoid (float t) { return 1 / (1 + exp(-t)); }
inline float real_to_01 (float t) { return (1 + t / (1 + max(t,-t))) / 2; }
inline float real_to_std (float t) { return t / (1 + max(t,-t)); }

//----( random generators )---------------------------------------------------

inline uint32_t random_int () { return random(); }
inline uint32_t random_max () { return RAND_MAX; }

bool random_bit ();

inline float random_01 ()
{
  return (random_int()) * (1.0f / random_max());
}

inline float random_unif (float lb, float ub)
{
  return affine_sum(lb, ub, random_01());
}

inline float random_std ()
{
  // zero mean, unit variance
  const float scale = sqrt(12) / random_max();
  const float shift = -sqrt(12) / 2.0;
  return random_int() * scale + shift;
}

inline int random_choice (int size)
{
  return random_int() % size;
}

inline bool random_bernoulli (float mean)
{
  return random_01() < mean;
}

unsigned random_poisson (float mean);

inline complex random_normal_complex ()
{
  // this uses the Box-Muller transform
  return sqrtf(-2.0f * logf(random_01())) * exp_2_pi_i(random_01());
}

class RandomNormal
{
  float m_queue;
  bool m_ready;

public:

  RandomNormal () : m_queue(NAN), m_ready(false) {}
  float operator() ()
  {
    if (m_ready) {
      m_ready = false;
      return m_queue;
    } else {
      complex z = random_normal_complex();
      m_queue = z.real();
      m_ready = true;
      return z.imag();
    }
  }
};

//----( common structures )---------------------------------------------------

class Rectangle
{
protected:

  size_t m_width;
  size_t m_height;

public:

  Rectangle (size_t width, size_t height) : m_width(width), m_height(height) {}

  size_t size () const { return m_width * m_height; }
  size_t width () const { return m_width; }
  size_t height () const { return m_height; }
  float radius () const { return sqrtf(sqr(m_width) + sqr(m_height)); }

  Rectangle transposed () const { return Rectangle(m_height, m_width); }
  Rectangle scaled (float scale) const
  {
    return Rectangle(roundu(m_width * scale), roundu(m_height * scale));
  }

  bool operator== (const Rectangle & other) const
  {
    return (other.m_width == m_width) and (other.m_height == m_height);
  }
  bool operator!= (const Rectangle & other) const
  {
    return (other.m_width != m_width) or (other.m_height != m_height);
  }

  friend inline ostream & operator<< (ostream & o, const Rectangle & r)
  {
    return o << "Rectangle(" << r.width() << ", " << r.height() << ")";
  }
};

//----( unix magic )----------------------------------------------------------

void daemonize (const char * logfilename);

#endif // KAZOO_COMMON_H

