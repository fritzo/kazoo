#ifndef KAZOO_GESTURES_H
#define KAZOO_GESTURES_H

#include "common.h"
#include "array.h"
#include "vectors.h"
#include "bounded.h"
#include "image_types.h"

#define GESTURES_MAX_FINGERS        (10)
#define GESTURES_MAX_TRACKS         (2 * GESTURES_MAX_FINGERS)
#define GESTURES_MAX_CHORDS         (choose_2(GESTURES_MAX_TRACKS))
#define GESTURES_CHORD_LENGTH_INCH  (6.0f)
#define GESTURES_FINGER_LENGTH_INCH (3.0f)
#define FINGERS_PER_HAND            (5)

// TODO rename 3D Finger -> FingerTip and add real 5D Finger class

namespace Gestures
{

//----( gesture recognition tools )-------------------------------------------

inline float soft_clamp_to_grid (float x, float hardness)
{
  ASSERT1_LE(0, hardness);
  ASSERT1_LE(hardness, 1);

  const float scale = 2 * M_PI;
  return x - hardness * sinf(scale * x) / scale;
}

inline float is_small (float speed)
{
  return 1 / (1 + sqr(speed));
}

// DEPRICATED
template<class Descriptor>
struct Estimator
{
  // this returns number of estimates
  virtual size_t estimate (
      Vector<Id> & ids,
      Vector<Descriptor> & descriptors) const = 0;
};

//----( descriptors )---------------------------------------------------------

class Finger : public float8
{
public:

  void clear () { float8::operator=(0.0f); }

  float get_energy () const { return data[0]; }
  float get_z      () const { return data[1]; }
  float get_x      () const { return data[2]; }
  float get_y      () const { return data[3]; }
  float get_z_t    () const { return data[4]; }
  float get_x_t    () const { return data[5]; }
  float get_y_t    () const { return data[6]; }
  float get_age    () const { return data[7]; }

  float3 get_pos () const { return float3(data[2], data[3], data[1]); }
  float3 get_vel () const { return float3(data[5], data[6], data[4]); }

  void get_pos (Image::Point & p) const { p.x = get_x(); p.y = get_y(); }
  void get_vel (Image::Point & v) const { v.x = get_x_t(); v.y = get_y_t(); }

  void set_energy (float value) { data[0] = value; }
  void set_z      (float value) { data[1] = value; }
  void set_x      (float value) { data[2] = value; }
  void set_y      (float value) { data[3] = value; }
  void set_z_t    (float value) { data[4] = value; }
  void set_x_t    (float value) { data[5] = value; }
  void set_y_t    (float value) { data[6] = value; }
  void set_age    (float value) { data[7] = value; }

  void set_pos (const float3 &  xyz)
  {
    data[2] = xyz[0];
    data[3] = xyz[1];
    data[1] = xyz[2];
  }
  void set_vel (const float3 & xyz)
  {
    data[5] = xyz[0];
    data[6] = xyz[1];
    data[4] = xyz[2];
  }

  void set_pos (const Image::Point & p) { set_x(p.x); set_y(p.y); }
  void set_vel (const Image::Point & v) { set_x_t(v.x); set_y_t(v.y); }

  float & energy () { return data[0]; }
  float & z      () { return data[1]; }
  float & x      () { return data[2]; }
  float & y      () { return data[3]; }
  float & z_t    () { return data[4]; }
  float & x_t    () { return data[5]; }
  float & y_t    () { return data[6]; }
  float & age    () { return data[7]; }

  float get_impact () const
  {
    float z = bound_to(0.0f, 1.0f, get_z());
    float z_t = max(0.0f, get_z_t());

    //return z * z_t;
    return sqrt(z) * z_t;
  }

  float distance_to (const Finger & other, float x_unit, float y_unit) const
  {
    float dx = (other[2] - data[2]) / x_unit;
    float dy = (other[3] - data[3]) / y_unit;
    return sqrtf(sqr(dx) + sqr(dy));
  }
  float angle_to (const Finger & other, float x_unit, float y_unit) const
  {
    float dx = (other[2] - data[2]) / x_unit;
    float dy = (other[3] - data[3]) / y_unit;
    return atan2(dy, dx);
  }

  Finger extrapolate (float dt) const
  {
    Finger result = * this;
    result.z() += result.get_z_t() * dt;
    result.x() += result.get_x_t() * dt;
    result.y() += result.get_y_t() * dt;
    result.age() += dt;
    return result;
  }
};

class Chord : public float12
{
public:

  void clear () { float12::operator=(0.0f); }

  float get_energy   () const { return data[0]; }
  float get_z        () const { return data[1]; }
  float get_x        () const { return data[2]; }
  float get_y        () const { return data[3]; }
  float get_length   () const { return data[4]; }
  float get_angle    () const { return data[5]; }
  float get_x_t      () const { return data[7]; }
  float get_y_t      () const { return data[8]; }
  float get_z_t      () const { return data[6]; }
  float get_length_t () const { return data[9]; }
  float get_angle_t  () const { return data[10]; }
  float get_age      () const { return data[11]; }

  void set_energy   (float value) { data[0] = value; }
  void set_z        (float value) { data[1] = value; }
  void set_x        (float value) { data[2] = value; }
  void set_y        (float value) { data[3] = value; }
  void set_length   (float value) { data[4] = value; }
  void set_angle    (float value) { data[5] = value; }
  void set_x_t      (float value) { data[7] = value; }
  void set_y_t      (float value) { data[8] = value; }
  void set_z_t      (float value) { data[6] = value; }
  void set_length_t (float value) { data[9] = value; }
  void set_angle_t  (float value) { data[10] = value; }
  void set_age      (float value) { data[11] = value; }

  void set (
      const Finger & f,
      const Finger & g,
      float x_unit,
      float y_unit)
  {
    // first compute relative coordinates
    float u = (f.get_x() - g.get_x()) / x_unit;
    float v = (f.get_y() - g.get_y()) / y_unit;
    float u_t = (f.get_x_t() - g.get_x_t()) / x_unit;
    float v_t = (f.get_y_t() - g.get_y_t()) / y_unit;
    float r2 = sqr(u) + sqr(v);
    float r = sqrtf(r2);

    float blend = max(0.0f, 1 - r2);

    set_energy((f.get_energy() + g.get_energy()) * blend);

    set_z(f.get_z() * g.get_z() * blend);
    set_x((f.get_x() + g.get_x()) / 2);
    set_y((f.get_y() + g.get_y()) / 2);

    set_length(sqrtf(r2));
    set_angle(wrap(atan2(v, u) / M_PI));

    set_z_t(NAN); // TODO
    set_x_t((f.get_x_t() + g.get_x_t()) / 2);
    set_y_t((f.get_y_t() + g.get_y_t()) / 2);

    set_length_t((u_t * u + v_t * v) / r);
    set_angle_t((v_t * u - u_t * v) / r2);

    set_age(min(f.get_age(), g.get_age()));
  }
};

//----( gesture recognition )-------------------------------------------------

void fingers_to_chords (
    const BoundedMap<Id, Finger> & fingers,
    BoundedMap<Id, Chord> & chords,
    float chord_width = GESTURES_CHORD_LENGTH_INCH / GRID_SPACING_X_INCH,
    float chord_height = GESTURES_CHORD_LENGTH_INCH / GRID_SPACING_Y_INCH);

void relativize_fingers (
    BoundedMap<Id, Finger> & hands,
    BoundedMap<Id, Finger> & fingers,
    float sigma_x = GESTURES_FINGER_LENGTH_INCH / GRID_SPACING_X_INCH,
    float sigma_y = GESTURES_FINGER_LENGTH_INCH / GRID_SPACING_Y_INCH);

} // namespace Gestures

#endif // KAZOO_GESTURES_H
