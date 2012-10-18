
#include "filters.h"

#define LIKE_TOL (1e-2f)
#define TIME_TOL (1e-8f)

namespace Filters
{

template<size_t size>
void NCP<size>::mix (float weight, NCP<size> other)
{
  if (weight < LIKE_TOL) return;
  if (weight > 1 / LIKE_TOL) { * this = other; return; }

  Array<float, size> dx = Ex - other.Ex;
  Ex += weight * other.Ex;

  float cross_prob = 2 * weight / sqr(1 + weight);
  Vxx += weight * other.Vxx + cross_prob * dx * dx;
}

template<size_t size>
void NCV<size>::mix (float weight, NCV<size> other)
{
  if (weight < LIKE_TOL) return;
  if (weight > 1 / LIKE_TOL) { * this = other; return; }

  Array<float, size> dx = Ex - other.Ex;
  Array<float, size> dy = Ey - other.Ey;
  Ex += weight * other.Ex;
  Ey += weight * other.Ey;

  float cross_prob = 2 * weight / sqr(1 + weight);
  Vxx += weight * other.Vxx + cross_prob * dx * dx;
  Vxy += weight * other.Vxy + cross_prob * dx * dy;
  Vyy += weight * other.Vyy + cross_prob * dy * dy;
}

template<size_t size>
void NCV<size>::advance (float dt, Array<float, size> process_noise)
{
  float ds = fabs(dt); if (ds < TIME_TOL) return;

  Ex += dt * Ey;

  Vxx += dt * ( 2 * Vxy
            + dt * ( Vyy
                   + ds / 3 * process_noise
                   )
            );
  Vxy += dt * ( Vyy
              + dt / 2 * process_noise
              );
  Vyy += ds * process_noise;
}

/** Kalman filter update.

  Variables:
    x = mean    x0 = initial mean
    P = cov     P0 = initial cov
    H = [1, 0]
    z = observed.mean
    R = observed.cov

  We want to find an updated mean x that minimizes

    Phi = ||x - x0||_P^2 + ||H x - z||_R^2

  The objective function gradient is

    dPhi' = P0^-1 |x - x0> + H' R^-1 |H x - z>

  which is zero when

    (P0^-1 + H' R^-1 H) x = P0^-1 x0 + H' R^-1 z

  whence

    x = (P0^-1 + H' R^-1 H) \ (P0^-1 x0 + H' R^-1 z)
      = (P0^-1 + H' R^-1 H) \ ((P0^-1 + H' R^-1 H) x0 + H' R^-1 (z - H x0))
      = x0 + (P0^-1 + H' R^-1 H) \ H' R^-1 (z - H x0)
      = x0 + P H' R^-1 (z - H x0)
      =: x0 + [ Pxx R^-1 dz,
                Pxy R^-1 dz ]

  where the updated covariance matrix is

    P = (P0^-1 + H' R^-1 H)^-1

  The velocity residual
  
    Pxy R^-1 dz
  
  can be used to estimate actual process noise.
*/

template<size_t size>
Array<float, size> NCV<size>::update (NCP<size> obs)
{
  Array<float, size> obs_info = 1 / obs.Vxx;

  invert_P();
  Vxx += obs_info;
  invert_P();

  Array<float, size> dx = obs_info * (obs.Ex - Ex);
  Ex += Vxx * dx;
  Ey += Vxy * dx;

  return Vxy * dx;
}

template<size_t size>
void NCV<size>::fuse (NCV<size> other)
{
  invert_P();
  other.invert_P();

  Array<float, size> dx
    = Vxx * Ex + other.Vxx * other.Ex
    + Vxy * Ey + other.Vxy * other.Ey;
  Array<float, size> dy
    = Vxy * Ex + other.Vxy * other.Ex
    + Vyy * Ey + other.Vyy * other.Ey;

  Vxx += other.Vxx;
  Vxy += other.Vxy;
  Vyy += other.Vyy;
  invert_P();

  Ex = Vxx * dx
     + Vxy * dy;
  Ey = Vxy * dx
     + Vyy * dy;
}

template<size_t size>
inline void NCV<size>::invert_P ()
{
  Array<float, size> det_I = 1 / (Vxx * Vyy - sqr(Vxy));

  Array<float, size> Ixx =  det_I * Vyy;
  Array<float, size> Ixy = -det_I * Vxy;
  Array<float, size> Iyy =  det_I * Vxx;

  Vxx = Ixx;
  Vxy = Ixy;
  Vyy = Iyy;
}

//----( template instantiation )----------------------------------------------

#define INSTANTIATE_TEMPLATES(size) \
  template void NCP<size>::mix(float, NCP<size>); \
  template void NCV<size>::advance(float, Array<float, size>); \
  template Array<float, size> NCV<size>::update(NCP<size>); \
  template void NCV<size>::fuse(NCV<size>);

INSTANTIATE_TEMPLATES(3)


} // namespace Filters

