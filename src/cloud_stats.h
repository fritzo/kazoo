#ifndef KAZOO_CLOUD_STATS_H
#define KAZOO_CLOUD_STATS_H

#include "common.h"

namespace Cloud
{

struct QuantizeStats
{
  double sum_H;
  double sum_U;
  double sum_U2;
  double sum_Ucold;

  QuantizeStats () : sum_H(0), sum_U(0), sum_U2(0), sum_Ucold(0) {}
  QuantizeStats (double H, double U, double U2, double Ucold)
    : sum_H(H), sum_U(U), sum_U2(U2), sum_Ucold(Ucold)
  {}

  void zero () { sum_H = 0; sum_U = 9; sum_U2 = 0; sum_Ucold = 0; }

  void operator+= (const QuantizeStats & other)
  {
    sum_H += other.sum_H;
    sum_U += other.sum_U;
    sum_U2 += other.sum_U2;
    sum_Ucold += other.sum_Ucold;
  }

  void operator/= (double sum_1)
  {
    sum_H /= sum_1;
    sum_U /= sum_1;
    sum_U2 /= sum_1;
    sum_Ucold /= sum_1;
  }

  double get_rms_error (double sum_1 = 1) { return sqrt(2 * sum_U / sum_1); }

  double mean_entropy (double sum_1 = 1) const { return sum_H / sum_1; }
  double mean_energy (double sum_1 = 1) const { return sum_U / sum_1; }
  double var_energy (double sum_1 = 1) const
  {
    return sum_U2 / sum_1 - sqr(mean_energy(sum_1));
  }
  double mean_energy_cold (double sum_1 = 1) const { return sum_Ucold / sum_1; }
};

struct ConstructStats
{
  double error;
  double info;
  double surprise;

  ConstructStats () : error(0), info(0), surprise(0) {}
  ConstructStats (double e, double i, double s)
    : error(e), info(i), surprise(s)
  {}

  void zero () { error = 0; info = 0; surprise = 0; }

  void operator+= (const ConstructStats & other)
  {
    error += other.error;
    info += other.info;
    surprise += other.surprise;
  }

  void operator/= (double sum_1)
  {
    error /= sum_1;
    info /= sum_1;
    surprise /= sum_1;
  }

  double get_rms_error (size_t num_probes = 1) const
  {
    return sqrt(error / num_probes);
  }

  double get_beta_step () const { return surprise / info; }
  double get_log_beta_step (double beta) const
  {
    return surprise / (beta * info);
  }
  double get_log_radius_step (double radius) const
  {
    return -sqr(radius) * surprise / info;
  }
};

} // namespace Cloud

#endif // KAZOO_CLOUD_STATS_H
