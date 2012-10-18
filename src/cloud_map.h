#ifndef KAZOO_CLOUD_MAP_H
#define KAZOO_CLOUD_MAP_H

#include "common.h"
#include "cloud_points.h"

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

namespace Gpu { class SparseMultiplier; }

namespace Cloud
{

class MapOptimizer
{
public:

  const PointSet & dom;
  const PointSet & cod;
  const JointPrior & dom_flow;
  const JointPrior & cod_flow;

protected:

  const float m_dom_flow_sum;
  const float m_cod_flow_sum;

  VectorXf m_dom_prior;
  VectorXf m_cod_prior;
  MatrixXf m_conditional;
  MatrixXf m_joint;

  mutable VectorXf m_temp_dom;
  mutable VectorXf m_temp_cod;

  mutable MatrixSf m_temp_dQ;
  mutable MatrixSf m_temp_dQt;

  mutable Gpu::SparseMultiplier * m_P;
  mutable Gpu::SparseMultiplier * m_Pt;

public:

  bool debug;
  bool logging;

  MapOptimizer (const JointPrior & dom, const JointPrior & cod);
  ~MapOptimizer ();

  void init (const MatrixXf & initial_guess, float tol);
  void init (const MatrixSf & initial_guess, float tol);
  void init_random (float tol);
  void add_noise (float tol);

  void solve (MatrixXf & dense_out, float tol, size_t max_steps = 100);
  void solve (MatrixSf & sparse_out, float tol, size_t max_steps = 100);

protected:

  void update_joint ();
  void update_conditional ();

  double compute_relentropy () const;
  double compute_relentropy (MatrixXf & direction) const;
  double compute_relentropy_cpu () const;
  double compute_relentropy_cpu (MatrixXf & direction) const;
  double compute_relentropy_gpu () const;
  double compute_relentropy_gpu (MatrixXf & direction) const;

  void initialize_P_gpu () const;

  void precondition_direction (MatrixXf & direction) const;
  void constrain_direction (MatrixXf & direction, float tol) const;
  void scale_stepsize (MatrixXf & direction) const;

  void densify_joint (float tol);
  void constrain_joint (float tol);
};

} // namespace Cloud

#endif // KAZOO_CLOUD_MAP_H
