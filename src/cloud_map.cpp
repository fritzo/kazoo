
#include "cloud_map.h"
#include "eigen_gpu.h"
#include "gpu.h"
#include <iomanip>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

/** Results: GPU vs CPU

> time ./learn nopgu gg new \
    -d data/gloves/big.cloud \
    -c data/gloves/big.cloud \
    -o data/gloves/big-big.map

  real	62m1.322s
  user	208m1.570s
  sys	0m6.200s

> time ./learn gg new \
    -d data/gloves/big.cloud \
    -c data/gloves/big.cloud \
    -o data/gloves/big-big.map

  real	25m13.199s
  user	132m19.040s
  sys	0m18.100s
  XXX these are with HACK_TO_LIMIT_GPU_POWER_USAGE
  XXX with no such hack, GPU is about 2x faster

*/

#define LOG1(message)

namespace Cloud
{

MapOptimizer::MapOptimizer (const JointPrior & d, const JointPrior & c)
  : dom(d.dom),
    cod(c.dom),
    dom_flow(d),
    cod_flow(c),

    m_dom_flow_sum(dom_flow.joint.sum()),
    m_cod_flow_sum(cod_flow.joint.sum()),

    m_dom_prior(int(dom.size)),
    m_cod_prior(int(cod.size)),
    m_conditional(int(cod.size), int(dom.size)),
    m_joint(int(cod.size), int(dom.size)),

    m_temp_dom(int(dom.size)),
    m_temp_cod(int(cod.size)),

    // m_temp_dQ & cod_flow.joint have the same sparsity pattern
    m_temp_dQ(cod_flow.joint),
    m_temp_dQt(cod_flow.joint.transpose()),

    m_P(NULL),
    m_Pt(NULL),

    debug(false),
    logging(false)
{
  LOG("Building MapOptimizer : " << dom.size << " -> " << cod.size);
  dom.get_prior(m_dom_prior);
  cod.get_prior(m_cod_prior);

  ASSERT_LT(0, m_dom_prior.minCoeff());
  ASSERT_LT(0, m_cod_prior.minCoeff());

  const size_t X = dom.size;
  const size_t Y = cod.size;

  if (X > Y) m_cod_prior.array() *= float(X) / Y;
  if (Y > X) m_dom_prior.array() *= float(Y) / X;

  float tol = 1e-4f;
  ASSERT_LT(sqr(m_dom_prior.sum() - m_cod_prior.sum()), tol);

  PRINT2(density(m_dom_prior), density(m_cod_prior));
  PRINT2(density(dom_flow.joint), density(cod_flow.joint));
  PRINT2(dom_flow.joint.nonZeros(), cod_flow.joint.nonZeros());
}

MapOptimizer::~MapOptimizer ()
{
  if (m_P) delete m_P;
  if (m_Pt) delete m_Pt;
}

void MapOptimizer::update_joint ()
{
  const size_t Y = cod.size;

  const MatrixXf & H = m_conditional;
  MatrixXf & J = m_joint;
  const VectorXf & p = m_dom_prior;

  J = H.cwiseProduct(MatrixXf::Ones(Y,1) * p.transpose());
}

void MapOptimizer::update_conditional ()
{
  MatrixXf & H = m_conditional;
  const MatrixXf & J = m_joint;

  H = J;
  normalize_columns_l1(H);
}

void MapOptimizer::init (const MatrixXf & joint, float tol)
{
  const size_t X = dom.size;
  const size_t Y = cod.size;

  ASSERT_EQ(joint.rows(), int(Y));
  ASSERT_EQ(joint.cols(), int(X));

  m_joint = joint;

  constrain_joint(tol);
  update_conditional();
}

void MapOptimizer::init (const MatrixSf & joint, float tol)
{
  const size_t X = dom.size;
  const size_t Y = cod.size;

  ASSERT_EQ(joint.rows(), int(Y));
  ASSERT_EQ(joint.cols(), int(X));

  m_joint = joint;

  constrain_joint(tol);
  densify_joint(tol);
  constrain_joint(tol);

  update_conditional();
}

void MapOptimizer::init_random (float tol)
{
  const size_t XY = dom.size * cod.size;

  float * restrict J_ = & m_joint.coeffRef(0,0);

  for (size_t xy = 0; xy < XY; ++xy) {

    J_[xy] = expf(random_std());
  }

  constrain_joint(tol);
  update_conditional();
}

void MapOptimizer::add_noise (float tol)
{
  const size_t XY = dom.size * cod.size;

  float * restrict J_ = & m_joint.coeffRef(0,0);

  for (size_t xy = 0; xy < XY; ++xy) {

    J_[xy] *= expf(random_std());
  }

  constrain_joint(tol);
  update_conditional();
}

void MapOptimizer::solve (MatrixXf & dense_out, float tol, size_t max_steps)
{
  ASSERT_EQ(dense_out.rows(), int(cod.size));
  ASSERT_EQ(dense_out.cols(), int(dom.size));

  LOG("Optimizing map for at most " << max_steps << " iterations");

  LOG("-------------"
      "-------------"
      "-------------");
  LOG( std::setw(13) << "relentropy"
    << std::setw(13) << "density(J)"
    << std::setw(13) << "stepsize");

  MatrixXf & J = m_joint;
  MatrixXf & dJ = dense_out; // used as temporary

  double relentropy = INFINITY;
  float stepsize = NAN;

  Timer timer;

  size_t step = 0;
  while (step < max_steps) {
    if (logging) LOG(" step " << step << "/" << max_steps);

    double new_relentropy = compute_relentropy(dJ);

    if (new_relentropy > relentropy) {
      WARN(" terminating due to objective function increase");
      break;
    }

    if (relentropy - new_relentropy < tol) {
      LOG(" map optimization converged to tolerance");
      break;
    }

    relentropy = new_relentropy;

    // should we be adaptively selecting stepsize?

    precondition_direction(dJ);
    constrain_direction(dJ, tol);
    scale_stepsize(dJ);
    stepsize = (dJ.array() / J.array()).square().sum();

    LOG( std::setw(13) << relentropy
      << std::setw(13) << density(J)
      << std::setw(13) << stepsize);

    J += dJ;

    densify_joint(tol);
    constrain_joint(tol);
    update_conditional();

    ++step;
  }

  LOG( std::setw(13) << compute_relentropy()
    << std::setw(13) << density(J)
    << std::setw(13) << "-");
  LOG("-------------"
      "-------------"
      "-------------");

  float time = timer.elapsed();
  LOG(" optimization took " << time << " sec = "
      << (time / step) << " sec/step x " << step << " steps");

  dense_out = m_joint;
}

void MapOptimizer::solve (MatrixSf & sparse_out, float tol, size_t max_steps)
{
  MatrixXf dense(int(cod.size), int(dom.size));

  solve(dense, tol, max_steps);

  size_t max_entries = max_entries_heuristic(dense);
  sparsify_hard_relative_to_row_col_max(dense, sparse_out, tol, max_entries);
}

// We minimize the relentropy between Q and the projection H P' H
//
//   KL(Q || H P H')
//                                           Q(y,y')
//       = sum y,y'. Q(y,y') log ---------------------------------
//                               sum x,x'. H(y,x) P(x,x') H(y',x')
//
//       = sum y,y'. Q(y,y') log dQ(y,y')
//
// using gradient descent with the derivative
//
//   -d/dH KL(Q || H P H') = dQ H P' + dQ' H P
//
// where dQ = Q ./ (H P H')

double MapOptimizer::compute_relentropy () const
{
  if (logging) LOG("  computing relentropy");

  return Gpu::using_cuda()
    ? compute_relentropy_gpu()
    : compute_relentropy_cpu();
}

double MapOptimizer::compute_relentropy (MatrixXf & dH) const
{
  if (logging) LOG("  computing relentropy & descent direction");

  return Gpu::using_cuda()
    ? compute_relentropy_gpu(dH)
    : compute_relentropy_cpu(dH);
}

double MapOptimizer::compute_relentropy_cpu () const
{
  const MatrixXf & H = m_conditional;
  const MatrixSf & P = dom_flow.joint;

  MatrixXf HP = H * P;
  MatrixXf HPH = HP * H.transpose();

  const float sum_P = m_dom_flow_sum;
  const float sum_Q = m_cod_flow_sum;
  const float dQ_scale = sum_P / sum_Q;

  const MatrixSf & Q = cod_flow.joint;

  double relentropy = 0;
  for (int i = 0; i < Q.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(Q,i); iter; ++iter) {

      float Q_yy = iter.value();
      float HPH_yy = HPH(iter.row(), iter.col());

      relentropy += Q_yy * log(dQ_scale * Q_yy / HPH_yy);
    }
  }
  relentropy /= sum_Q;
  ASSERT_LE(0, relentropy);

  return relentropy;
}

double MapOptimizer::compute_relentropy_cpu (MatrixXf & dH) const
{
  const MatrixXf & H = m_conditional;
  const MatrixSf & P = dom_flow.joint;

  MatrixXf HP(H.rows(), H.cols());
  MatrixXf HPt(H.rows(), H.cols());

  #pragma omp parallel sections
  {
    #pragma omp section
    HP.noalias() = H * P;

    #pragma omp section
    HPt.noalias() = H * P.transpose();
  }

  MatrixXf HPH = HP * H.transpose();

  const float sum_P = m_dom_flow_sum;
  const float sum_Q = m_cod_flow_sum;
  const float dQ_scale = sum_P / sum_Q;

  const MatrixSf & Q = cod_flow.joint;
  MatrixSf & dQ = m_temp_dQ;

  double relentropy = 0;
  for (int i = 0; i < Q.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(Q,i); iter; ++iter) {

      const float Q_yy = iter.value();
      const float HPH_yy = HPH(iter.row(), iter.col());
      const float dQ_yy = dQ_scale * Q_yy / HPH_yy;

      relentropy += Q_yy * log(dQ_yy);

      dQ.coeffRef(iter.row(), iter.col()) = dQ_yy;
    }
  }
  relentropy /= sum_Q;
  ASSERT_LE(0, relentropy);

  MatrixXf dH2(H.rows(), H.cols());

  #pragma omp parallel sections
  {
    #pragma omp section
    dH.noalias() = dQ.transpose() * HP;

    #pragma omp section
    dH2.noalias() = dQ * HPt;
  }

  dH += dH2;

  return relentropy;
}

void MapOptimizer::initialize_P_gpu () const
{
  const MatrixSf & P = dom_flow.joint;
  if (not m_P) {
    m_P = new Gpu::SparseMultiplier(P);
  }

  if (not m_Pt) {
    MatrixSf dPt = P.transpose();
    m_Pt = new Gpu::SparseMultiplier(dPt);
  }
}

double MapOptimizer::compute_relentropy_gpu () const
{
  initialize_P_gpu();

  const MatrixXf & H = m_conditional;

  MatrixXf PHt = H.transpose();
  m_Pt->left_imul(PHt, true);

  MatrixXf HPHt(H.rows(), H.rows());
#ifdef HACK_TO_LIMIT_GPU_POWER_USAGE
  HPHt.noalias() = H * PHt;
#else // HACK_TO_LIMIT_GPU_POWER_USAGE
  Gpu::matrix_multiply(H, false, PHt, false, HPHt);
#endif // HACK_TO_LIMIT_GPU_POWER_USAGE

  const float sum_P = m_dom_flow_sum;
  const float sum_Q = m_cod_flow_sum;
  const float dQ_scale = sum_P / sum_Q;

  const MatrixSf & Q = cod_flow.joint;

  double relentropy = 0;
  for (int i = 0; i < Q.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(Q,i); iter; ++iter) {

      const float Q_yy = iter.value();
      const float HPH_yy = HPHt(iter.row(), iter.col());
      const float dQ_yy = dQ_scale * Q_yy / HPH_yy;

      relentropy += Q_yy * log(dQ_yy);
    }
  }
  relentropy /= sum_Q;
  ASSERT_LE(0, relentropy);

  return relentropy;
}

double MapOptimizer::compute_relentropy_gpu (MatrixXf & dH) const
{
  initialize_P_gpu();

  const MatrixXf & H = m_conditional;

  MatrixXf PHt = H.transpose();
  MatrixXf PtHt = PHt;

  //TODO run cusparse in two different cuda streams
  m_P->left_imul(PtHt, true);
  m_Pt->left_imul(PHt, true);

  MatrixXf HPHt(H.rows(), H.rows());
#ifdef HACK_TO_LIMIT_GPU_POWER_USAGE
  HPHt.noalias() = H * PHt;
#else // HACK_TO_LIMIT_GPU_POWER_USAGE
  Gpu::matrix_multiply(H, false, PHt, false, HPHt);
#endif // HACK_TO_LIMIT_GPU_POWER_USAGE

  const float sum_P = m_dom_flow_sum;
  const float sum_Q = m_cod_flow_sum;
  const float dQ_scale = sum_P / sum_Q;

  const MatrixSf & Q = cod_flow.joint;
  MatrixSf & dQ = m_temp_dQ;
  MatrixSf & dQt = m_temp_dQt;

  double relentropy = 0;
  for (int i = 0; i < Q.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(Q,i); iter; ++iter) {

      const float Q_yy = iter.value();
      const float HPH_yy = HPHt(iter.row(), iter.col());
      const float dQ_yy = dQ_scale * Q_yy / HPH_yy;

      relentropy += Q_yy * log(dQ_yy);

      dQ.coeffRef(iter.row(), iter.col()) = dQ_yy;
      dQt.coeffRef(iter.col(), iter.row()) = dQ_yy;
    }
  }
  relentropy /= sum_Q;
  ASSERT_LE(0, relentropy);

  MatrixXf HP(H.rows(), H.cols());
  MatrixXf HPt(H.rows(), H.cols());

  #pragma omp sections
  {
    #pragma omp section
    HP = PtHt.transpose();

    #pragma omp section
    HPt = PHt.transpose();
  }

  //TODO run cusparse in two different cuda streams
  Gpu::SparseMultiplier(dQ).left_mul(HP, dH, true);
  Gpu::SparseMultiplier(dQt).left_fma(HPt, dH, true);

  return relentropy;
}

void MapOptimizer::precondition_direction (MatrixXf & dH) const
{
  if (logging) LOG("  preconditioning direction");

  // Transform descent direction from H(y,x) coords to log(H(y,x)) coords,
  // which are equivalent to log(J(y,x)) coords (so hereafter dH = dJ).

  const MatrixXf & H = m_conditional;

  dH.array() *= H.array();
}

void MapOptimizer::constrain_direction (MatrixXf & dJ, float tol) const
{
  // Enforce the simultaneous constraints
  //
  //   /\x. sum y. dJ(y,x) = 0
  //   /\y. sum x. dJ(y,x) = 0
  //
  // We combine the two constraints by iteratively weakly enforcing both:
  // Let Px,Py project to the feasible subspaces for constraints 1,2, resp.
  // Each projection has eigenvalues in {0,1}.
  // We approximate the desired projection Pxy as a linear combination of Px,Py
  //   Pxy' = 1 - alpha ((1-Px) + (1-Py))
  // which has eigenvalues in {1} u [1 - alpha, 1 - 2 alpha].
  // Hence Pxy = lim n->infty Pxy'^n, where convergence rate depends on alpha.
  // The optimal alpha is 2/3, yielding Pxy' eigenvalues in {1} u [-1/3,1/3],
  // and resulting in project_scale = -alpha below.

  if (logging) LOG("  constraining direction");

  const size_t X = dom.size;
  const size_t Y = cod.size;

  const MatrixXf & J = m_joint;
  const VectorXf & sum_y_J = m_dom_prior;
  const VectorXf & sum_x_J = m_cod_prior;
  const float sum_xy_J = m_cod_prior.sum();

  VectorXf sum_y_dJ(J.cols());
  VectorXf sum_x_dJ(J.rows());

  // this is iterative, so we hand-optimize by merging loops

  const float * restrict J_ = J.data();
  const float * restrict sum_y_J_ = sum_y_J.data();
  const float * restrict sum_x_J_ = sum_x_J.data();

  float * restrict dJ_ = dJ.data();
  float * restrict project_y_ = sum_y_dJ.data();
  float * restrict project_x_ = sum_x_dJ.data();
  const float project_scale = -2/3.0;

  Vector<float> accum_x_dJ(Y);
  float * restrict accum_x_dJ_ = accum_x_dJ;

  // accumulate first projection
  accum_x_dJ.zero();

  for (size_t x = 0; x < X; ++x) {

    const float * restrict dJ_x_ = dJ_ + Y * x;

    float accum_y_dJ = 0;

    for (size_t y = 0; y < Y; ++y) {

      float dJ_xy = dJ_x_[y];

      accum_y_dJ += dJ_xy;
      accum_x_dJ_[y] += dJ_xy;
    }

    project_y_[x] = project_scale * accum_y_dJ / sum_y_J_[x];
  }

  for (size_t y = 0; y < Y; ++y) {
    project_x_[y] = project_scale * accum_x_dJ_[y] / sum_x_J_[y];
    accum_x_dJ_[y] = 0;
  }

  // apply previous projection and accumulate next projection
  for (size_t iter = 0; iter < 100; ++iter) {

    float error = 0;

    for (size_t x = 0; x < X; ++x) {

      const float * restrict J_x_ = J_ + Y * x;
      float * restrict dJ_x_ = dJ_ + Y * x;

      float accum_y_dJ = 0;

      for (size_t y = 0; y < Y; ++y) {

        float dJ_xy = dJ_x_[y] += J_x_[y] * (project_x_[y] + project_y_[x]);

        accum_y_dJ += dJ_xy;
        accum_x_dJ_[y] += dJ_xy;
      }

      project_y_[x] = project_scale * accum_y_dJ / sum_y_J_[x];
      imax(error, max(-accum_y_dJ, accum_y_dJ));
    }

    for (size_t y = 0; y < Y; ++y) {

      float accum_x_dJ_y = accum_x_dJ_[y];
      accum_x_dJ_[y] = 0;

      project_x_[y] = project_scale * accum_x_dJ_y / sum_x_J_[y];
      imax(error, max(-accum_x_dJ_y, accum_x_dJ_y));
    }

    if (error < tol) {
      if (logging) {
        LOG("   after " << (1+iter) << " iterations, error < " << error);
      }
      break;
    }
  }

  // apply final projection
  for (size_t x = 0; x < X; ++x) {

    const float * restrict J_x_ = J_ + Y * x;
    float * restrict dJ_x_ = dJ_ + Y * x;

    for (size_t y = 0; y < Y; ++y) {
      dJ_x_[y] += J_x_[y] * (project_x_[y] + project_y_[x]);
    }
  }

  if (debug) {

    sum_y_dJ = dJ.colwise().sum();
    sum_x_dJ = dJ.rowwise().sum();
    float sum_xy_dJ = sum_x_dJ.sum();

    DEBUG("max constraint errors = "
        << sqrt(sum_x_dJ.array().square().maxCoeff())<< ", "
        << sqrt(sum_y_dJ.array().square().maxCoeff())<< ", "
        << sum_xy_dJ);

    sum_y_dJ.array() /= sum_y_J.array();
    sum_x_dJ.array() /= sum_x_J.array();
    sum_xy_dJ /= sum_xy_J;

    DEBUG("max relative constraints errors = "
        << sqrt(sum_x_dJ.array().square().maxCoeff()) << ", "
        << sqrt(sum_y_dJ.array().square().maxCoeff()) << ", "
        << sum_xy_dJ);

    DEBUG("max(|dJ|) = " << dJ.array().abs().maxCoeff()
        << ", rms(dJ) = " << sqrt(dJ.array().square().mean()));
    DEBUG("max(J) / min(J) = " << (J.maxCoeff() / J.minCoeff()));
    DEBUG("max(sum x. J) / min(sum x. J) = "
        << (sum_x_J.maxCoeff() / sum_x_J.minCoeff()));
    DEBUG("max(sum y. J) / min(sum y. J) = "
        << (sum_y_J.maxCoeff() / sum_y_J.minCoeff()));
  }
}

void MapOptimizer::scale_stepsize (MatrixXf & dJ) const
{
  if (logging) LOG("  computing stepsize");

  const size_t XY = dom.size * cod.size;

  const float * restrict J_ = m_joint.data();
  const float * restrict dJ_ = dJ.data();

  float time_to_boundary = INFINITY;

  for (size_t xy = 0; xy < XY; ++xy) {

    float dJ_xy = dJ_[xy];
    if (dJ_xy < 0) {

      imin(time_to_boundary, -J_[xy] / dJ_xy);
    }
  }

  float keepaway = 1 / M_E;
  float scale = time_to_boundary * keepaway;
  if (logging) LOG("   scaling step size by " << scale);

  dJ *= scale;
}

void MapOptimizer::densify_joint (float tol)
{
  // add a completely uniform prior just below the sparsify threshold

  if (logging) LOG("  bounding joint probabilities away from zero");

  MatrixXf & J = m_joint;
  VectorXf dom_thresh(J.cols());
  VectorXf cod_thresh(J.rows());

  dom_thresh = J.colwise().maxCoeff();
  cod_thresh = J.rowwise().maxCoeff();

  const size_t X = dom.size;
  const size_t Y = cod.size;

  const float * restrict dom_thresh_ = dom_thresh.data();
  const float * restrict cod_thresh_ = cod_thresh.data();
  const float * restrict dom_prior_ = m_dom_prior.data();
  const float * restrict cod_prior_ = m_cod_prior.data();

  float scale = INFINITY;
  for (size_t x = 0; x < X; ++x) {
    for (size_t y = 0; y < Y; ++y) {

      float thresh = min(dom_thresh_[x], cod_thresh_[y]);
      float prior = dom_prior_[x] * cod_prior_[y];

      imin(scale, thresh / prior);
    }
  }

  scale *= 0.1f * tol;
  ASSERT_LT(0, scale);

  J += scale * m_cod_prior * m_dom_prior.transpose();
}

void MapOptimizer::constrain_joint (float tol)
{
  constrain_marginals_bp(
      m_joint,
      m_dom_prior,
      m_cod_prior,
      m_temp_dom,
      m_temp_cod,
      tol);
}

} // namespace Cloud

