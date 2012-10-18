
#include "cloud_math.h"
#include <algorithm>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#define LOG1(message)

namespace Cloud
{

//----( random numbers )------------------------------------------------------

void randomize (Vector<float> x, float sigma)
{
  const size_t x_size = x.size;
  float * restrict x_data = x.data;

  for (size_t i = 0; i < x_size; ++i) {
    x_data[i] = sigma * random_std();
  }
}

size_t random_index (const Vector<float> & likes)
{
  float total = sum(likes);
  ASSERT_LT(0, total);

  while (true) {
    float t = random_unif(0, total);

    for (int i = 0, I = likes.size; i < I; ++i) {

      t -= likes[i];
      if (t < 0) return i;
    }
  }
}

size_t random_index (const VectorXf & likes)
{
  float total = likes.sum();
  ASSERT_LT(0, total);

  while (true) {
    float t = random_unif(0, total);

    for (int i = 0, I = likes.size(); i < I; ++i) {

      t -= likes(i);
      if (t < 0) return i;
    }
  }
}

int random_index (const VectorSf & likes)
{
  float total = likes.sum();
  ASSERT_LT(0, total);

  while (true) {
    float t = random_unif(0, total);

    for (VectorSf::InnerIterator iter(likes); iter; ++iter) {

      t -= iter.value();
      if (t < 0) return iter.index();
    }
  }
}

void generate_noise (Vector<int8_t> & noise, float sigma)
{
  const float quantization_correction = 1 / 6.0f;
  sigma = sqrtf(sqr(sigma) + quantization_correction);

  for (size_t i = 0, I = noise.size; i < I; ++i) {
    noise[i] = roundi(sigma * random_std());
  }
}

//----( normalization )-------------------------------------------------------

void normalize_l1 (VectorXf & x, float tot) { x *= tot / x.sum(); }
void normalize_l1 (VectorSf & x, float tot) { x *= tot / x.sum(); }
void normalize_l1 (MatrixXf & x, float tot) { x *= tot / x.sum(); }
void normalize_l1 (MatrixSf & x, float tot) { x *= tot / x.sum(); }

void normalize_rows_l1 (size_t I, size_t J, Vector<float> & A)
{
  for (size_t i = 0; i < I; ++i) {

    Vector<float> row = A.block(J, i);

    row /= sum(row);
  }
}

void normalize_columns_l1 (MatrixXf & A)
{
  const int I = A.rows();
  const int J = A.cols();

  for (int j = 0; j < J; ++j) {

    Vector<float> col(I, & A.coeffRef(0,j));

    col /= sum(col);
  }
}

void normalize_columns_l1 (MatrixXd & A)
{
  const int I = A.rows();
  const int J = A.cols();

  for (int j = 0; j < J; ++j) {

    Vector<double> col(I, & A.coeffRef(0,j));

    col /= sum(col);
  }
}

void normalize_columns_l1 (MatrixSf & A)
{
  // WARNING this assumes column-major ordering

  for (int i = 0; i < A.outerSize(); ++i) {
    A.col(i) *= 1.0f / A.col(i).sum();
  }
}

//----( sparse tools )--------------------------------------------------------

double density (const VectorXf & x)
{
  const size_t I = x.size();

  double sum_x1 = 0;
  double sum_x2 = 0;

  const float * restrict x_ = x.data();

  for (size_t i = 0; i < I; ++i) {
    double xi = x_[i];

    sum_x1 += max(-xi,xi);
    sum_x2 += xi * xi;
  }

  return sqr(sum_x1) / sum_x2 / I;
}

float density (const VectorSf & x)
{
  float sum_x1 = 0;
  float sum_x2 = 0;

  for (VectorSf::InnerIterator iter(x); iter; ++iter) {
    float xi = iter.value();

    sum_x1 += max(-xi,xi);
    sum_x2 += xi * xi;
  }

  return sqr(sum_x1) / sum_x2 / x.size();
}

double density (const MatrixXf & x)
{
  const size_t I = x.rows() * x.cols();

  double sum_x1 = 0;
  double sum_x2 = 0;

  const float * restrict x_ = x.data();

  for (size_t i = 0; i < I; ++i) {
    double xi = x_[i];

    sum_x1 += max(-xi,xi);
    sum_x2 += xi * xi;
  }

  return sqr(sum_x1) / sum_x2 / I;
}

double density (const MatrixSf & x)
{
  double sum_x1 = 0;
  double sum_x2 = 0;

  for (int i = 0; i < x.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(x,i); iter; ++iter) {
      double xi = iter.value();

      sum_x1 += max(-xi,xi);
      sum_x2 += xi * xi;
    }
  }

  return sqr(sum_x1) / sum_x2 / x.size();
}

namespace {
struct SparseVectorEntry
{
  float value;
  int index;

  SparseVectorEntry () {}
  SparseVectorEntry (float v, int i) : value(v), index(i) {}

  bool operator< (const SparseVectorEntry & other) const
  {
    return value > other.value;
  }
};
struct SparseMatrixEntry
{
  float value;
  int row;
  int col;

  SparseMatrixEntry () {}
  SparseMatrixEntry (float v, int i, int j) : value(v), row(i), col(j) {}

  bool operator< (const SparseMatrixEntry & other) const
  {
    return value > other.value;
  }
};
} // anonymous namespace

float sparsify_size (VectorSf & sparse, int size)
{
  ASSERT_LT(0, size);
  ASSERT_LT(size, sparse.nonZeros());

  static std::vector<SparseVectorEntry> entries;

  for (VectorSf::InnerIterator iter(sparse); iter; ++iter) {
    entries.push_back(SparseVectorEntry(iter.value(), iter.index()));
  }

  std::nth_element(entries.begin(), entries.begin() + size, entries.end());

  VectorSf sparser(sparse.size());
  sparser.reserve(size);

  for (int e = 0; e < size; ++e) {
    const SparseVectorEntry & entry = entries[e];
    sparser.insert(entry.index) = entry.value;
  }

  sparser.finalize();
  entries.clear();

  std::swap(sparse, sparser);

  return sparser.sum() - sparse.sum();
}

float sparsify_size (MatrixSf & sparse, int size)
{
  ASSERT_LT(0, size);
  ASSERT_LT(size, sparse.nonZeros());

  LOG("sparsifying " << sparse.rows() << " x " << sparse.cols()
      << " matrix from " << sparse.nonZeros() << " to " << size << " entries");

  static std::vector<SparseMatrixEntry> entries;

  for (int i = 0; i < sparse.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(sparse,i); iter; ++iter) {
      entries.push_back(
          SparseMatrixEntry(iter.value(), iter.row(), iter.col()));
    }
  }

  std::nth_element(entries.begin(), entries.begin() + size, entries.end());

  MatrixSf sparser(sparse.rows(), sparse.cols());
  sparser.reserve(size);

  for (int e = 0; e < size; ++e) {
    const SparseMatrixEntry & entry = entries[e];
    sparser.insert(entry.row, entry.col) = entry.value;
  }

  sparser.finalize();
  entries.clear();

  std::swap(sparse, sparser);

  float old_sum = sparser.sum();
  float new_sum = sparse.sum();
  float loss = (old_sum - new_sum) / old_sum;
  float density = float(size) / sparser.nonZeros();
  LOG("sparsifying to density " << density
      << " loses " << (loss*100) << "% of mass");

  return old_sum - new_sum;
}

float sparsify_absolute (
    const VectorXf & dense,
    VectorSf & sparse,
    float thresh)
{
  ASSERT_LE(0, thresh);

  const int I = dense.size();

  sparse.resize(I);

  float loss = 0;

  for (int i = 0; i < I; ++i) {

    const float dense_i = dense(i);
    const float abs_i = fabsf(dense_i);

    if (abs_i > thresh) sparse.insert(i) = dense_i;
    else loss += abs_i;
  }

  sparse.finalize();

  return loss;
}

float sparsify_absolute (
    const VectorSf & sparse,
    VectorSf & sparser,
    float thresh)
{
  ASSERT_LE(0, thresh);

  sparser.resize(sparse.size());

  float loss = 0;

  for (VectorSf::InnerIterator iter(sparse); iter; ++iter) {

    const float value_i = iter.value();
    const float abs_i = fabsf(value_i);

    if (abs_i > thresh) sparser.insert(iter.index()) = value_i;
    else loss += abs_i;
  }

  sparser.finalize();

  return loss;
}

float sparsify_absolute (
    const VectorXf & dense,
    VectorSf & sparse,
    const Vector<float> & thresh)
{
  ASSERT1_LE(0, min(thresh));

  const int I = dense.size();

  sparse.resize(I);

  float loss = 0;

  for (int i = 0; i < I; ++i) {

    const float dense_i = dense(i);
    const float abs_i = fabsf(dense_i);

    if (abs_i > thresh[i]) sparse.insert(i) = dense_i;
    else loss += abs_i;
  }

  sparse.finalize();

  return loss;
}

void sparsify_absolute (const MatrixXf & dense, MatrixSf & sparse, float thresh)
{
  ASSERT_LE(0, thresh);

  LOG("sparsifying " << dense.rows() << " x " << dense.cols()
      << " matrix to threshold " << thresh);

  const int I = dense.rows();
  const int J = dense.cols();

  sparse.resize(I,J);

  double sum_dense = 0;
  double sum_sparse = 0;

  for (int j = 0; j < J; ++j) {
    for (int i = 0; i < I; ++i) {

      const float dense_ij = dense(i,j);
      const float abs_ij = fabsf(dense_ij);
      sum_dense += abs_ij;

      if (abs_ij > thresh) {

        sparse.insert(i,j) = dense_ij;
        sum_sparse += abs_ij;
      }
    }
  }

  sparse.finalize();

  float density = sparse.nonZeros() / float(I * J);
  float loss = (sum_dense - sum_sparse) / sum_dense;
  LOG("sparsifying to density " << density
      << " loses " << (100 * loss) << "% of mass");
}

void sparsify_soft_relative_to_row_col_max (
    const MatrixXf & dense,
    MatrixSf & sparse,
    float relthresh,
    bool ignore_diagonal)
{
  ASSERT_LT(0, relthresh);

  LOG("sparsifying " << dense.rows() << " x " << dense.cols()
      << " positive matrix to relative threshold " << relthresh);

  VectorXf row_max;
  VectorXf col_max;

  if (ignore_diagonal) {

    VectorXf diag = dense.diagonal();
    const_cast<MatrixXf &>(dense).diagonal().setZero();

    row_max = dense.rowwise().maxCoeff();
    col_max = dense.colwise().maxCoeff();

    const_cast<MatrixXf &>(dense).diagonal() = diag;

  } else {

    row_max = dense.rowwise().maxCoeff();
    col_max = dense.colwise().maxCoeff();

  }

  const int I = dense.rows();
  const int J = dense.cols();

  sparse.resize(I,J);

  double sum_dense = 0;
  double sum_sparse = 0;

  for (int j = 0; j < J; ++j) {
    for (int i = 0; i < I; ++i) {

      const float dense_ij = dense(i,j);
      sum_dense += dense_ij;

      const float thresh = relthresh * min(row_max(i), col_max(j));
      if (dense_ij > thresh) {

        sparse.insert(i,j) = dense_ij;
        sum_sparse += dense_ij;
      }
    }
  }

  sparse.finalize();

  float density = sparse.nonZeros() / float(I * J);
  float loss = (sum_dense - sum_sparse) / sum_dense;
  LOG("sparsifying to density " << density
      << " loses " << (100 * loss) << "% of mass");
}

float max_entries_heuristic (MatrixXf & dense)
{
  int N = dense.rows() + dense.cols();
  return N * log2f(N);
}

void sparsify_hard_relative_to_row_col_max (
    const MatrixXf & dense,
    MatrixSf & sparse,
    float relthresh,
    int max_entries,
    bool ignore_diagonal)
{
  do {
    sparsify_soft_relative_to_row_col_max(
        dense,
        sparse,
        relthresh,
        ignore_diagonal);
    relthresh *= 1.5f;
  } while (sparse.nonZeros() > max_entries);
}

//----( joint probabilities )-------------------------------------------------

double likelihood_entropy (const VectorXf & likes)
{
  double sum_l = 0;
  double sum_l_log_l = 0;

  for (size_t i = 0, I = likes.size(); i < I; ++i) {

    float li = likes[i];
    if (li > 0) {

      sum_l += li;
      sum_l_log_l += li * log(li);
    }
  }

  return log(sum_l) - sum_l_log_l / sum_l;
}

double likelihood_entropy (const VectorSf & likes)
{
  double sum_l = 0;
  double sum_l_log_l = 0;

  for (VectorSf::InnerIterator iter(likes); iter; ++iter) {

    float li = iter.value();
    if (li > 0) {

      sum_l += li;
      sum_l_log_l += li * log(li);
    }
  }

  return log(sum_l) - sum_l_log_l / sum_l;
}

double likelihood_entropy_rate (const MatrixSf & joint_likes)
{
  double sum_L = 0;
  double sum_L_H = 0;

  for (int i = 0; i < joint_likes.outerSize(); ++i) {

    double sum_l = 0;
    double sum_l_log_l = 0;

    for (MatrixSf::InnerIterator iter(joint_likes,i); iter; ++iter) {

      float li = iter.value();
      if (li > 0) {

        sum_l += li;
        sum_l_log_l += li * log(li);
      }
    }

    sum_L += sum_l;
    sum_L_H += sum_l * log(sum_l) - sum_l_log_l;
  }

  return sum_L_H / sum_L;
}

double likelihood_mutual_info (const MatrixSf & joint_likes)
{
  double sum_lij = 0;
  double sum_lij_log_lij = 0;

  double sum_li = 0;
  double sum_li_log_li = 0;

  Vector<double> likes_j(joint_likes.innerSize());
  likes_j.set(0);

  for (int i = 0; i < joint_likes.outerSize(); ++i) {

    double li = 0;

    for (MatrixSf::InnerIterator iter(joint_likes,i); iter; ++iter) {

      float lij = iter.value();
      if (lij > 0) {

        sum_lij += lij;
        sum_lij_log_lij += lij * log(lij);

        li += lij;
        likes_j[iter.index()] += lij;
      }
    }

    if (li > 0) {
      sum_li += li;
      sum_li_log_li += li * log(li);
    }
  }

  double sum_lj = 0;
  double sum_lj_log_lj = 0;

  for (size_t j = 0, J = likes_j.size; j < J; ++j) {

    double lj = likes_j[j];
    if (lj > 0) {

      sum_lj += lj;
      sum_lj_log_lj += lj * log(lj);
    }
  }

  double hij = log(sum_lij) - sum_lij_log_lij / sum_lij;
  double hi = log(sum_li) - sum_li_log_li / sum_li;
  double hj = log(sum_lj) - sum_lj_log_lj / sum_lj;

  return hi + hj - hij;
}

void constrain_marginals_bp (
    MatrixXf & joint,
    const VectorXf & prior_dom,
    const VectorXf & prior_cod,
    VectorXf & temp_dom,
    VectorXf & temp_cod,
    float tol,
    size_t max_steps,
    bool logging)
{
  // Enforce simultaineous constraints on a joint PMF
  //
  //   /\x. sum y. J(y,x) = p(x)
  //   /\y. sum x. J(y,x) = q(y)

  ASSERT_EQ(prior_dom.size(), joint.cols());
  ASSERT_EQ(prior_cod.size(), joint.rows());
  ASSERT_LT(0, prior_dom.minCoeff());
  ASSERT_LT(0, prior_cod.minCoeff());

  if (logging) LOG("  constraining marginals via full BP");

  const size_t X = joint.cols();
  const size_t Y = joint.rows();

  const Vector<float> p = as_vector(prior_dom);
  const Vector<float> q = as_vector(prior_cod);

  Vector<float> J = as_vector(joint);
  Vector<float> sum_y_J = as_vector(temp_dom);
  Vector<float> sum_x_J = as_vector(temp_cod);

  float stepsize = 0;
  size_t steps = 0;
  while (steps < max_steps) {
    ++steps;
    if (logging) cout << "   step " << steps << "/" << max_steps << flush;

    stepsize = 0;

    // constrain sum y. J(y,x) = 1 first,
    // in case joint is initalized with conditional

    for (size_t x = 0; x < X; ++x) {
      Vector<float> J_x = J.block(Y, x);
      sum_y_J[x] = sum(J_x);
    }
    ASSERT_LT(0, min(sum_y_J)); // XXX error here
    imax(stepsize, sqrtf(max_dist_squared(sum_y_J, p)));
    idiv_store_rhs(p, sum_y_J);
    for (size_t x = 0; x < X; ++x) {
      Vector<float> J_x = J.block(Y, x);
      J_x *= sum_y_J[x];
    }

    sum_x_J.zero();
    for (size_t x = 0; x < X; ++x) {
      Vector<float> J_x = J.block(Y, x);
      sum_x_J += J_x;
    }
    ASSERT_LT(0, min(sum_x_J));
    imax(stepsize, sqrtf(max_dist_squared(sum_x_J, q)));
    idiv_store_rhs(q, sum_x_J);
    for (size_t x = 0; x < X; ++x) {
      Vector<float> J_x = J.block(Y, x);
      J_x *= sum_x_J;
    }

    if (logging) LOG(", stepsize = " << stepsize);
    if (stepsize < tol) break;
  }
}

//----( distances )-----------------------------------------------------------

float squared_distance (const Point & x, const Point & y)
{
  ASSERT_EQ(x.size, y.size);

  typedef int Accum;

  const uint8_t * restrict x_data = x.data;
  const uint8_t * restrict y_data = y.data;

  Accum result = 0;
  for (size_t i = 0, I = x.size; i < I; ++i) {
    result += sqr(Accum(x_data[i]) - Accum(y_data[i]));
  }

  return result;
}

} // namespace Cloud

