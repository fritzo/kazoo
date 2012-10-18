
#include "cloud_flow.h"
#include "cloud_video.h"
#include "filters.h"
#include <algorithm>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#define LOG1(message)

namespace Cloud
{

class FlowEstimator
{
public:

  const PointSet & points;

  FlowEstimator (const PointSet & points, MatrixXf & flow);
  virtual ~FlowEstimator ();

  void solve (VideoSequence & seq, float tol, size_t max_em_steps = 8);

protected:

  enum { max_state_size = 64 };

  static const size_t s_bp_steps;

  struct Tile
  {
    Vector<int> m_head_idx;
    Vector<int> m_tail_idx;
    Vector<float> m_head;
    Vector<float> m_tail;

    static size_t s_max_steps;
    static size_t s_total_steps;
    static size_t s_total_solves;

  public:

    Tile (const VectorSf & head, const VectorSf & tail);

    void init (MatrixXf & accum);
    void solve (
        const MatrixXf & prior,
        MatrixXf & accum,
        float tol,
        float * work);

    static size_t pop_max_steps ();
    static float pop_mean_steps ();

  private:

    inline void solve_guts (float * restrict flow, float tol);
  };

  MatrixXf & m_flow;
  MatrixXf m_accum;
  std::vector<Tile *> m_tiles;

  void solve (float tol, size_t max_steps);
  void init_tiles ();
  void solve_tiles (float tol);
  void constrain (float tol);
};

void estimate_flow (
    const PointSet & points,
    MatrixXf & flow,
    VideoSequence & seq,
    float tol)
{
  FlowEstimator estimator(points, flow);
  estimator.solve(seq, tol);
}

const size_t FlowEstimator::s_bp_steps = 40;

FlowEstimator::FlowEstimator (const PointSet & p, MatrixXf & flow)
  : points(p),
    m_flow(flow),
    m_accum(flow.rows(), flow.cols())
{
  ASSERT_EQ(flow.cols(), int(points.size));
  ASSERT_EQ(flow.rows(), int(points.size));
}

FlowEstimator::~FlowEstimator ()
{}

void FlowEstimator::solve (
    VideoSequence & video_seq,
    float tol,
    size_t max_steps)
{
  LOG("Estimating pairwise probability density");
  LOG(" tol = " << tol << ", max_steps = " << max_steps);

  video_seq.sort();

  PointSequence seq(video_seq, points.dim);

  const size_t batch_size = PointSet::max_batch_size;
  Vector<float> likes_batch(points.size * batch_size);

  VectorXf dense(points.size);
  Vector<float> dense_vect = as_vector(dense);

  LOG(" building sparsity thresholds from " << seq.size << " observations");
  LOG("  observations per point = " << (float(seq.size) / points.size));

  float min_o_max_p_likes = INFINITY;
  Vector<float> max_o_likes(points.size);
  max_o_likes.set(-INFINITY);

  Filters::DebugStats<double> obs_density;

  for (PointSequence::BatchIterator i(seq, batch_size); i; i.next()) {
    const Point points_i = i.points();
    Vector<float> likes_i(points.size * i.buffer_size(), likes_batch);

    points.quantize_batch(points_i, likes_i);

    for (size_t j = 0; j < i.buffer_size(); ++j) {
      const Vector<float> likes = likes_i.block(points.size, j);

      imax(max_o_likes, likes);
      imin(min_o_max_p_likes, max(likes));

      obs_density.add(density(likes));
    }
  }

  float min_p_max_o_likes = min(max_o_likes);
  ASSERT_LT(0, min_p_max_o_likes);
  LOG("  min p:point. max o:obs. P(p|o) = "
      << (min_p_max_o_likes / points.size));
  ASSERT_LT(0, min_o_max_p_likes);
  LOG("  min o:obs. max p:point. P(p|o) = "
      << (min_o_max_p_likes / points.size));

  LOG("  obs density: " << obs_density);

  LOG(" building transition tiles from " << seq.size << " observations");

  VectorXf prior(points.size);
  prior.setZero();

  Vector<float> thresh(points.size);
  Vector<float> point_thresh(points.size);
  multiply(tol, max_o_likes, point_thresh);
  ASSERT_LT(0, min(point_thresh));

  size_t num_cropped_states = 0;
  Filters::DebugStats<double> state_density;
  Filters::DebugStats<double> state_entropy;
  Filters::DebugStats<double> state_size;
  Filters::DebugStats<double> state_loss;
  Filters::DebugStats<double> tile_area;

  VectorSf state;
  VectorSf prev_state;

  m_tiles.reserve(seq.size);

  for (PointSequence::BatchIterator i(seq, batch_size); i; i.next()) {
    const Point points_i = i.points();
    Vector<float> likes_i(points.size * i.buffer_size(), likes_batch);

    points.quantize_batch(points_i, likes_i);

    for (size_t j = 0; j < i.buffer_size(); ++j) {
      const Vector<float> likes = likes_i.block(points.size, j);
      dense_vect = likes;

      std::swap(prev_state, state);

      float obs_thresh = tol * dense.maxCoeff();
      ASSERT_LT(0, obs_thresh);
      minimum(obs_thresh, point_thresh, thresh);

      float loss = sparsify_absolute(dense, state, thresh);
      if (state.nonZeros() > max_state_size) {
        loss += sparsify_size(state, max_state_size);
        ++num_cropped_states;
      }
      state_loss.add(loss / points.size);
      state_size.add(state.nonZeros());

      normalize_l1(state, state.size());
      ASSERT_LT(0, state.sum());
      prior += state;

      state_density.add(density(state));
      state_entropy.add(likelihood_entropy(state));
      state_size.add(state.nonZeros());

      if (i.has_prev(j)) {

        m_tiles.push_back(new Tile(prev_state, state));

        tile_area.add(prev_state.nonZeros() * state.nonZeros());
      }
    }
  }

  float portion_cropped = float(num_cropped_states) / seq.size;
  LOG("  cropped " << (100 * portion_cropped)
      << "% of states to " << max_state_size);
  LOG("  state size: " << state_size);
  LOG("  state loss: " << state_loss);
  LOG("  state density: " << state_density);
  LOG("  state entropy: " << state_entropy);
  LOG("  tile area: " << tile_area);

  float density = state_size.mean() / points.size;
  float loss = state_loss.mean();
  LOG("  sparsifying states to density " << density
      << " loses " << (100 * loss) << "% of mass");

  normalize_l1(prior, prior.size());
  ASSERT_LT(0, prior.minCoeff()); // XXX error here
  Vector<float> prior_sparse = as_vector(prior);
  const Vector<float> & prior_dense = points.get_prior();
  float relentropy_d_s = relentropy(prior_dense, prior_sparse, true);
  LOG("  relentropy(dense prior, sparse prior) = " << relentropy_d_s);

  solve(tol, max_steps);

  delete_all(m_tiles.begin(), m_tiles.end());
  m_tiles.clear();

  constrain(tol);
}

void FlowEstimator::solve (float tol, size_t max_em_steps)
{
  LOG(" estimating flow via EM + local BP");

  Timer timer;

  init_tiles();

  for (size_t step = 0; step < max_em_steps; ++step) {

    cout << "  step " << (1+step) << "/" << max_em_steps << flush;

    solve_tiles(tol);

    cout << ", tile steps mean = " << Tile::pop_mean_steps()
      << ", max = " << Tile::pop_max_steps() << endl;
  }

  LOG(" EM + local BP took " << timer.elapsed() << " sec");
}

void FlowEstimator::init_tiles ()
{
  const size_t num_tiles = m_tiles.size();
  Tile ** restrict tiles = & m_tiles[0];

  m_flow.setZero();

  #pragma omp parallel for schedule(dynamic)
  for (size_t t = 0; t < num_tiles; ++t) {
    tiles[t]->init(m_flow);
  }
}

void FlowEstimator::solve_tiles (float tol)
{
  const size_t num_tiles = m_tiles.size();
  Tile ** restrict tiles = & m_tiles[0];

  m_accum.setZero();

  float * work;
  #pragma omp parallel private(work)
  {
    work = malloc_float(max_state_size * max_state_size);

    #pragma omp for schedule(dynamic)
    for (size_t t = 0; t < num_tiles; ++t) {
      tiles[t]->solve(m_flow, m_accum, tol, work);
    }

    free_float(work);
  }

  m_flow = m_accum;
}

void FlowEstimator::constrain (float tol)
{
  VectorXf prior(int(points.size));
  VectorXf marginals(int(points.size));

  points.get_prior(prior);

  constrain_marginals_bp(
      m_flow,
      prior,
      prior,
      marginals,
      marginals,
      tol);
}

//----( tiles )----

size_t FlowEstimator::Tile::s_max_steps = 0;
size_t FlowEstimator::Tile::s_total_steps = 0;
size_t FlowEstimator::Tile::s_total_solves = 0;

size_t FlowEstimator::Tile::pop_max_steps ()
{
  size_t max_steps = s_max_steps;

  s_max_steps = 0;

  return max_steps;
}

float FlowEstimator::Tile::pop_mean_steps ()
{
  ASSERT_LT(0, s_total_solves);

  float mean_steps = float(s_total_steps) / float(s_total_solves);

  s_total_steps = 0;
  s_total_solves = 0;

  return mean_steps;
}

FlowEstimator::Tile::Tile (
    const VectorSf & head,
    const VectorSf & tail)
  : m_head_idx(head.nonZeros()),
    m_tail_idx(tail.nonZeros()),
    m_head(head.nonZeros()),
    m_tail(tail.nonZeros())
{
  {
    size_t h = 0;
    for (VectorSf::InnerIterator iter(head); iter; ++iter) {

      m_head_idx[h] = iter.index();
      m_head[h] = iter.value();
      ++h;
    }
  }
  {
    size_t t = 0;
    for (VectorSf::InnerIterator iter(tail); iter; ++iter) {

      m_tail_idx[t] = iter.index();
      m_tail[t] = iter.value();
      ++t;
    }
  }

  // we expect head & tail to be l1-normalized
}

void FlowEstimator::Tile::init (MatrixXf & accum)
{
  const size_t H = m_head.size;
  const size_t T = m_tail.size;

  const float * restrict head = m_head;
  const float * restrict tail = m_tail;

  for (size_t h = 0; h < H; ++h) { int i = m_head_idx[h];
  for (size_t t = 0; t < T; ++t) { int j = m_tail_idx[t];

    #pragma omp atomic
    accum(j,i) += head[h] * tail[t];
  }}
}

void FlowEstimator::Tile::solve (
    const MatrixXf & prior,
    MatrixXf & accum,
    float tol,
    float * work)
{
  const size_t H = m_head.size;
  const size_t T = m_tail.size;

  Vector<float> flow(H * T, work);

  for (size_t h = 0; h < H; ++h) { int i = m_head_idx[h];
  for (size_t t = 0; t < T; ++t) { int j = m_tail_idx[t];
    flow[H * t + h] = prior(j,i);
  }}

  solve_guts(flow, tol);

  for (size_t h = 0; h < H; ++h) { int i = m_head_idx[h];
  for (size_t t = 0; t < T; ++t) { int j = m_tail_idx[t];

    #pragma omp atomic
    accum(j,i) += flow[T * h + t];
  }}
}

inline void FlowEstimator::Tile::solve_guts (float * restrict flow, float tol)
{
  const size_t H = m_head.size;
  const size_t T = m_tail.size;

  const float tol2 = tol / 100;
  const float tol2_over_H = tol2 / H;
  const float tol2_over_T = tol2 / T;

  const float * restrict head = m_head;
  const float * restrict tail = m_tail;

  size_t steps = 0;
  while (steps < s_bp_steps) {
    ++steps;

    float stepsize = 0;

    for (size_t h = 0; h < H; ++h) {

      float flow_h = 0;
      for (size_t t = 0; t < T; ++t) {
        flow_h += flow[T * h + t];
      }
      ASSERT_LT(0, flow_h);

      imax(stepsize, fabsf(head[h] - flow_h));

      const float scale = head[h] / (flow_h + tol2);
      for (size_t t = 0; t < T; ++t) {
        float & restrict flow_ht = flow[T * h + t];
        flow_ht = scale * (flow_ht + tol2_over_T);
      }
    }

    for (size_t t = 0; t < T; ++t) {

      float flow_t = 0;
      for (size_t h = 0; h < H; ++h) {
        flow_t += flow[T * h + t];
      }
      ASSERT_LT(0, flow_t);

      imax(stepsize, fabsf(tail[t] - flow_t));

      const float scale = tail[t] / (flow_t + tol2);
      for (size_t h = 0; h < H; ++h) {
        float & restrict flow_ht = flow[T * h + t];
        flow_ht = scale * (flow_ht + tol2_over_H);
      }
    }

    if (stepsize < tol) break;
  }

  imax(s_max_steps, steps);
  s_total_steps += H * T * steps;
  s_total_solves += H * T;
}

} // namespace Cloud

