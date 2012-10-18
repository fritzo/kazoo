
#include "cloud_points.h"
#include "cloud_video.h"
#include "cloud_flow.h"
#include "cloud_kernels.h"
#include "histogram.h"
#include "config.h"
#include <climits>
#include <cstring>
#include <algorithm>
#include <iomanip>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#define LOG1(message)

#define TOK(expected_tok) \
  file >> tok; \
  ASSERT(not file.fail(), \
      "failed to read expected token '" expected_tok "'"); \
  ASSERT_EQ(tok, expected_tok);

#define PARSE(type, variable) \
  type variable; \
  file >> variable; \
  ASSERT(not file.fail(), \
      "failed to parse variable '" #variable "' of type " #type);

namespace Cloud
{

//----( abstract point sets )-------------------------------------------------

PointSet::PointSet (
    size_t d,
    size_t c,
    Rectangle r,
    const char * config_filename)

  : m_config(* new ConfigParser("config/default.cloud.conf")),

    dim(d),
    size(c),
    shape(r),

    m_fit_points_to_recon(m_config("fit_points_to_recon", false)),
    m_fit_radius_to_recon(m_config("fit_radius_to_recon", true)),
    m_target_radius(m_config("target_radius", 0)),
    m_target_dof(m_config("target_dof", 0.0f)),
    m_target_entropy(logf(size) * m_config("target_entropy_factor", 0.4f)),

    m_batch_size(bound_to(1, int(max_batch_size),
                          m_config("batch_size", max_batch_size))),
    m_fit_rate_tol(m_config("fit_rate_tol", 0.0f)),
    m_construct_deriv_tol(m_config("construct_deriv_tol", 0.0f)),
    m_construct_tol(m_config("construct_tol", 0.0f)),
    m_purturb_factor(m_config("purturb_factor", 1.0f)),

    m_radius(NAN),
    m_fit_rate(1.0f),

    m_prior(size),

    m_temp_squared_distances(size),
    m_temp_recon(dim),

    m_construct_one_stats_total(0),
    m_construct_one_stats_count(0)
{
  ASSERT_DIVIDES(16, dim); // stride 16 ensures points are aligned for sse ops

  ASSERT_LT(0, m_target_entropy);
  ASSERT_LT(m_target_entropy, logf(size));

  ASSERT_LT(0, m_purturb_factor);
  ASSERT_LE(m_purturb_factor, 1);
}

PointSet::~PointSet ()
{
  if (m_construct_one_stats_total) {
    float mean_terms_in_construct = float(m_construct_one_stats_total)
                                  / float(m_construct_one_stats_count);
    PRINT(mean_terms_in_construct);
  }

  delete & m_config;
}

void PointSet::load_sequence (const VideoSequence & video_seq)
{
  PointSequence seq(video_seq, dim);
  ASSERT_LE(size, seq.size);

  LOG("loading " << size << " of " << seq.size << " points");

  size_t p = 0;
  for (PointSequence::Iterator i(seq); p < size; i.next(), ++p) {
    Point point(dim, i.data());
    set_point(p, point);
  }
}

void PointSet::init_radius ()
{
  ASSERT_LE(2, size);

  LOG("Initializing radius from " << size << " nearest neighbors");

  Vector<float> & sd = m_temp_squared_distances;
  Point point(dim);

  double total_r2 = 0;
  for (size_t p = 0; p < size; ++p) {

    get_point(p, point);
    measure(point);

    std::partial_sort(sd.begin(), sd.begin() + 2, sd.end());
    ASSERT_EQ(sd[0], 0);
    total_r2 += sd[1];
  }

  float rms_radius = sqrt(total_r2 / size);
  PRINT(rms_radius);
  set_radius(rms_radius);
}

void PointSet::fit_radius (
    const VideoSequence & video_seq,
    float tol,
    size_t max_iters)
{
  PointSequence seq(video_seq, dim);

  LOG("\nEstimating radius from "
      << seq.size << " observations x "
      << max_iters << " iterations");

  if (m_target_radius > 0) {
    LOG(" fixing target radius = " << m_target_radius);
  } else if (m_target_dof > 0) {
    LOG(" fitting radius to target dof = " << m_target_dof);
  } else {
    LOG(" fitting radius to target entropy = " << m_target_entropy
        << " (perplexity = " << expf(m_target_entropy) << ")");
  }

  LOG("-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------");
  LOG( std::setw(13) << "Radius"
    << std::setw(13) << "ObsError"
    << std::setw(13) << "ReconError"
    << std::setw(13) << "ObsEntropy"
    << std::setw(13) << "Density"
    << std::setw(13) << "Max/Min");
  LOG( std::setw(13) << get_radius()
    << std::setw(13) << "-"
    << std::setw(13) << "-"
    << std::setw(13) << "-"
    << std::setw(13) << "-");

  for (size_t iter = 0; iter < max_iters; ++iter) {

    init_prior_accum();
    init_stats_accum();

    for (PointSequence::BatchIterator i(seq, m_batch_size); i; i.next()) {
      accum_stats(i.points(), i.buffer_size());
    }

    update_prior_accum();
    update_stats_accum();

    float obs_error = m_quantize_stats_accum.get_rms_error() * get_radius();
    float recon_error = m_construct_stats_accum.get_rms_error();
    ASSERT_FINITE(obs_error);
    ASSERT_FINITE(recon_error);

    float stepsize = fit_radius();

    LOG( std::setw(13) << get_radius()
      << std::setw(13) << obs_error
      << std::setw(13) << recon_error
      << std::setw(13) << (m_quantize_stats_accum.mean_entropy() / logf(2))
      << std::setw(13) << density(m_prior)
      << std::setw(13) << (max(m_prior) / min(m_prior)));

    if (fabs(stepsize) < tol) break;
  }
  LOG("-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------");
}

void PointSet::init_stats_accum ()
{
  m_count_stats_accum = 0;
  m_quantize_stats_accum.zero();
  m_construct_stats_accum.zero();
}

void PointSet::update_stats_accum ()
{
  ASSERT_LT(0, m_count_stats_accum);
  m_quantize_stats_accum /= m_count_stats_accum;
  m_construct_stats_accum /= m_count_stats_accum;
}

void PointSet::init_from_sequence (const VideoSequence & video_seq)
{
  PointSequence seq(video_seq,dim);

  LOG("Randomly adding " << size << " of " << seq.size << " points");

  size_t num_added = 0;
  for (PointSequence::Iterator i(seq); i; i.next()) {

    Point point(dim, i.data());

    if (num_added < size) {

      // fill linearly
      set_point(num_added, point);

    } else {

      // replace randomly
      if (random_bernoulli(float(size) / float(1 + num_added)))
      set_point(random_choice(size), point);
    }

    ++num_added;
  }

  if (m_target_radius > 0) {
    set_radius(m_target_radius);
  } else {
    init_radius();
  }

  fit_radius(video_seq);
}

void PointSet::init_from_smaller (const PointSet & smaller)
{
  ASSERT_LE(smaller.size, size);
  ASSERTW_DIVIDES(smaller.size, size);

  const size_t P = size;
  const size_t Q = smaller.size;

  Point point(dim);

  for (size_t p = 0; p < P; ++p) {
    size_t q = p * Q / P;

    smaller.get_point(q, point);
    set_point(p, point);

    m_prior[p] = smaller.m_prior[q];
  }

  set_radius(smaller.get_radius());

  purturb_points(P / Q, m_purturb_factor);

  update_fit_rates();
}

void PointSet::fit_sequence (VideoSequence & video_seq, size_t num_passes)
{
  ASSERT_LT(0, num_passes);

  PointSequence seq(video_seq, dim);
  ASSERT_LE(size, seq.size);
  ASSERTW_LE(size, video_seq.size());

  LOG("\nFitting " << size << " points to " << seq.size
      << (fitting_points_to_recon() ? " reconstructed" : "")
      << " observations x " << num_passes << " passes...");

  if (m_target_radius > 0) {
    LOG(" fixing target radius = " << m_target_radius);
  } else if (m_target_dof > 0) {
    LOG(" fitting radius to target dof = " << m_target_dof);
  } else if (fitting_radius_to_recon()) {
    LOG(" fitting radius to minimize reconstruction error");
  } else {
    LOG(" fitting radius to target entropy = " << m_target_entropy
        << " (perplexity = " << expf(m_target_entropy) << ")");
  }

  if (fitting_points_to_recon()) {
    LOG(" fitting points to minimize reconstruction error");
  } else {
    LOG(" fitting points to minimimize observation error");
  }

  float optimal_fit_rate = float(size) / seq.size;
  float safe_fit_rate = 1.0f / m_batch_size;
  ASSERTW_LE(optimal_fit_rate, safe_fit_rate);
  set_fit_rate(min(optimal_fit_rate, safe_fit_rate));
  LOG(" point fit rate = " << get_fit_rate());

  Timer timer;

  LOG("-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------");
  LOG( std::setw(13) << "Radius"
    << std::setw(13) << "ObsError"
    << std::setw(13) << "ReconError"
    << std::setw(13) << "ObsEntropy"
    << std::setw(13) << "Density"
    << std::setw(13) << "Max/Min");
  LOG( std::setw(13) << get_radius()
    << std::setw(13) << "-"
    << std::setw(13) << "-"
    << std::setw(13) << "-"
    << std::setw(13) << density(m_prior)
    << std::setw(13) << (max(m_prior) / min(m_prior)));

  bool already_warned_about_max_over_min = false;

  for (size_t pass = 0; pass < num_passes; ++pass) {

    video_seq.shuffle();

    init_prior_accum();
    init_stats_accum();

    for (PointSequence::BatchIterator i(seq, m_batch_size); i; i.next()) {
      fit_points(i.points(), i.buffer_size());
    }

    update_prior_accum();
    update_stats_accum();

    float obs_error = m_quantize_stats_accum.get_rms_error() * get_radius();
    float recon_error = m_construct_stats_accum.get_rms_error();
    float max_over_min = max(m_prior) / min(m_prior);
    ASSERT_FINITE(obs_error);
    ASSERT_FINITE(recon_error);
    if (not already_warned_about_max_over_min) {
      ASSERTW_LT(max_over_min, 100);
      if (not (max_over_min < 100)) already_warned_about_max_over_min = true;
    }
    ASSERT_LT(max_over_min, 1000);

    fit_radius();

    LOG( std::setw(13) << get_radius()
      << std::setw(13) << obs_error
      << std::setw(13) << recon_error
      << std::setw(13) << (m_quantize_stats_accum.mean_entropy() / logf(2))
      << std::setw(13) << density(m_prior)
      << std::setw(13) << max_over_min);
  }
  LOG("-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------"
      "-------------");

  float rate = seq.size * num_passes / timer.elapsed();
  LOG("...fitted at speed " << rate << " probes/sec with "
      << size << " points");
}

Histogram * PointSet::get_histogram (
    const VideoSequence & video_seq,
    float bins_per_unit,
    float tol,
    size_t max_bins)
{
  PointSequence seq(video_seq, dim);
  LOG("building histogram of " << seq.size << " x " << size
      << " probe-point distances");

  Histogram * hist = new Histogram(1 / bins_per_unit);

  Vector<float> likes(size);
  Vector<float> & sd = m_temp_squared_distances;

  const float unit_scale = 1 / sqr(get_radius());

  for (PointSequence::Iterator i(seq); i; i.next()) {
    Point probe(dim, i.data());

    measure(probe);

    Cpu::quantize_one(get_radius(), sd, likes);

    for (size_t i = 0, I = size; i < I; ++i) {

      float like = likes[i];
      if (like > tol) {

        hist->add(unit_scale * sd[i], like);
      }
    }

    ASSERT_LE(hist->size(), max_bins);
  }

  hist->normalize();

  return hist;
}

void PointSet::set_radius (float radius)
{
  ASSERT_LT(0, radius);
  ASSERT_LE(radius, 255 * dim / sqrtf(2));

  m_radius = radius;
}

float PointSet::fit_radius ()
{
  float log_step;

  if (m_target_radius > 0) {

    return 0;

  } else if (m_target_dof > 0) {

    float obs_error = m_quantize_stats_accum.get_rms_error() * get_radius();
    float estimated_radius = obs_error / m_target_dof;
    log_step = logf(estimated_radius / get_radius());

  } else if (fitting_radius_to_recon()) {

    log_step = m_construct_stats_accum.get_log_radius_step(get_radius());

  } else {

    // this is a Newton step in solving entropy(radius) = target_entropy

    float target_entropy = m_target_entropy;
    float current_entropy = m_quantize_stats_accum.mean_entropy();
    log_step = (target_entropy - current_entropy)
             / (2 * m_quantize_stats_accum.var_energy());
  }

  float radius_factor = exp(bound_to(-0.5f, 0.5f, log_step));
  set_radius(radius_factor * get_radius());

  float stepsize = fabsf(logf(radius_factor));
  return stepsize;
}

void PointSet::set_fit_rate (float rate)
{
  ASSERT_LE(0, rate);
  ASSERT_LE(rate, 1);
  ASSERTW_LT(0, rate);

  m_fit_rate = rate;

  update_fit_rates();
}

void PointSet::get_prior (VectorXf & prior) const
{
  Vector<float> prior_vect = as_vector(prior);
  prior_vect = m_prior;
}

void PointSet::quantize (const Point & probe, VectorXf & likes) const
{
  Vector<float> likes_vect = as_vector(likes);
  quantize(probe, likes_vect);
}

void PointSet::construct (const VectorXf & likes, Point & point) const
{
  Vector<float> likes_vect = as_vector(likes);
  construct(likes_vect, point);
}

void PointSet::write (ostream & o) const
{
  string aviname = Persistent::filestem + string(".avi");

  LOG(" writing PointSet");

  o << "\n""PointSet"
    << "\n dim = " << dim
    << "\n size = " << size
    << "\n radius = " << get_radius()
    << "\n points = " << aviname
    << "\n prior = ";

  const size_t line_size = 8;
  for (size_t p = 0, P = size; p < P; ++p) {
    if (p % line_size == 0) o << "\n ";
    o << " " << m_prior[p];
  }

  Streaming::VideoEncoder video(aviname, shape);

  switch (detect_video_format(shape, dim)) {

    case YUV_SINGLE: {

      Streaming::Gloves8Image image(shape.size());
      for (size_t p = 0, P = size; p < P; ++p) {
        get_point(p, image);
        video.push(image);
      }

    } break;

    case MONO_SINGLE: {

      Streaming::Mono8Image image(shape.size());
      for (size_t p = 0, P = size; p < P; ++p) {
        get_point(p, image);
        video.push(image);
      }

    } break;

    case MONO_BATCH: {

      const size_t points_per_frame = shape.width();
      const size_t num_frames = (size + points_per_frame - 1)
                              / points_per_frame;

      Streaming::Mono8Image image(shape.size());
      size_t p = 0;
      for (size_t f = 0; f < num_frames; ++f) {
        for (size_t l = 0; l < points_per_frame; ++l) {

          Point line = image.block(dim, l);

          if (p < size) get_point(p, line); else line.zero();

          ++p;
        }
        video.push(image);
      }

    } break;
  }
}

PointSet * PointSet::create (istream & file)
{
  LOG(" reading PointSet");

  string tok;

  TOK("PointSet")
  TOK("dim") TOK("=") PARSE(size_t, dim)
  TOK("size") TOK("=") PARSE(size_t, size)
  TOK("radius") TOK("=") PARSE(float, radius)
  TOK("points") TOK("=") PARSE(string, cloud_avi)
  TOK("prior") TOK("=")

  Vector<float> prior(size);
  for (size_t p = 0; p < size; ++p) {
    PARSE(float, prior_p)
    prior[p] = prior_p;
  }
  float sum_prior = sum(prior);
  prior *= size / sum_prior;

  VideoSequence seq(cloud_avi);
  ASSERT_LE(size, PointSequence(seq, dim).size);

  PointSet * points = PointSet::create(dim, size, seq.shape());

  points->load_sequence(seq);
  points->set_radius(radius);
  points->set_prior(prior);

  return points;
}

//----( joint priors )--------------------------------------------------------

JointPrior::JointPrior (const PointSet & p)
  : dom(p),
    cod(p),
    joint(* new MatrixSf(int(cod.size), int(dom.size))),

    m_dom_scale(* new VectorXf(int(dom.size))),
    m_cod_scale(* new VectorXf(int(cod.size))),
    m_dom_temp(* new VectorXf(int(dom.size))),
    m_cod_temp(* new VectorXf(int(cod.size)))
{
  LOG("Building JointPrior on " << dom.size << " points");

  update_priors();
}

JointPrior::JointPrior (const PointSet & d, const PointSet & c)
  : dom(d),
    cod(c),
    joint(* new MatrixSf(int(cod.size), int(dom.size))),

    m_dom_scale(* new VectorXf(int(dom.size))),
    m_cod_scale(* new VectorXf(int(cod.size))),
    m_dom_temp(* new VectorXf(int(dom.size))),
    m_cod_temp(* new VectorXf(int(cod.size)))
{
  LOG("Building JointPrior : " << dom.size << " -> " << cod.size << " points");

  update_priors();
}

bool JointPrior::empty () const
{
  return joint.nonZeros() == 0;
}

void JointPrior::clear ()
{
  joint.resize(int(cod.size), int(dom.size));
}

JointPrior::~JointPrior ()
{
  delete & joint;
  delete & m_dom_scale;
  delete & m_cod_scale;
  delete & m_dom_temp;
  delete & m_cod_temp;
}

void JointPrior::update_priors ()
{
  dom.get_prior(m_dom_scale);
  Vector<float> dom_scale = as_vector(m_dom_scale);
  idiv_store_rhs(float(dom.size), dom_scale);

  cod.get_prior(m_cod_scale);
  Vector<float> cod_scale = as_vector(m_cod_scale);
  idiv_store_rhs(float(cod.size), cod_scale);
}

void JointPrior::fit_sequence (VideoSequence & seq, float tol)
{
  ASSERT_EQ(& dom, & cod);
  const PointSet & points = dom;

  MatrixXf dense(int(points.size), int(points.size));

  estimate_flow(points, dense, seq, tol);

  size_t max_entries = max_entries_heuristic(dense);
  bool ignore_diagonal = true;
  sparsify_hard_relative_to_row_col_max(
      dense,
      joint,
      tol,
      max_entries,
      ignore_diagonal);

  LOG("flow entropy rate = " << likelihood_entropy_rate(joint));
  LOG("flow mutual info = " << likelihood_mutual_info(joint));

  update_priors();
}

void JointPrior::init_spline (float tol)
{
  ASSERT_EQ(dom.dim, cod.dim);
  ASSERT(dom.size >= cod.size, "spline must be used : large -> small");

  LOG("Creating cloud spline : " << dom.size << " -> " << cod.size);

  VectorXf dom_prior(int(dom.size));
  VectorXf cod_prior(int(cod.size));
  dom.get_prior(dom_prior);
  cod.get_prior(cod_prior);
  cod_prior *= dom_prior.sum() / cod_prior.sum();

  MatrixXf dense(int(cod.size), int(dom.size));
  Point point(dom.dim);

  // TODO switch to quantizing in batch
  for (size_t x = 0; x < dom.size; ++x) {

    dom.get_point(x, point);

    // WARNING this assumes column-major orientation
    Vector<float> col(cod.size, & dense.coeffRef(0,x));

    cod.quantize(point, col);
    col *= dom_prior(x);
  }

  constrain_marginals_bp(
      dense,
      dom_prior,
      cod_prior,
      m_dom_temp,
      m_cod_temp,
      tol);

  size_t max_entries = max_entries_heuristic(dense);
  sparsify_hard_relative_to_row_col_max(dense, joint, tol, max_entries);

  update_priors();
}

void JointPrior::get_push_forward (MatrixSf & transform)
{
  ASSERT_EQ(transform.rows(), joint.rows());
  ASSERT_EQ(transform.cols(), joint.cols());

  transform = joint;

  for (int i = 0; i < transform.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(transform,i); iter; ++iter) {
      iter.valueRef() *= m_dom_scale(iter.col());
    }
  }
}

void JointPrior::get_pull_back (MatrixSf & transform)
{
  ASSERT_EQ(transform.rows(), joint.cols());
  ASSERT_EQ(transform.cols(), joint.rows());

  transform = joint.transpose();

  for (int i = 0; i < transform.outerSize(); ++i) {
    for (MatrixSf::InnerIterator iter(transform,i); iter; ++iter) {
      iter.valueRef() *= m_dom_scale(iter.row());
    }
  }
}

void JointPrior::push_forward (
    const VectorXf & dom_likes,
    VectorXf & cod_likes) const
{
  m_dom_temp = m_dom_scale.cwiseProduct(dom_likes);
  cod_likes = joint * m_dom_temp;
}

void JointPrior::push_back (
    const VectorXf & cod_likes,
    VectorXf & dom_likes) const
{
  m_cod_temp = m_cod_scale.cwiseProduct(cod_likes);
  dom_likes = joint.transpose() * m_cod_temp;
}

void JointPrior::pull_forward (
    const VectorXf & dom_deriv,
    VectorXf & cod_deriv) const
{
  m_cod_temp = joint * dom_deriv;
  cod_deriv = m_cod_scale.cwiseProduct(m_cod_temp);
}

void JointPrior::pull_back (
    const VectorXf & cod_deriv,
    VectorXf & dom_deriv) const
{
  m_dom_temp = joint.transpose() * cod_deriv;
  dom_deriv = m_dom_scale.cwiseProduct(m_dom_temp);
}

void JointPrior::write (ostream & o) const
{
  LOG(" writing JointPrior");

  o << "\n""JointPrior";
  o << "\n joint = "; write_matrix(joint, o);
}

void JointPrior::read (istream & file)
{
  LOG(" reading JointPrior");

  string tok;

  TOK("JointPrior")
  TOK("joint") TOK("=")
  read_matrix(joint, file);
  ASSERT_EQ(joint.rows(), int(cod.size));
  ASSERT_EQ(joint.cols(), int(dom.size));

  update_priors();
}

} // namespace Cloud

