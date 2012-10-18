
#include "clouds.h"
#include "cloud_map.h"
#include "histogram.h"
#include "config.h"
#include <fstream>

#define EIGEN_YES_I_KNOW_SPARSE_MODULE_IS_NOT_STABLE_YET
#include <Eigen/Eigen>
#include <Eigen/Sparse>

#define LOG1(message)

namespace Cloud
{

//----( persistence )---------------------------------------------------------

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
  // DEBUG("read " #type " " #variable " = " << variable);

//----( clouds )--------------------------------------------------------------

Cloud::Cloud (size_t dim, size_t bits, Rectangle shape)
  : m_points(1),
    m_flow(1)
{
  ASSERT_LT(0, bits);

  LOG("Building Cloud with a single " << bits << "-bit grid");

  PointSet * points = PointSet::create(dim, 1 << bits, shape);
  m_points[0] = points;
  m_flow[0] = new JointPrior(* points);
}

Cloud::Cloud (size_t dim, size_t min_bits, size_t max_bits, Rectangle shape)
{
  ASSERT_LT(0, min_bits);
  ASSERT_LE(min_bits, max_bits);

  LOG("Building Cloud with " << num_grids() << " grids");

  for (size_t i = 0, bits = min_bits; bits <= max_bits; ++bits, ++i) {

    PointSet * points = PointSet::create(dim, 1 << bits, shape);
    m_points.push_back(points);
    m_flow.push_back(new JointPrior(* points));
  }
}

Cloud::Cloud (string filename) { load(filename); }
Cloud::Cloud (istream & file) { read(file); }

Cloud::~Cloud ()
{
  delete_all(m_flow.begin(), m_flow.end());
  delete_all(m_points.begin(), m_points.end());
}

void Cloud::init_points (VideoSequence & seq, size_t fit_passes)
{
  PointSet & smallest = points(0);
  smallest.init_from_sequence(seq);
  smallest.fit_sequence(seq, fit_passes);

  for (size_t i = 1, I = num_grids(); i < I; ++i) {

    const PointSet & smaller = points(i-1);
    PointSet & larger = points(i);

    larger.init_from_smaller(smaller);
    larger.fit_sequence(seq, fit_passes);
  }
}

void Cloud::grow_points (VideoSequence & seq, size_t fit_passes)
{
  const PointSet & smaller = points();

  PointSet & larger = * PointSet::create(
      smaller.dim,
      smaller.size * 2,
      smaller.shape);

  larger.init_from_smaller(smaller);
  larger.fit_sequence(seq, fit_passes);

  m_points.push_back(& larger);
  m_flow.push_back(new JointPrior(larger));
}

void Cloud::fit_points (VideoSequence & seq, size_t fit_passes)
{
  LOG("Fitting points in each of " << num_grids() << " grids");

  for (auto i = m_flow.begin(); i != m_flow.end(); ++i) {
    (*i)->clear();
  }

  for (auto i = m_points.begin(); i != m_points.end(); ++i) {

    LOG("\nFitting " << (*i)->size << " points");

    (*i)->fit_sequence(seq, fit_passes);
  }
}

void Cloud::init_flow (VideoSequence & seq, float tol)
{
  LOG("Estimating flow in each of " << num_grids() << " grids");

  for (auto i = m_flow.begin(); i != m_flow.end(); ++i) {

    LOG("\nEstimating flow among  " << (*i)->dom.size << " points");

    (*i)->fit_sequence(seq, tol);
  }
}

void Cloud::save_priors (string filename, int transition_order)
{
  LOG("Saving " << num_grids() << " priors to " << filename);
  std::ofstream file(filename);

  file << "dict(\n";

  // save prior 

  LOG(" computing prior and " << (1+transition_order)
      << " staying probabilities");

  VectorXf prior(points().size);
  points().get_prior(prior);
  normalize_l1(prior);

  file << "prior = ";
  write_to_python(prior, file);
  file << ",\n";

  // save probability of transition-to-self

  LOG(" computing T^1(x|x)");

  MatrixXf trans = flow().joint;
  normalize_l1(trans);
  VectorXf diag = trans.diagonal();
  prior = trans.rowwise().sum();
  diag.array() /= prior.array();

  file << "stay0 = ";
  write_to_python(diag, file);
  file << ",\n";

  // save probability of transition-to-self over 2^n steps
  for (int n = 1; n <= transition_order; ++n) {

    LOG(" computing T^" << (1 << n) << "(x|x)");

    trans = trans * trans; // this could be GPU optimized
    normalize_l1(trans);
    diag = trans.diagonal();
    prior = trans.rowwise().sum();
    diag.array() /= prior.array();

    file << "stay" << n << " = ";
    write_to_python(diag, file);
    file << ",\n";
  }

  file << ")\n";
}

void Cloud::save_histograms (
    string filename,
    const VideoSequence & seq,
    float bins_per_unit)
{
  LOG("saving " << num_grids() << " histograms to " << filename);
  std::ofstream file(filename);

  file << "[\n";
  for (size_t i = 0; i < num_grids(); ++i) {
    PointSet & p = points(i);

    Histogram * hist = p.get_histogram(seq, bins_per_unit);

    LOG(" writing " << hist->size() << "-bin histogram");
    file << "("
      << p.size << ",\n"
      << p.get_radius() << ",\n"
      << (* hist) << "),\n";

    delete hist;
  }
  file << "]\n";
}

Cloud * Cloud::crop_below (size_t min_size)
{
  ASSERT_LE(min_size, m_points.back()->size);

  LOG("Cropping point sets smaller than " << min_size);

  std::vector<PointSet *> new_points;
  std::vector<JointPrior *> new_flow;

  for (size_t i = 0; i < num_grids(); ++i) {
    if (m_points[i]->size < min_size) {
      delete m_points[i];
      delete m_flow[i];
    } else {
      new_points.push_back(m_points[i]);
      new_flow.push_back(m_flow[i]);
    }
  }

  std::swap(m_points, new_points);
  std::swap(m_flow, new_flow);

  return this;
}

Cloud * Cloud::crop_above (size_t max_size)
{
  ASSERT_LE(m_points.front()->size, max_size);

  LOG("Cropping point sets larger than " << max_size);

  std::vector<PointSet *> new_points;
  std::vector<JointPrior *> new_flow;

  for (size_t i = 0; i < num_grids(); ++i) {
    if (m_points[i]->size > max_size) {
      delete m_points[i];
      delete m_flow[i];
    } else {
      new_points.push_back(m_points[i]);
      new_flow.push_back(m_flow[i]);
    }
  }

  std::swap(m_points, new_points);
  std::swap(m_flow, new_flow);

  return this;
}

void Cloud::write (ostream & o) const
{
  LOG(" writing Cloud with " << num_grids() << " grids");

  o << "\n"
    << "\n""Cloud = (points flow) x " << num_grids() << " grids";

  for (size_t i = 0, I = num_grids(); i < I; ++i) {

    std::ostringstream stem;
    stem << Persistent::filestem;
    if (I > 1) {
      int num_points = points(i).size;
      int bits = log2i(num_points);
      if (num_points == 1 << bits) {
        stem << "-" << bits << "b";
      } else {
        stem << "-" << num_points;
      }
    }
    points(i).filestem = stem.str();

    o << "\n""grid " << i << " =";
    o << "\n"; points(i).write(o);
    o << "\n"; flow(i).write(o);
  }
}

void Cloud::read (istream & file)
{
  LOG(" reading Cloud");

  ASSERT_EQ(m_points.size(), 0);
  ASSERT_EQ(m_flow.size(), 0);

  string tok;
  TOK("Cloud") TOK("=") TOK("(points") TOK("flow)")
  TOK("x") PARSE(size_t, grids) TOK("grids")

  LOG("  reading " << grids << " grids");

  for (size_t i = 0, I = grids; i < I; ++i) {

    TOK("grid") PARSE(size_t, grid_num);
    ASSERT_EQ(grid_num, i);
    TOK("=")

    PointSet * points = PointSet::create(file);
    JointPrior * flow = new JointPrior(* points);
    flow->read(file);

    m_points.push_back(points);
    m_flow.push_back(flow);
  }
}

//----( controllers )---------------------------------------------------------

Controller::Controller (string dom_filename, string cod_filename)
  : m_config(* new ConfigParser("config/default.cloud.conf")),

    m_coalesce_track(m_config("coalesce_track", true)),
    m_track_timescale(
        m_config("track_timescale_sec", DEFAULT_TRACK_TIMESCALE_SEC)
        * DEFAULT_VIDEO_FRAMERATE),
    m_observation_weight(
        m_config("observation_weight", DEFAULT_OBSERVATION_WEIGHT)),

    m_dom(new Cloud(dom_filename)),
    m_cod((new Cloud(cod_filename))->crop_above(m_dom->points().size)),
    m_map(new JointPrior(dom(), cod())),

    m_observe(* new MatrixSf(int(dom().size), int(cod().size))),
    m_advance(* new MatrixSf(int(dom().size), int(dom().size))),

    m_observe_gpu(NULL),
    m_advance_gpu(NULL),

    m_dom_prior(* new VectorXf(int(dom().size))),
    m_cod_observation(* new VectorXf(int(cod().size))),
    m_dom_observation(* new VectorXf(int(dom().size))),
    m_dom_prediction(* new VectorXf(int(dom().size))),
    m_dom_state(* new VectorXf(int(dom().size)))
{
  init_transforms();
  dom().get_prior(m_dom_prior);
  m_dom_state = m_dom_prior;
}

Controller::Controller (string filename)
  : m_config(* new ConfigParser("config/default.cloud.conf")),

    m_coalesce_track(m_config("coalesce_track", true)),
    m_track_timescale(
        m_config("track_timescale_sec", DEFAULT_TRACK_TIMESCALE_SEC)
        * DEFAULT_VIDEO_FRAMERATE),
    m_observation_weight(
        m_config("observation_weight", DEFAULT_OBSERVATION_WEIGHT)),

    m_dom(NULL),
    m_cod(NULL),
    m_map(NULL),

    m_observe(* new MatrixSf()),
    m_advance(* new MatrixSf()),

    m_observe_gpu(NULL),
    m_advance_gpu(NULL),

    m_dom_prior(* new VectorXf()),
    m_cod_observation(* new VectorXf()),
    m_dom_observation(* new VectorXf()),
    m_dom_prediction(* new VectorXf()),
    m_dom_state(* new VectorXf())
{
  load(filename);

  m_observe.resize(dom().size, cod().size);
  m_advance.resize(dom().size, dom().size);

  m_dom_prior.resize(dom().size);
  m_cod_observation.resize(cod().size);
  m_dom_observation.resize(dom().size);
  m_dom_prediction.resize(dom().size);
  m_dom_state.resize(dom().size);

  init_transforms();
  dom().get_prior(m_dom_prior);
  m_dom_state = m_dom_prior;
}

Controller::~Controller ()
{
  delete & m_config;

  delete m_map;
  delete m_dom;
  delete m_cod;

  delete & m_observe;
  delete & m_advance;

  if (m_observe_gpu) delete m_observe_gpu;
  if (m_advance_gpu) delete m_advance_gpu;

  delete & m_cod_observation;
  delete & m_dom_observation;
  delete & m_dom_prediction;
  delete & m_dom_state;
}

void Controller::init_transforms ()
{
  map().get_pull_back(m_observe);
  dom_flow().get_push_forward(m_advance);

  MatrixSf temp = m_observe.transpose();
  m_observe_gpu = new Gpu::SparseMultiplier(temp);

  temp = m_advance.transpose();
  m_advance_gpu = new Gpu::SparseMultiplier(temp);
}

void Controller::optimize (float tol, size_t max_iters)
{
  const size_t dom_grids = m_dom->num_grids();    ASSERT_LT(0, dom_grids);
  const size_t cod_grids = m_cod->num_grids();    ASSERT_LT(0, cod_grids);

  MatrixSf & sparse = map().joint;

  if (sparse.nonZeros()) {

    LOG("Re-optimizing map using one grid");

    MapOptimizer optimizer(dom_flow(), cod_flow());
    optimizer.init(sparse, tol);
    optimizer.solve(sparse, tol, max_iters);

  } else if (dom_grids == 1 and cod_grids == 1) {

    LOG("Optimizing map using one grid");

    MapOptimizer optimizer(dom_flow(), cod_flow());
    optimizer.init_random(tol);
    optimizer.solve(sparse, tol, max_iters);

  } else {

    LOG("Optimizing map using " << dom_grids << "," << cod_grids << " grids");

    // create smallest map

    LOG("\nMultigrid step " << dom(0).size << " -> " << cod(0).size);

    MatrixXf dense(int(cod(0).size), int(dom(0).size));

    MapOptimizer optimizer(dom_flow(0), cod_flow(0));
    optimizer.init_random(tol);
    optimizer.solve(dense, tol, max_iters);

    // create progressively larger maps

    for (size_t d = 0, c = 0;;) {

      // transform small problem's solution to larger problem

      if (c < d and c + 1 < cod_grids) {

        ++c;
        LOG("\nMultigrid step " << dom(d).size << " -> " << cod(c).size);

        VectorXf cod_prior(int(cod(c-1).size));
        cod(c-1).get_prior(cod_prior);

        JointPrior cod_spline(cod(c), cod(c-1));
        cod_spline.init_spline(tol);

        dense.array() /= (cod_prior * MatrixXf::Ones(1,dense.cols())).array();
        dense = cod_spline.joint.transpose() * dense;

      } else if (d + 1 < dom_grids) {

        ++d;
        LOG("\nMultigrid step " << dom(d).size << " -> " << cod(c).size);

        VectorXf dom_prior(int(dom(d-1).size));
        dom(d-1).get_prior(dom_prior);

        JointPrior dom_spline(dom(d), dom(d-1));
        dom_spline.init_spline(tol);

        dense.array() /= ( MatrixXf::Ones(dense.rows(),1)
                         * dom_prior.transpose() ).array();
        dense = dense * dom_spline.joint;

      } else break;

      // improve solution to large problem

      MapOptimizer optimizer(dom_flow(d), cod_flow(c));
      optimizer.init(dense, tol);
      optimizer.add_noise(tol);
      optimizer.solve(dense, tol, max_iters);
    }

    // sparsify

    size_t max_entries = max_entries_heuristic(dense);
    sparsify_hard_relative_to_row_col_max(dense, sparse, tol, max_entries);
  }
}

void Controller::set_track (const Point & dom_point)
{
  dom().quantize(dom_point, m_dom_state);
}

void Controller::update_track (const Point & cod_point, Point & dom_point)
{
  cod().quantize(cod_point, m_cod_observation);
  m_dom_observation = m_observe * m_cod_observation;

  if (m_track_timescale > 0) {

    m_dom_prediction = m_advance * m_dom_state;

    float reset_rate = 1 - expf(-1.0f / m_track_timescale);
    float reset_like = reset_rate / (1 - reset_rate);
    m_dom_prediction += reset_like * m_dom_prior;

    m_dom_state.array() = ( m_dom_observation.array()
                          / m_dom_prior.array()
                          ).pow(m_observation_weight)
                        * ( m_dom_prediction.array()
                          ).pow(1 - m_observation_weight);

  } else {

    m_dom_state = m_dom_observation;
  }

  normalize_l1(m_dom_state, dom().size);
  dom().construct(m_dom_state, dom_point);
}

void Controller::crop ()
{
  LOG("Cropping map");

  m_dom->crop_below(m_dom->points().size);
  m_cod->crop_below(m_cod->points().size);
  m_cod->flow().clear();
}

void Controller::write (ostream & o) const
{
  LOG(" writing Controller");

  m_dom->filestem = Persistent::filestem + string("-dom");
  m_cod->filestem = Persistent::filestem + string("-cod");

  o << "\n"
    << "\n""Controller = domain codomain map";

  o << "\n"; m_dom->write(o);
  o << "\n"; m_cod->write(o);
  o << "\n"; m_map->write(o);
}

void Controller::read (istream & file)
{
  LOG(" reading Controller");

  ASSERT_NULL(m_dom);
  ASSERT_NULL(m_cod);
  ASSERT_NULL(m_map);

  string tok;
  TOK("Controller") TOK("=") TOK("domain") TOK("codomain") TOK("map")

  m_dom = new Cloud(file);
  m_cod = new Cloud(file);

  m_map = new JointPrior(dom(), cod());
  map().read(file);
}

} // namespace Cloud

