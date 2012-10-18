
#include "audio_alignment.h"
#include "config.h"

namespace AudioAlignment
{

//----( alignment model )-----------------------------------------------------

AlignmentModel::AlignmentModel (size_t size, const char * config_filename)
  : m_config(* new ConfigParser(config_filename)),

    feature_size(size),

    feature_sigma(m_config("feature_sigma", 1.0f)),
    feature_dof(m_config("feature_dof", 1.0f)),
    segment_timescale(m_config("segment_timescale", 1.0f)),
    log_pos_sigma(m_config("log_pos_sigma", 1.0f)),
    log_vel_sigma(m_config("log_vel_sigma", 1.0f))
{
  ASSERT_LT(0, feature_sigma);
  ASSERT_LE(1, feature_dof);
  ASSERT_LE(1, segment_timescale);
}

AlignmentModel::~AlignmentModel ()
{
  delete & m_config;
}

//----( feature buffer )------------------------------------------------------

FeatureBuffer::FeatureBuffer (
    const AlignmentModel & model,
    size_t max_duration)
  : m_image(NULL),
    m_feature_size(model.feature_size),
    m_max_duration(max_duration)
{
}

FeatureBuffer::~FeatureBuffer ()
{
  delete_all(m_frames.begin(), m_frames.end());
  if (m_image) delete m_image;
}

void FeatureBuffer::push (Seconds time, const Vector<uint8_t> & features)
{
  ASSERT_EQ(m_feature_size, features.size);

  if (m_max_duration and duration() == m_max_duration) return;

  m_times.push_back(time);
  Vector<uint8_t> * frame = new Vector<uint8_t>(features.size);
  * frame = features;
  m_frames.push_back(frame);
}

void FeatureBuffer::finish ()
{
  if (m_image) return;

  ASSERT_LT(0, m_frames.size());
  ASSERT_EQ(m_image, NULL);

  const size_t I = duration();
  const size_t J = feature_size();

  m_image = new Vector<uint8_t>(I * J);
  for (size_t i = 0; i < I; ++i) {
    Vector<uint8_t> frame = m_image->block(J, i);
    frame = * m_frames[i];
  }

  delete_all(m_frames.begin(), m_frames.end());
  m_frames.clear();
}

//----( cost matrix )---------------------------------------------------------

CostMatrix::CostMatrix (
    const AlignmentModel & model,
    FeatureBuffer & buffer,
    bool symmetric)
  : m_size(buffer.duration()),
    m_cost(sqr(m_size)),
    m_mean_cost(NAN)
{
  LOG("building cost matrix...");

  const size_t duration = m_size;

  const uint8_t * restrict features = buffer.image();
  float * restrict cost = m_cost;

  Timer timer;

  float mean_cost = 0;
  for (size_t i = 0, j = 1; j < duration; ++i, ++j) {
    mean_cost += model.feature_distance(
        features + model.feature_size * i,
        features + model.feature_size * j);
  }
  mean_cost *= 0.5f / (duration - 1);
  PRINT(mean_cost);
  m_mean_cost = mean_cost;

  if (symmetric) {

    // TODO this could be optimized by only computing lower triangle

    #pragma omp parallel for
    for (size_t i = 0; i < duration; ++i) {
      for (size_t j = 0; j < duration; ++j) {
        cost[duration * i + j]
          = model.feature_distance(
            features + model.feature_size * i,
            features + model.feature_size * j);
      }
    }

  } else {

    #pragma omp parallel for
    for (size_t i = 0; i < duration; ++i) {
      for (size_t j = 0; j < duration; ++j) {
        cost[duration * i + j]
          = model.feature_divergence(
            features + model.feature_size * i,
            features + model.feature_size * j);
      }
    }
  }

  LOG("...built cost matrix in " << timer.elapsed() << " sec");
}

//----( alignment matrix )----------------------------------------------------

AlignmentMatrix::AlignmentMatrix (const AlignmentModel & model, size_t size)
  : m_model(model),
    m_size(size),
    m_posterior(sqr(m_size))
{
}

//----( marginals )----

namespace
{
inline float bound (float like)
{
  return min(1e16f, like);
}
inline float bound_imul (float & lhs, float rhs)
{
  return lhs = bound(lhs * rhs);
}
} // anonymous namespace

void AlignmentMatrix::init_marginal (CostMatrix & cost_matrix)
{
  LOG("Computing marginal audio alignment via Smith-Waterman...");
  // TODO also implement a NCV model using model.log_vel_sigma
  Timer timer;

  const size_t I = m_size;

  const float P_break = m_model.get_break_prob();
  const float P_continue = m_model.get_continue_prob();
  const float P_skip = m_model.get_skip_prob();
  PRINT3(P_break, P_continue, P_skip);

  LOG(" converting distances -> likelihood ratios");
  const float cost_shift = 0.5f * m_model.feature_dof;
  float * restrict obs = cost_matrix.cost().data;
  #pragma omp parallel for
  for (size_t ij = 0; ij < I*I; ++ij) {
    obs[ij] = expf(cost_shift - obs[ij]);
  }

  LOG("  max(present) = " << max(cost_matrix.cost()));

  // forward pass:
  //   post <- sum path : past. prod t:path obs(t)

  float * restrict post = m_posterior.data;
  const size_t di = I;
  const size_t dj = 1;

  LOG(" propagating forward");
  {
    { size_t ij = 0;

      post[ij] = bound(P_break);
    }

    for (size_t i = 1; i < I; ++i) {
      size_t ij = I * i;

      post[ij] = bound( P_break
                      + P_skip * post[ij - di] * obs[ij - di]);
    }

    for (size_t j = 1; j < I; ++j) {
      size_t ij = j;

      post[ij] = bound( P_break
                      + P_skip * post[ij - dj] * obs[ij - dj] );
    }

    for (size_t i = 1; i < I; ++i) {
    for (size_t j = 1; j < I; ++j) {
      size_t ij = I * i + j;

      post[ij] = bound( P_break
                      + P_continue * post[ij - di - dj] * obs[ij - di - dj]
                      + P_skip * ( post[ij - di] * obs[ij - di]
                                 + post[ij - dj] * obs[ij - dj] ) );
    }}
  }

  LOG("  max(past) = " << max(m_posterior));

  // backward pass:
  //   obs <- sum path : present+future. prod t:path obs(t)
  //   post <- sum path : past+present+future. prod t:path obs(t)

  LOG(" propagating backward");
  {
    { size_t ij = I * I - 1;

      post[ij] *= obs[ij]
               *= bound(P_break);
    }

    for (size_t i = I - 1; i; --i) {
      size_t ij = I * (i-1);

      bound_imul(post[ij],
      bound_imul(obs[ij],
           + P_break
           + P_skip * obs[ij + di]
           ));
    }

    for (size_t j = I - 1; j; --j) {
      size_t ij = j - 1;

      bound_imul(post[ij],
      bound_imul(obs[ij],
          + P_break
          + P_skip * obs[ij + dj]
          ));
    }

    for (size_t i = I - 1; i; --i) {
    for (size_t j = I - 1; j; --j) {
      size_t ij = I * (i-1) + (j-1);

      bound_imul(post[ij],
      bound_imul(obs[ij],
          + P_break
          + P_continue * obs[ij + di + dj]
          + P_skip * ( obs[ij + di]
                     + obs[ij + dj] )
          ));
    }}
  }

  LOG("  max(present+future) = " << max(cost_matrix.cost()));
  LOG("  max(past+present+future) = " << max(m_posterior));

  // convert likelihood ratios to probabilities

  LOG(" converting likelihood ratios to probabilities");

  for (size_t ij = 0, II = I * I; ij < II; ++ij) {
    float post_ij = post[ij];
    post[ij] = post_ij / (1 + post_ij);
  }

  LOG("...aligned audio in " << timer.elapsed() << " sec");
}

//----( maximum likelihood )----

void AlignmentMatrix::init_maxlike (CostMatrix & cost_matrix)
{
  LOG("Computing maximum likelihood audio alignment via Smith-Waterman...");
  // TODO also implement a NCV model using model.log_vel_sigma
  Timer timer;

  const size_t I = m_size;

  const float E_break = m_model.get_break_energy();
  const float E_continue = m_model.get_continue_energy();
  const float E_skip = m_model.get_skip_energy();
  PRINT3(E_break, E_continue, E_skip);

  LOG(" converting distances -> energies");
  const float dof = m_model.feature_dof;
  float * restrict obs = cost_matrix.cost().data;
  for (size_t ij = 0, IJ = I*I; ij < IJ; ++ij) {
    obs[ij] = 0.5f * (obs[ij] - dof);
  }

  LOG("  present : "
      << min(cost_matrix.cost()) << ", "
      << max(cost_matrix.cost()));

  // forward pass:
  //   post <- min path : past. sum t:path obs(t)

  float * restrict post = m_posterior.data;
  const size_t di = I;
  const size_t dj = 1;

  LOG(" propagating forward");
  {
    { size_t ij = 0;

      post[ij] = E_break;
    }

    for (size_t i = 1; i < I; ++i) {
      size_t ij = I * i;

      post[ij] = min( E_break,
                      E_skip + post[ij - di] + obs[ij - di] );
    }

    for (size_t j = 1; j < I; ++j) {
      size_t ij = j;

      post[ij] = min( E_break,
                      E_skip + post[ij - dj] + obs[ij - dj] );
    }

    for (size_t i = 1; i < I; ++i) {
    for (size_t j = 1; j < I; ++j) {
      size_t ij = I * i + j;

      post[ij] = min( E_break,
                 min( E_continue + post[ij - di - dj] + obs[ij - di - dj],
                      E_skip + min( post[ij - di] + obs[ij - di],
                                    post[ij - dj] + obs[ij - dj] )));
    }}
  }

  LOG("  past : " << min(m_posterior) << ", " << max(m_posterior));

  // backward pass:
  //   obs <- min path : present+future. sum t:path obs(t)
  //   post <- min path : past+present+future. sum t:path obs(t)

  LOG(" propagating backward");
  {
    { size_t ij = I * I - 1;

      post[ij] +=
      obs[ij] += E_break;
    }

    for (size_t i = I - 1; i; --i) {
      size_t ij = I * (i-1);

      post[ij] +=
      obs[ij] += min( E_break,
                      E_skip + obs[ij + di] );
    }

    for (size_t j = I - 1; j; --j) {
      size_t ij = j - 1;

      post[ij] +=
      obs[ij] += min( E_break,
                      E_skip + obs[ij + dj] );
    }

    for (size_t i = I - 1; i; --i) {
    for (size_t j = I - 1; j; --j) {
      size_t ij = I * (i-1) + (j-1);

      post[ij] +=
      obs[ij] += min( E_break,
                 min( E_continue + obs[ij + di + dj],
                      E_skip + min( obs[ij + di],
                                    obs[ij + dj] )));
    }}
  }

  LOG("  present+future : "
      << min(cost_matrix.cost()) << ", "
      << max(cost_matrix.cost()));
  LOG("  past+present+future : "
      << min(m_posterior) << ", "
      << max(m_posterior));

  LOG("...aligned audio in " << timer.elapsed() << " sec");

}

//----( alignment path )------------------------------------------------------

AlignmentPath::AlignmentPath (
    const AlignmentModel & model,
    const CostMatrix & cost)

  : m_size(cost.size()),
    m_model(model),

    m_cost(cost.cost()),
    m_post(sqr(m_size)),

    m_pressure(0),
    m_dpressure(0)
{
  propagate();
}

void AlignmentPath::advance (float pressure_reltol)
{
  float one_plus_reltol = 1 + pressure_reltol;
  ASSERT_LT(one_plus_reltol, 1.0f);

  float pressure = max(m_pressure + m_dpressure, m_pressure * one_plus_reltol);
  LOG("advancing pressure " << m_pressure << " -> " << pressure);

  m_pressure = pressure;
  m_dpressure = 0;

  propagate();
}

struct MaybeDeriv
{
  float x,dx;

  void update (float new_x, float new_dx)
  {
    if (new_x < x) { x = new_x; dx = new_dx; }
  }
};

void AlignmentPath::propagate ()
{
  LOG("Propagating alignment energy at pressure = " << m_pressure << "...");
  Timer timer;

  // TODO figure out how to propagate after fusing vertices

  const size_t I = size();
  const size_t di = I;
  const size_t dj = 1;

  ASSERT_LE(0, m_pressure);
  const float pressure = m_pressure;
  const float E_break = logf(m_model.segment_timescale);
  const float E_skip = -logf(0.5f * m_model.log_pos_sigma);

  Vector<float> temp1(I * I);
  Vector<float> temp2(I * I);
  Vector<float> temp3(I * I);

  const float * obs = m_cost;
  float * restrict fwd = m_post;
  float * restrict dfwd = temp1;
  float * restrict bwd = temp2;
  float * restrict dbwd = temp3;

  LOG("propagating fwd & bwd");
  #pragma omp parallel sections
  {
    // forward pass
    #pragma omp section
    {
      for (size_t i = 0; i < I; ++i) {
      for (size_t j = 0; j < I; ++j) {
        size_t ij = I * i + j;

        float p0 = obs[ij] - pressure;
        float dp0 = -1;

        MaybeDeriv p = {p0 + E_break, dp0};

        if (i and j) p.update(p0 + fwd[ij - di - dj], dp0 + dfwd[ij - di - dj]);
        if (i) p.update(p0 + fwd[ij - di] + E_skip, dp0 + dfwd[ij - di]);
        if (j) p.update(p0 + fwd[ij - dj] + E_skip, dp0 + dfwd[ij - dj]);

        fwd[ij] = p.x;
        dfwd[ij] = p.dx;
      }}
    }

    // backward pass
    #pragma omp section
    {
      for (size_t i = 0; i < I; ++i) {
      for (size_t j = 0; j < I; ++j) {
        size_t ij = I * (I-1 - i) + (I-1 - j);

        float p0 = obs[ij] - pressure;
        float dp0 = -1;

        MaybeDeriv p = {p0 + E_break, dp0};

        if (i and j) p.update(p0 + bwd[ij + di + dj], dp0 + dbwd[ij + di + dj]);
        if (i) p.update(p0 + bwd[ij + di] + E_skip, dp0 + dbwd[ij + di]);
        if (j) p.update(p0 + bwd[ij + dj] + E_skip, dp0 + dbwd[ij + dj]);

        bwd[ij] = p.x;
        dbwd[ij] = p.dx;
      }}
    }
  }

  // fusion pass
  LOG("fusing fwd + bwd data");
  float collision_time = INFINITY;
  size_t fused_pairs = 0;
  for (size_t i = 0; i < I; ++i) {
  for (size_t j = 0; j < I; ++j) {
    size_t ij = I * i + j;

    float p0 = obs[ij] - pressure;
    float dp0 = -1;

    float p = fwd[ij] + bwd[ij] - p0;
    float dp = dfwd[ij] + dbwd[ij] - dp0;

    if (p < 0) {

      ++fused_pairs;
      fwd[ij] = 0;

    } else {

      float t = -p / dp;
      imin(collision_time, t);
      fwd[ij] = t;
    }
  }}

  PRINT2(collision_time, fused_pairs);
  m_dpressure = collision_time;

  LOG("...propagated alignment energy in " << timer.elapsed() << " sec");
}

} // namespace AudioAlignment

