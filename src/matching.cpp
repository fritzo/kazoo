
#include "matching.h"
#include <set>

#define LOG1(message)

#define ASSERT_PROB(x) { \
  ASSERT(TOL <= (x), #x " is below tolerance: " << (x)); \
  ASSERT((x) <= 1/ TOL, #x " is above tolerance: " << (x)) }
#define ASSERT1_PROB(x)

#define ASSERTW_COST(x) { \
  ASSERTW(-BIG <= (x), #x " is below tolerance: " << (x)); \
  ASSERTW((x) <= BIG, #x " is above tolerance: " << (x)); }
#define ASSERT1_COST(x)

namespace Matching
{

inline void clamp_nonneg (float & value)
{
  if (value < 0) value = 0;
}

inline float safe (float value)
{
  if (not (value > TOL)) return TOL;
  if (not (value < 1 / TOL)) return 1 / TOL;
  return value;
}

inline float safe_div (float numerator, float denominator)
{
  if (not (numerator > denominator * TOL)) return TOL;
  if (not (denominator > numerator * TOL)) return 1 / TOL;
  return numerator / denominator;
}

inline float likelihood_ratio (float part, float whole)
{
  return part / (whole - part);
}

std::ostream & operator<< (std::ostream & os, const Arc & arc)
{
  return os << '(' << arc.i << ',' << arc.j << ')';
}

inline bool operator< (const Arc & arc_1, const Arc & arc_2)
{
  return arc_1.i != arc_2.i ? arc_1.i < arc_2.i
                              : arc_1.j < arc_2.j;
}

//----( soft assignment solver )----------------------------------------------

void SoftMatching::print_prior () const
{
  LOG("matching.prior = ");
  for (size_t i = 0; i < size_1(); ++i) {
    LOG(" (" << i << ",-) \t" << prior_1_non(i));
  }
  for (size_t j = 0; j < size_2(); ++j) {
    LOG(" (-," << j << ") \t" << prior_2_non(j));
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    LOG(" " << arc(ij) << "\t" << prior_ass(ij));
  }
}

void SoftMatching::print_post () const
{
  LOG("matching.post = ");
  for (size_t i = 0; i < size_1(); ++i) {
    LOG(" (" << i << ",-) \t" << post_1_non(i));
  }
  for (size_t j = 0; j < size_2(); ++j) {
    LOG(" (-," << j << ") \t" << post_2_non(j));
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    LOG(" " << arc(ij) << "\t" << post_ass(ij));
  }
}

//----( input )----

void SoftMatching::clear ()
{
  m_arcs.clear();

  m_prior_1_non.clear();
  m_prior_2_non.clear();
  m_prior_1_ass.clear();
  m_prior_2_ass.clear();

  m_post_1_non.clear();
  m_post_2_non.clear();
  m_post_1_ass.clear();
  m_post_2_ass.clear();

  m_scale_1.clear();
  m_scale_2.clear();
  m_total.clear();
  m_message.clear();
}

void SoftMatching::validate_problem () const
{
  LOG1("validating matching");

  size_t size_1 = m_prior_1_non.size();
  size_t size_2 = m_prior_2_non.size();
  size_t size_arc = m_prior_1_ass.size();

  for (size_t i = 0; i < size_1; ++i)       ASSERT_PROB(m_prior_1_non[i]);
  for (size_t j = 0; j < size_2; ++j)       ASSERT_PROB(m_prior_2_non[j]);
  for (size_t ij = 0; ij < size_arc; ++ij)  ASSERT_PROB(m_prior_1_ass[ij]);

  std::set<Arc> unique_arcs;
  for (size_t ij = 0; ij < size_arc; ++ij) {
    Arc arc = m_arcs[ij];

    ASSERT_LT(arc.i, size_1);
    ASSERT_LT(arc.j, size_2);
    ASSERT(unique_arcs.find(arc) == unique_arcs.end(),
           "arc " << arc << " appears twice");
    unique_arcs.insert(arc);
  }
}

void SoftMatching::validate_solution ()
{
  LOG1("validating solution");

  m_total.resize(max(size_1(), size_2()));

  LOG1("checking normalization over arcs (i,-)");
  for (size_t i = 0; i < size_1(); ++i) {
    ASSERT_LE(0, m_post_1_non[i]);
    ASSERT_LE(m_post_1_non[i], 1);
    m_total[i] = m_post_1_non[i];
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_total[arc.i] += m_post_1_ass[ij];
  }
  for (size_t i = 0; i < size_1(); ++i) {
    ASSERT_LE(fabs(m_total[i] - 1), TOL);
  }

  LOG1("checking normalization over arcs (-,j)");
  for (size_t j = 0; j < size_2(); ++j) {
    ASSERT_LE(0, m_post_2_non[j]);
    ASSERT_LE(m_post_2_non[j], 1);
    m_total[j] = m_post_2_non[j];
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_total[arc.j] += m_post_1_ass[ij];
  }
  for (size_t j = 0; j < size_2(); ++j) {
    ASSERT_LE(fabs(m_total[j] - 1), TOL);
  }
}

//----( propagation tools )----

void SoftMatching::total_1 ()
{
  for (size_t i = 0; i < size_1(); ++i) {
    m_total[i] = m_post_1_non[i];
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_total[arc.i] += m_post_1_ass[ij];
  }
}

void SoftMatching::total_2 ()
{
  for (size_t j = 0; j < size_2(); ++j) {
    m_total[j] = m_post_2_non[j];
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_total[arc.j] += m_post_2_ass[ij];
  }
}

void SoftMatching::propagate_12 ()
{
  total_1();

  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];

    float projected = likelihood_ratio(m_post_1_ass[ij], m_total[arc.i]);
    m_message[ij] = safe(projected / (m_message[ij] * m_prior_1_ass[ij]));
    m_prior_1_ass[ij] = projected;

    float lifted = m_message[ij] * m_scale_2[arc.j];
    m_post_2_ass[ij] = safe(m_post_2_ass[ij] * lifted);

    ASSERT1_PROB(m_message[ij]);
    ASSERT1_PROB(m_prior_1_ass[ij]);
    ASSERT1_PROB(m_post_2_ass[ij]);
  }
}

void SoftMatching::propagate_21 ()
{
  total_2();

  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];

    float projected = likelihood_ratio(m_post_2_ass[ij], m_total[arc.j]);
    m_message[ij] = safe(projected / (m_message[ij] * m_prior_2_ass[ij]));
    m_prior_2_ass[ij] = projected;

    float lifted = m_message[ij] * m_scale_1[arc.i];
    m_post_1_ass[ij] = safe(m_post_1_ass[ij] * lifted);

    ASSERT1_PROB(m_message[ij]);
    ASSERT1_PROB(m_prior_2_ass[ij]);
    ASSERT1_PROB(m_post_1_ass[ij]);
  }
}

void SoftMatching::normalize ()
{
  total_1();

  for (size_t i = 0; i < size_1(); ++i) {
    m_total[i] = 1 / m_total[i];
    m_post_1_non[i] = 1;
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_post_1_ass[ij] *= m_total[arc.i];
  }

  total_2();

  for (size_t j = 0; j < size_2(); ++j) {
    m_total[j] = 1 / m_total[j];
    m_post_2_non[j] = 1;
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_post_2_ass[ij] *= m_total[arc.j];

    float prob = min(m_post_1_ass[ij], m_post_2_ass[ij]);

    m_post_1_ass[ij] = prob;
    m_post_1_non[arc.i] -= prob;
    m_post_2_non[arc.j] -= prob;
  }

  for (size_t i = 0; i < size_1(); ++i) {
    clamp_nonneg(m_post_1_non[i]);
  }
  for (size_t j = 0; j < size_2(); ++j) {
    clamp_nonneg(m_post_2_non[j]);
  }
}

//----( propagation )----

/** 2D assignment by belief propagation.

naming conventions in this note:
  p = prior
  P = post
  m = message

types:
  p2,P1 : 1
  p2,P2 : 2
  p12,m12,m21 : 12
  ^1 : 12 -> 1
  ^2 : 12 -> 2
  _12 : 1 -> 12
  _12 : 2 -> 12

algorithm:

  initialize:
    P1(0) = p1 p12^1
    P2(0) = p2 p12^2
    m21(0) = 1

  iteratively propagate belief:
    for t in [0,...,T-1]:
      m12(t+1) = P1(t)_12 / (p12 m21(t))
      P2(t+1) = P2(t) m12(t+1)^2
      m21(t+1) = P2(t)_12 / (p12 m12(t))
      P1(t+1) = P1(t) m21(t+1)^1

  finalize:
    P[i,j] = min(P1(T)[i,j], P2(T)[i,j])
    P[i,-] = 1 - sum j. P1(T)[i,j]
    P[-,j] = 1 - sum i. P2(T)[i,j]


representation:
  P1[i,-] = post_1_non[i] / total
  P1[i,j] = post_1_ass[ij] / total
  P2[-,j] = post_2_non[j] / total
  P2[i,j] = post_2_ass[ij] / total

operations:
  projection:
    P1_12[ij] = post_1_ass[ij] / (total - post_1_ass[ij])
  lifting:
    m^1[ij] = m[ij] * (size_1[i] - 1),   whence size_1[i] - 1 =: scale1[i]
  fusion:
    (pointwise multiplication)
*/
void SoftMatching::solve (size_t num_iters)
{
  m_post_1_non.resize(size_1());
  m_post_2_non.resize(size_2());
  m_post_1_ass.resize(size_arc());
  m_post_2_ass.resize(size_arc());
  m_scale_1.resize(size_1());
  m_scale_2.resize(size_2());
  m_total.resize(max(size_1(), size_2()));
  m_message.resize(size_arc());

  LOG1("initialize");

  for (size_t i = 0; i < size_1(); ++i) {
    m_scale_1[i] = 0;
  }
  for (size_t j = 0; j < size_2(); ++j) {
    m_scale_2[j] = 0;
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];

    m_scale_1[arc.i] += 1;
    m_scale_2[arc.j] += 1;
  }

  for (size_t i = 0; i < size_1(); ++i) {
    m_post_1_non[i] = m_prior_1_non[i] * m_scale_1[i];
  }
  for (size_t j = 0; j < size_2(); ++j) {
    m_post_2_non[j] = m_prior_2_non[j] * m_scale_2[j];
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];

    m_post_1_ass[ij] = m_prior_1_ass[ij] * m_scale_1[arc.i];
    m_post_2_ass[ij] = m_prior_2_ass[ij] * m_scale_2[arc.j];

    m_message[ij] = 1 / m_scale_2[arc.j]; // uniform
  }

  LOG1("propagate belief");

  for (size_t iter = 0; iter < num_iters; ++iter) {

    LOG1(" propagation step " << iter);

    propagate_12();
    propagate_21();
  }

  LOG1("finalize to ensure normalization");

  normalize();
}

//----( hard assignment solver )----------------------------------------------

void HardMatching::print_prior () const
{
  LOG("matching.prior = ");
  for (size_t i = 0; i < size_1(); ++i) {
    LOG(" (" << i << ",-) \t" << prior_1_non(i));
  }
  for (size_t j = 0; j < size_2(); ++j) {
    LOG(" (-," << j << ") \t" << prior_2_non(j));
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    LOG(" " << arc(ij) << "\t" << prior_ass(ij));
  }
}

void HardMatching::print_post () const
{
  LOG("matching.post = ");
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    LOG(" " << arc(ij) << "\t" << m_post_ass[ij]);
  }

  LOG("matching.solution = ");
  for (size_t i = 0; i < size_1(); ++i) {
    if (post_1_non(i)) LOG(" (" << i << ",-)");
  }
  for (size_t j = 0; j < size_2(); ++j) {
    if (post_2_non(j)) LOG(" (-," << j << ")");
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    if (post_ass(ij)) LOG(" " << arc(ij));
  }
}

//----( input )----

void HardMatching::clear ()
{
  m_arcs.clear();

  m_prior_1_non.clear();
  m_prior_2_non.clear();
  m_prior_ass.clear();

  m_message_12.clear();
  m_message_21.clear();
  m_post_ass.clear();

  m_optima.clear();

  m_1_non.clear();
  m_2_non.clear();
  m_1_ass.clear();
  m_2_ass.clear();
}

void HardMatching::validate_problem () const
{
  LOG1("validating matching");

  for (size_t i = 0; i < size_1(); ++i)       ASSERTW_COST(m_prior_1_non[i]);
  for (size_t j = 0; j < size_2(); ++j)       ASSERTW_COST(m_prior_2_non[j]);
  for (size_t ij = 0; ij < size_arc(); ++ij)  ASSERTW_COST(m_prior_ass[ij]);

  std::set<Arc> unique_arcs;
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];

    ASSERT_LT(arc.i, size_1());
    ASSERT_LT(arc.j, size_2());
    ASSERT(unique_arcs.find(arc) == unique_arcs.end(),
           "arc " << arc << " appears twice");
    unique_arcs.insert(arc);
  }
}

void HardMatching::validate_solution ()
{
  LOG1("validating solution");

  std::vector<size_t> total(max(size_1(), size_2()));

  LOG1("checking normalization over arcs (i,-)");
  for (size_t i = 0; i < size_1(); ++i) {
    total[i] = post_1_non(i) ? 1 : 0;
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    total[arc.i] += post_ass(ij) ? 1 : 0;
  }
  for (size_t i = 0; i < size_1(); ++i) {
    ASSERT_EQ(total[i], 1);
  }

  LOG1("checking normalization over arcs (-,j)");
  for (size_t j = 0; j < size_2(); ++j) {
    total[j] = post_2_non(j) ? 1 : 0;
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    total[arc.j] += post_ass(ij) ? 1 : 0;
  }
  for (size_t j = 0; j < size_2(); ++j) {
    ASSERT_EQ(total[j], 1);
  }
}

//----( propagation tools )----

void HardMatching::optimize_1 ()
{
  for (size_t i = 0; i < size_1(); ++i) {
    m_optima[i].init(m_prior_1_non[i]);
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_optima[arc.i].update(m_post_ass[ij]);
  }
}

void HardMatching::optimize_2 ()
{
  for (size_t j = 0; j < size_2(); ++j) {
    m_optima[j].init(m_prior_2_non[j]);
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_optima[arc.j].update(m_post_ass[ij]);
  }
}

void HardMatching::propagate_12 ()
{
  optimize_1();

  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];

    float message = m_message_12[ij]
                 = -m_optima[arc.i].best_alternative(m_post_ass[ij]);
    m_post_ass[ij] = message
                   + m_message_21[ij]
                   + m_prior_ass[ij];

    ASSERT1_COST(m_post_ass[ij]);
  }
}

void HardMatching::propagate_21 ()
{
  optimize_2();

  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];

    float message = m_message_21[ij]
                 = -m_optima[arc.j].best_alternative(m_post_ass[ij]);
    m_post_ass[ij] = message
                   + m_message_12[ij]
                   + m_prior_ass[ij];

    ASSERT1_COST(m_post_ass[ij]);
  }
}

void HardMatching::normalize ()
{
  optimize_1();

  for (size_t i = 0; i < size_1(); ++i) {
    m_1_non[i] = true;
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_1_ass[ij] = (m_post_ass[ij] == m_optima[arc.i]);
  }

  optimize_2();

  for (size_t j = 0; j < size_2(); ++j) {
    m_2_non[j] = true;
  }
  for (size_t ij = 0; ij < size_arc(); ++ij) {
    Arc arc = m_arcs[ij];
    m_2_ass[ij] = (m_post_ass[ij] == m_optima[arc.j]);

    if (post_1_ass(ij) and post_2_ass(ij)) {
      m_1_ass[ij] = true;
      m_1_non[arc.i] = false;
      m_2_non[arc.j] = false;
    } else {
      m_1_ass[ij] = false;
    }
  }
}

//----( propagation )----

/** 2D assignment by belief propagation.
*/
void HardMatching::solve (size_t num_iters)
{
  m_message_12.resize(size_arc());
  m_message_21.resize(size_arc());
  m_post_ass.resize(size_arc());

  m_optima.resize(max(size_1(), size_2()));

  m_1_non.resize(size_1());
  m_2_non.resize(size_2());
  m_1_ass.resize(size_arc());
  m_2_ass.resize(size_arc());

  LOG1("initialize");

  for (size_t ij = 0; ij < size_arc(); ++ij) {
    m_message_12[ij] = 0;
    m_message_21[ij] = 0;
  }
  m_post_ass = m_prior_ass;

  LOG1("propagate belief");

  for (size_t iter = 0; iter < num_iters; ++iter) {

    LOG1(" propagation step " << iter);

    propagate_12();
    propagate_21();
  }

  LOG1("finalize to ensure normalization");

  normalize();
}

} // namespace Matching

