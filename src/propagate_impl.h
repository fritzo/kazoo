#ifndef KAZOO_PROPAGATE_IMPL_H
#define KAZOO_PROPAGATE_IMPL_H

/** Belief Propagation Algorithms

  Definition:
    Let e : X -> float be an energy function on the set X.
    For an element x:X, we define
    * the likelihood of x           z(x) = exp(-e(x))
    * the partition function        Z = sum x:X. z(x)
    * the (Helmoholtz) normalized  A = -log(Z)
    * the normalized of x          a(x) =
    * the probability of x          p(x) = z(x) / Z = exp(a(x) - A)
    * the relative of x           de(x) = -log(p(x) / (1 - p(x)))
    []

  Definition:
    Let a be an arc with energy e and views view(i) into nodes node(i).
    We say (e,views) are consistent iff
    * e : relative, and
    * before any other arc.propagate(),
        after node(i).normalize(),
          bernoulli_energy_gap(view(i)) = e.
    []

  Definition:
    Let arcs : set<Arc> and node : set<Node> be an abstract propagation problem
    and e_none : nodes -> float and e_arc : arcs -> float be relative.
    The abstract propagation algorithm is then

      nodes.init()
      arcs.init()
      nodes.normalize()
      until converged:
        arcs.propagate()
        nodes.normalize()
      arcs.constrain()
      nodes.constrain()

    or equivalently

      nodes.init()
      arcs.init()
      until converged:
        nodes.normalize()
        arcs.propagate()
      nodes.normalize()
      arcs.constrain()
      nodes.constrain()

    []

  Lemmata:
    All precondition,postcondition pairs below.
    TODO ensure these all hold

  Theorem:
    Let num_iters >= 0.
    After the abstract propagation algorithm,
    all nodes and arcs are feasible consistent normalized.
  Proof:
    By induction, composing precondition,postcondition pairs.

      nodes.init()
      |- nodes relative
      arcs.init()
      |- nodes relative & arcs relative
      nodes.normalize()
      |- nodes normalized & arcs relative
      loop:
        |- nodes normalized & arcs relative
        arcs.propagate()
        |- nodes normalized & arcs relative
        nodes.normalize()
        |- nodes feasible normalized & arcs relative
      |- nodes feasible normalized
      arcs.constrain()
      |- nodes feasible normalized & arcs consistent normalized
      nodes.contrain()
      |- nodes feasible consistent normalized & arcs consistent normalized
    [pending precondition,postcondition lemmata]

  Desired Theorem:
    Propagation is exact in the trivial case of exactly one arc.
  Proof:
    TODO

  Desired Theorem:
    Propagation is exact in trees.
  Proof:
    TODO

  TODO write unit tests for trees (where bp should be exact)
*/

#include "common.h"

//#define KAZOO_NDEBUG_PROPAGATE
#ifdef KAZOO_NDEBUG_PROPAGATE

#define ASSERT2(cond,mess)
#define ASSERT2_EQ(x,y)
#define ASSERT2_LE(x,y)
#define ASSERT2_LT(x,y)

#else // KAZOO_NDEBUG_PROPAGATE

#define ASSERT2 ASSERT
#define ASSERT2_EQ ASSERT_EQ
#define ASSERT2_LE ASSERT_LE
#define ASSERT2_LT ASSERT_LT

#endif // KAZOO_NDEBUG_PROPAGATE

#define LOG1(mess) LOG(mess)

#define TOL                             (1e-4f)
#define PROPAGATE_MAX_LIKE              (1 / TOL)
#define PROPAGATE_MIN_LIKE              (TOL)
#define PROPAGATE_MAX_PROB              (1 - TOL)
#define PROPAGATE_MIN_PROB              (1 / TOL)
#define PROPAGATE_MIN_ENERGY            (logf(TOL))
#define PROPAGATE_MAX_ENERGY            (-logf(TOL))

//#define PROPAGATE_CONSTRAIN_NONE

namespace Propagate
{

//----( data structures )-----------------------------------------------------

template<class T, size_t fixed_size>
class Array
{
  T m_data[fixed_size];

public:

  operator T * () { return m_data; }
  operator const T * () const { return m_data; }

  Array (size_t) {} // for uniformity with dynamically-sized arrays

  size_t size () const { return fixed_size; }
};

template<class T>
class Array<T, 0>
{
  T * m_data;
  size_t m_size;

public:

  operator T * () { return m_data; }
  operator const T * () const { return m_data; }

  Array (size_t size) : m_data(new T[size]) {}
  ~Array () { delete[] m_data; }

  size_t size () const { return m_size; }
};

//----( safe math )-----------------------------------------------------------

inline float prob_to_like (float x)
{
  //if (x > PROPAGATE_MAX_PROB) return INFINITY;
  if (x < PROPAGATE_MIN_PROB) return 0;
  return x / (1 - x);
}

inline bool clamp_like (float & x)
{
  //if (x > PROPAGATE_MAX_LIKE) { x = INFINITY; return true; }
  if (x < PROPAGATE_MIN_LIKE) { x = 0.0f; return true; }
  return false;
}

inline bool energy_is_tiny (float e) { return e < PROPAGATE_MIN_ENERGY; }
inline bool energy_is_huge (float e) { return e > PROPAGATE_MAX_ENERGY; }

inline bool ensure_normal_like (float & x)
{
  if (x > PROPAGATE_MAX_LIKE) { x = PROPAGATE_MAX_LIKE; return true; }
  if (x < PROPAGATE_MIN_LIKE) { x = PROPAGATE_MIN_LIKE; return true; }
  return false;
}

//----( shared bernoulli variables )------------------------------------------

template<size_t fixed_size = 0>
class BernoulliArc
{
  float m_value;
  Array<float *, fixed_size> m_views;
  Array<float, fixed_size> m_messages;

  float & view (size_t i) { return * (m_views[i]); }
  size_t size () { return m_views.size(); }

public:

  //----( construction )----

  BernoulliArc (size_t size = 0) : m_views(size), m_messages(size) {}

  float * & operator[] (size_t i)
  {
    ASSERT2_LT(i, size());
    return m_views[i];
  }
  const float * init (float energy_gap)
  {
    m_value = expf(-energy_gap);
    if (ensure_normal_like(m_value)) {
      WARN("clipping out-of-bound arc energy " << energy_gap);
    }

    for (size_t i = 0; i < size(); ++i) {
      view(i) = m_value;
    }
    return & m_value;

    // postcondition: m_value relative
    // postcondition: /\i. view(i) = m_value
  }

  //----( solution )----

  void propagate ()
  {
    // precondition: m_value : relative
    // precondition: views are either
    //   independent relative, or dependent normalized

    if (clamp_like(m_value)) {
      LOG1("clamping BernoulliArc infinite energy");
      for (size_t i = 0; i < size(); ++i) {
        view(i) = m_value;
      }
      return;
    }

    float * restrict messages = m_messages;

    float total_message = 0;
    for (size_t i = 0; i < size(); ++i) {
      float observed = prob_to_like(view(i));
      float message_in = observed / m_value;

      messages[i] = message_in;
      total_message *= message_in;
    }

    m_value *= total_message;
    for (size_t i = 0; i < size(); ++i) {
      float message_out = total_message / messages[i];

      view(i) *= message_out;
    }

    // postcondition: m_value normalized
    // postcondition: (m_value,views) are consistent
  }

  void constrain ()
  {
    // precondition: views are normalized

    m_value = view(0);
    for (size_t i = 1; i < size(); ++i) {
      imin(m_value, view(i));
    }
    for (size_t i = 0; i < size(); ++i) {
      view(i) = m_value;
    }

    // postcondition: m_value feasible normalized
    // postcondition: /\i. view(i) = m_value
  }
};

//----( choice variables )----------------------------------------------------

template<size_t max_size = 5>
class ChoiceNode
{
  size_t m_size;
  float m_values[1 + max_size];

public:

  float & none () { return m_values[0]; }
  float & some (size_t i) { return m_values[1 + i]; }

  //----( construction )----

  ChoiceNode () : m_size(0) {}

  void init (float none_energy_gap)
  {
    none() = expf(-none_energy_gap);
    if (ensure_normal_like(none())) {
      WARN("clipping out-of-bound choice.none energy " << none_energy_gap);
    }

    // postcondition: m_values are relative
  }

  bool full () { return m_size == max_size; }
  float * new_choice ()
  {
    ASSERT2(not full(), "added too many choices to ChoiceNode");
    return m_values + ++m_size;
  }

  //----( solution )----

  float norm ()
  {
    float total = none();
    for (size_t i = 1; i < m_size; ++i) {
      total += some(i);
    }
    return total;
  }

  void normalize (float scale)
  {
    none() *= scale;
    for (size_t i = 0; i < m_size; ++i) {
      some(i) *= scale;
    }
  }

  void normalize ()
  {
    // precondition: m_values are either
    // * dependent normalized or
    // * independent relative

    normalize(1 / norm());

    // postcondition: m_values are normalized
  }

  void constrain ()
  {
    // precondition: m_values[1:] are feasible probabilities

    // put extra mass on first choice
    none() = 1;
    for (size_t i = 0; i < m_size; ++i) {
      none() -= some(i);
    }
    ASSERT_LE(0, none());

    // postcondition: m_values are feasible probabilities
  }
};

//----( product variables )---------------------------------------------------

template<size_t max_choices, size_t fixed_size = 0>
class ProductNode
{
public:

  typedef ChoiceNode<max_choices> Factor;

private:

  float m_none;
  Array<Factor, fixed_size> m_factors;
  Array<float, fixed_size> m_norms;

  size_t size () const { return m_factors.size(); }

public:

  ProductNode (size_t size = 0) : m_factors(size), m_norms(size) {}

  float & none () { return m_none; }
  Factor & factor (size_t i)
  {
    ASSERT2_LT(i, size());
    return m_factors[i];
  }

  //----( construction )----

  void init (float none_energy_gap)
  {
    none() = expf(-none_energy_gap);
    if (ensure_normal_like(none())) {
      WARN("clipping out-of-bound product.none energy " << none_energy_gap);
    }

    // postcondition: m_none is relative
  }

  //----( solution )----

  void normalize ()
  {
    // precondition: m_none, m_factors are ???

    Factor * restrict factors = m_factors;
    float * restrict norms = m_norms;

    float total = 1;
    for (size_t i = 0; i < size(); ++i) {
      norms[i] = factors[i].norm();
      total *= norms[i];
    }
    total += none();

    float scale = 1 / total;
    none() *= scale;
    for (size_t i = 0; i < size(); ++i) {
      factors[i].normalize(scale * norms[i]);
    }

    // postcondition: m_none, m_factors are normalized
  }

  void constrain ()
  {
    // precondition: m_factors are feasible normalized

    // put maximum extra mass on m_none
    // TODO work out more realistic constraint method
    none() = 0;
    for (size_t i = 0; i < size(); ++i) {
      Factor & f = factor(i);
      f.constrain();
      imin(none(), f.none());
    }
    for (size_t i = 0; i < size(); ++i) {
      factor(i).none() -= none();
    }

    // postcondition: m_none, m_factors are feasible probabilities
  }
};

}// namespace Propagate

#undef ASSERT2
#undef ASSERT2_EQ
#undef ASSERT2_LE
#undef ASSERT2_LT

#undef LOG1

#endif // KAZOO_PROPAGATE_IMPL_H
