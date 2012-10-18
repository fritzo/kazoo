#ifndef KAZOO_PROPAGATE_H
#define KAZOO_PROPAGATE_H

/** Belief Propagation Applications
*/

#include "propagate_impl.h"
#include "hash_map.h"
#include <utility>
#include <vector>
#include <algorithm>

#define LOG1(mess) LOG(mess)
#define LOG2(mess)
#define DEBUG1(mess) DEBUG(mess)
#define DEBUG2(mess)

namespace Propagate
{

//----( data structures )-----------------------------------------------------

template<class T>
T & add_position (std::vector<T> & v)
{
  v.resize(v.size() + 1);
  return v.back();
}

//----( multi-frame assignment )----------------------------------------------

template<size_t max_choices = 0>
class MFASolver
{
  typedef ChoiceNode<max_choices> Node;
  typedef BernoulliArc<0> Arc;

  std::vector<Node> m_nodes;
  std::vector<Arc> m_arcs;

public:

  //----( construction )----

  Id add_detection (float md_energy)
  {
    Id id = m_nodes.size();
    Node & node = add_position(m_nodes);

    node.init(md_energy);

    return id;
  }

  const float * try_add_track (const std::vector<Id> & ids, float energy)
  {
    for (size_t i = 0; i < ids.size(); ++i) {
      Node & detection = m_nodes[ids[i]];
      if (detection.full()) return NULL;
    }

    Arc & track = add_position(m_arcs);

    for (size_t i = 0; i < ids.size(); ++i) {
      Node & detection = m_nodes[ids[i]];
      track[i] = detection.new_choice();
    }

    return track.init(energy);
  }

  //----( solution )----

  void solve (size_t num_ters)
  {
    LOG("Propagating MFA solution with "
        << m_nodes.size() << " nodes and "
        << m_arcs.size() << " arcs");

    for (size_t iter = 0; iter < num_ters; ++iter) {
      LOG(" propagating iteration " << iter);
      for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].normalize();
      for (size_t i = 0; i < m_arcs.size(); ++i) m_arcs[i].propagate();
    }

    LOG(" converting energies to probabilities");
    for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].normalize();
    for (size_t i = 0; i < m_arcs.size(); ++i) m_arcs[i].constrain();
    for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].constrain();
  }
};

//----( factored multi-frame assignment )-------------------------------------

template<size_t max_choices = 0>
class FactoredMFASolver
{
  typedef ProductNode<max_choices, 2> Node;
  typedef typename Node::Factor Choice;
  typedef BernoulliArc<2> Arc;

  enum End {
    TAIL = 0,
    HEAD = 1
  };
  enum Direction {
    BEFORE = 0,
    AFTER = 1
  };

  std::vector<Node> m_nodes;
  std::vector<Arc> m_arcs;

public:

  //----( construction )----

  Id add_detection (float fa_energy, float begin_energy, float end_energy)
  {
    Id id = m_nodes.size();
    Node & node = add_position(m_nodes);

    node.init(fa_energy);
    node.factor(BEFORE).init(begin_energy);
    node.factor(AFTER).init(end_energy);

    return id;
  }

  const float * try_add_tracklet (Id source, Id destin, float energy)
  {
    if (energy_is_huge(energy)) return NULL;

    Choice & tail = m_nodes[source].factor(AFTER);
    Choice & head = m_nodes[destin].factor(BEFORE);

    if (tail.full() or head.full()) return NULL;

    Arc & arc = add_position(m_arcs);

    arc[TAIL] = tail.new_choice();
    arc[HEAD] = head.new_choice();

    return arc.init(energy);
  }

  //----( solution )----

  void solve (size_t num_ters)
  {
    LOG("Propagating factored MFA solution with "
        << m_nodes.size() << " nodes and "
        << m_arcs.size() << " arcs");

    for (size_t iter = 0; iter < num_ters; ++iter) {
      LOG(" propagating iteration " << iter);
      for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].normalize();
      for (size_t i = 0; i < m_arcs.size(); ++i) m_arcs[i].propagate();
    }

    LOG(" converting energies to probabilities");
    for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].normalize();
    for (size_t i = 0; i < m_arcs.size(); ++i) m_arcs[i].constrain();
    for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].constrain();
  }
};

template<size_t max_choices = 0>
class FactoredMFAProblem
{
  typedef FactoredMFASolver<max_choices> Solver;

  struct Detection
  {
    Id user_id;
    Id solver_id;

    float fa_energy;
    float begin_energy;
    float end_energy;

    Detection () {}
    Detection (Id id, float fa, float begin, float end, Solver & solver)
      : user_id(id),
        solver_id(solver.add_detection(fa, begin, end)),

        fa_energy(fa),
        begin_energy(begin),
        end_energy(end)
    {}
  };

  class Arc
  {
  public:
    Detection * tail;
    Detection * head;
    float energy;
  private:
    float m_prior;
    const float * m_posterior;
  public:

    Arc (Detection * t, Detection * h, float e)
      : tail(t),
        head(h),
        energy(e),
        m_prior( energy
               - tail->fa_energy
               - head->fa_energy
               - tail->end_energy + head->begin_energy
               ),
        m_posterior(NULL)
    {}

    bool operator< (const Arc & other) const
    {
      return m_prior < other.m_prior;
    }

    void try_add_to (Solver & solver)
    {
      Id tail_id = tail->solver_id;
      Id head_id = head->solver_id;

      m_posterior = solver.try_add_tracklet(tail_id, head_id, energy);
    }
    float posterior () const { return m_posterior ? * m_posterior : 0; }
  };

  //----( data )----

  Solver m_solver;

  typedef std::hash_map<Id, Detection> Detections;
  Detections m_detections;

  std::vector<Arc> m_arcs;

public:

  //----( input )----

  bool add_detection (Id id, float fa, float begin, float end)
  {
    if (energy_is_tiny(fa)) return false;

    m_detections[id] = Detection(id, fa, begin, end, m_solver);
    return true;
  }

  void add_arc (Id tail, Id head, float energy)
  {
    if (energy_is_huge(energy)) return;

    typename Detections::iterator tail_iter = m_detections.find(tail);
    if (tail_iter == m_detections.end()) return;
    Detection * tail_detection = & tail_iter->second;

    typename Detections::iterator head_iter = m_detections.find(head);
    if (head_iter == m_detections.end()) return;
    Detection * head_detection = & head_iter->second;

    m_arcs.push_back(Arc(tail_detection, head_detection, energy));
  }

  //----( solving )----

  void solve (size_t num_ters)
  {
    LOG("Solving factored MFA problem with "
        << m_detections.size() << " detections and "
        << m_arcs.size() << " arcs");

    // add the lowest energy arcs first, until solver is full

    std::sort(m_arcs.begin(), m_arcs.end());
    for (size_t i = 0; i < m_arcs.size(); ++i) {
      m_arcs[i].try_add_to(m_solver);
    }

    m_solver.solve(num_ters);
  }

  //----( output )----

  class ArcIterator
  {
    const std::vector<Arc> & m_arcs;
    size_t m_position;

    void advance_while_zero ()
    {
      while ((m_position < m_arcs.size())
          and (m_arcs[m_position].posterior() <= 0)) ++m_position;
    }

  public:

    ArcIterator (const std::vector<Arc> & arcs)
      : m_arcs(arcs),
        m_position(0)
    {
      advance_while_zero();
    }

    operator bool () const { return m_position < m_arcs.size(); }
    void next () { ++m_position; advance_while_zero(); }

    Id tail () const { return m_arcs[m_position].tail->user_id; }
    Id head () const { return m_arcs[m_position].head->user_id; }
    float prob () const { return m_arcs[m_position].posterior(); }
  };

  ArcIterator arcs () { return ArcIterator(m_arcs); }
};

//----( grid matching )-------------------------------------------------------

/** Grid Matching

  Problem:
    Suppose we have a set V of [candidate] gridpoints and
    sets NS and EW of [candidate] vertical and horizontal directed edges, resp.
    We wish to find subsets of V,NS,EW
    where each gridpoint has at most one edge in each N,E,S,W direction.
*/

template<size_t max_choices = 0>
class GridMatchingSolver
{
  typedef ProductNode<max_choices, 4> Node;
  typedef typename Node::Factor Choice;
  typedef BernoulliArc<2> Arc;

  enum End {
    TAIL = 0,
    HEAD = 1
  };
  enum Direction {
    NORTH = 0,
    SOUTH = 1,
    EAST = 2,
    WEST = 3
  };

  std::vector<Node> m_nodes;
  std::vector<Arc> m_varcs;
  std::vector<Arc> m_harcs;

public:

  //----( construction )----

  Id add_point (
      float fa_energy,
      float N_energy,
      float S_energy,
      float E_energy,
      float W_energy)
  {
    Id id = m_nodes.size();
    Node & node = add_position(m_nodes);

    node.init(fa_energy);
    node.factor(NORTH).init(N_energy);
    node.factor(SOUTH).init(S_energy);
    node.factor(EAST).init(E_energy);
    node.factor(WEST).init(W_energy);

    return id;
  }

  const float * try_add_arc (Id source, Id destin, bool vertical, float energy)
  {
    if (energy_is_huge(energy)) return NULL;

    Choice & tail = m_nodes[source].factor(vertical ? SOUTH : WEST);
    Choice & head = m_nodes[destin].factor(vertical ? NORTH : EAST);

    if (tail.full() or head.full()) return NULL;

    Arc & arc = add_position(m_varcs);

    arc[TAIL] = tail.new_choice();
    arc[HEAD] = head.new_choice();

    return arc.init(energy);
  }

  //----( solution )----

  void solve (size_t num_ters)
  {
    LOG("Propagating grid matching solution with "
        << m_nodes.size() << " nodes and "
        << m_varcs.size() << "v + " << m_harcs.size() << "h arcs");

    for (size_t iter = 0; iter < num_ters; ++iter) {
      LOG(" propagating iteration " << iter);
      for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].normalize();
      for (size_t i = 0; i < m_varcs.size(); ++i) m_varcs[i].propagate();
      for (size_t i = 0; i < m_harcs.size(); ++i) m_harcs[i].propagate();
    }

    LOG(" converting energies to probabilities");
    for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].normalize();
    for (size_t i = 0; i < m_varcs.size(); ++i) m_varcs[i].constrain();
    for (size_t i = 0; i < m_harcs.size(); ++i) m_harcs[i].constrain();
    for (size_t i = 0; i < m_nodes.size(); ++i) m_nodes[i].constrain();
  }
};

template<size_t max_degree = 0>
class GridMatchingProblem
{
  typedef GridMatchingSolver<max_degree> Solver;

  struct Point
  {
    Id user_id;
    Id solver_id;

    float fa_energy;
    float N_energy;
    float S_energy;
    float E_energy;
    float W_energy;

    Point () {}
    Point (Id id, float fa, float N, float S, float E, float W, Solver & solver)
      : user_id(id),
        solver_id(solver.add_point(fa, N, S, E, W)),

        fa_energy(fa),
        N_energy(N),
        S_energy(S),
        E_energy(E),
        W_energy(W)
    {}
  };

  class Arc
  {
  public:
    Point * tail;
    Point * head;
    bool vertical;
    float energy;
  private:
    float m_prior;
    const float * m_posterior;
  public:

    Arc (Point * t, Point * h, bool v, float e)
      : tail(t),
        head(h),
        vertical(v),
        energy(e),
        m_prior( energy
               - tail->fa_energy
               - head->fa_energy
               - (vertical ? tail->N_energy + head->S_energy
                           : tail->E_energy + head->W_energy)
               ),
        m_posterior(NULL)
    {}

    bool operator< (const Arc & other) const
    {
      return m_prior < other.m_prior;
    }

    void try_add_to (Solver & solver)
    {
      Id tail_id = tail->solver_id;
      Id head_id = head->solver_id;

      m_posterior = solver.try_add_arc(tail_id, head_id, vertical, energy);
    }
    float posterior () const { return m_posterior ? * m_posterior : 0; }
  };

  //----( data )----

  Solver m_solver;

  typedef std::hash_map<Id, Point> Points;
  Points m_points;

  std::vector<Arc> m_varcs;
  std::vector<Arc> m_harcs;

public:

  //----( input )----

  bool add_point (Id id, float fa, float N, float S, float E, float W)
  {
    if (energy_is_tiny(fa)) return false;

    m_points[id] = Point(id, fa, N, S, E, W, m_solver);
    return true;
  }

  void add_varc (Id tail, Id head, float energy)
  {
    if (energy_is_huge(energy)) return;

    typename Points::iterator tail_iter = m_points.find(tail);
    if (tail_iter == m_points.end()) return;
    Point * tail_point = & tail_iter->second;

    typename Points::iterator head_iter = m_points.find(head);
    if (head_iter == m_points.end()) return;
    Point * head_point = & head_iter->second;

    m_varcs.push_back(Arc(tail_point, head_point, true, energy));
  }

  void add_harc (Id tail, Id head, float energy)
  {
    if (energy_is_huge(energy)) return;

    typename Points::iterator tail_iter = m_points.find(tail);
    if (tail_iter == m_points.end()) return;
    Point * tail_point = & tail_iter->second;

    typename Points::iterator head_iter = m_points.find(head);
    if (head_iter == m_points.end()) return;
    Point * head_point = & head_iter->second;

    m_harcs.push_back(Arc(tail_point, head_point, false, energy));
  }

  //----( solving )----

  void solve (size_t num_ters)
  {
    LOG("Solving grid matching problem with "
        << m_points.size() << " points and "
        << m_varcs.size() << "v + " << m_harcs.size() << "h arcs");

    // add the lowest energy edges first, until solver is full

    std::sort(m_varcs.begin(), m_varcs.end());
    for (size_t i = 0; i < m_varcs.size(); ++i) {
      m_varcs[i].try_add_to(m_solver);
    }

    std::sort(m_harcs.begin(), m_harcs.end());
    for (size_t i = 0; i < m_harcs.size(); ++i) {
      m_harcs[i].try_add_to(m_solver);
    }

    m_solver.solve(num_ters);
  }

  //----( output )----

  class ArcIterator
  {
    const std::vector<Arc> & m_arcs;
    size_t m_position;

    void advance_while_zero ()
    {
      while ((m_position < m_arcs.size())
          and (m_arcs[m_position].posterior() <= 0)) ++m_position;
    }

  public:

    ArcIterator (const std::vector<Arc> & arcs)
      : m_arcs(arcs),
        m_position(0)
    {
      advance_while_zero();
    }

    operator bool () const { return m_position < m_arcs.size(); }
    void next () { ++m_position; advance_while_zero(); }

    Id tail () const { return m_arcs[m_position].tail->user_id; }
    Id head () const { return m_arcs[m_position].head->user_id; }
    float prob () const { return m_arcs[m_position].posterior(); }
  };

  ArcIterator varcs () { return ArcIterator(m_varcs); }
  ArcIterator harcs () { return ArcIterator(m_harcs); }
};

} // namespace Propagate

#undef LOG1
#undef LOG2
#undef DEBUG1
#undef DEBUG2

#endif // KAZOO_PROPAGATE_H
