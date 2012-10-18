
#include "network_flow.h"
#include <lemon/smart_graph.h>
#include <lemon/maps.h>
#include <lemon/network_simplex.h>

namespace lemon
{

//----( fast map type )-------------------------------------------------------

template <typename DGR, typename K, typename V>
class KazooMap
{
  Vector<V> m_vector;

public:

  typedef K Key;
  typedef V Value;
  typedef Value & Reference;
  typedef const Value & ConstReference;

  //typedef lemon::True ReferenceMapTag;

  KazooMap(int size, const Value &value = Value())
    : m_vector(size)
  {
    m_vector.set(value);
  }

  KazooMap (const KazooMap<DGR,K,V> & other)
    : m_vector(other.m_vector.size)
  {
    m_vector = other.m_vector;
  }

  int size () { return m_vector.size; }
  int * data () { return m_vector.data; }

private:

  KazooMap & operator= (const KazooMap &);

public:

  Reference operator[] (const Key & k)
  {
    return m_vector[DGR::id(k)];
  }
  ConstReference operator[] (const Key & k) const
  {
    return m_vector[DGR::id(k)];
  }
  void set (const Key & k, const Value & v)
  {
    m_vector[DGR::id(k)] = v;
  }
};

} // namespace lemon

typedef lemon::SmartDigraph Digraph;
typedef lemon::KazooMap<Digraph, Digraph::Node, int> NodeMap;
typedef lemon::KazooMap<Digraph, Digraph::Arc, int> ArcMap;
typedef lemon::NetworkSimplex<Digraph, int, int> Solver;

//----( uniform digraph )-----------------------------------------------------

class UniformDigraphGuts
{
  const size_t m_num_nodes;
  const size_t m_out_degree;

  Digraph m_graph;

public:

  UniformDigraphGuts (
      uint16_t num_nodes,
      uint16_t out_degree,
      const Vector<uint16_t> & tails);

  size_t num_nodes () const { return m_num_nodes; }
  size_t out_degree () const { return m_out_degree; }
  size_t num_arcs () const { return m_num_nodes * m_out_degree; }

  const Digraph & graph () const { return m_graph; }
};

UniformDigraphGuts::UniformDigraphGuts (
    uint16_t num_nodes,
    uint16_t out_degree,
    const Vector<uint16_t> & tails)

  : m_num_nodes(num_nodes),
    m_out_degree(out_degree),
    m_graph()
{
  ASSERT_LE(2, num_nodes);
  ASSERT_LE(1, out_degree);
  ASSERT_SIZE(tails, num_arcs());

  LOG("building UniformDigraph with " << num_nodes << " nodes x "
      << out_degree << " edges/node");

  Digraph & restrict g = m_graph;

  for (int n = 0; n < num_nodes; ++n) {
    ASSERT_EQ(n, Digraph::id(g.addNode()));
  }

  int e = 0;
  for (int h = 0; h < num_nodes; ++h) {
    Digraph::Node head = g.nodeFromId(h);

    const uint16_t * restrict tails_h = tails + out_degree * h;

    for (int t = 0; t < out_degree; ++t) {
      Digraph::Node tail = g.nodeFromId(tails_h[t]);

      ASSERT_EQ(e++, Digraph::id(g.addArc(head, tail)));
    }
  }
}

//----( network flow solver )-------------------------------------------------

class NetworkFlowSolverGuts
{
  enum { precision = 12, max_int = 1 << precision };

  const UniformDigraphGuts & m_graph;
  Solver m_solver;

  NodeMap m_supply;
  ArcMap m_flow;

public:

  NetworkFlowSolverGuts (
      const UniformDigraphGuts & graph,
      const Vector<float> & cost);

  void solve (const Vector<float> & supply, Vector<float> & flow);
};

NetworkFlowSolverGuts::NetworkFlowSolverGuts (
    const UniformDigraphGuts & graph,
    const Vector<float> & float_cost)
  : m_graph(graph),
    m_solver(graph.graph()),
    m_supply(graph.num_nodes()),
    m_flow(graph.num_arcs())
{
  ASSERT_SIZE(float_cost, graph.num_arcs());

  ArcMap int_cost(graph.num_arcs());

  const float max_cost = max(float_cost);
  const float scale = max_int / max_cost;

  const float * restrict float_c = float_cost;
  int * restrict int_c = int_cost.data();

  for (int e = 0, E = graph.num_arcs(); e < E; ++e) {
    int_c[e] = scale * float_c[e] + 0.5f;
  }

  m_solver.costMap(int_cost);
}

void NetworkFlowSolverGuts::solve (
    const Vector<float> & float_supply,
    Vector<float> & float_flow)
{
  ASSERT_SIZE(float_supply, m_graph.num_nodes());
  ASSERT_SIZE(float_flow, m_graph.num_arcs());

  NodeMap & int_supply = m_supply;
  ArcMap & int_flow = m_flow;

  // convert float -> int

  const float max_supply = max(float_supply);
  const float scale = max_int / max_supply;
  const float iscale = max_supply / max_int;

  const float * restrict float_s = float_supply;
  int * restrict int_s = int_supply.data();

  for (size_t n = 0, N = m_graph.num_nodes(); n < N; ++n) {
    int_s[n] = scale * float_s[n];
  }

  // run solver

  m_solver.supplyMap(m_supply);

  Solver::ProblemType type = m_solver.run();
  ASSERT(type = Solver::OPTIMAL, "failed to find a minimum cost flow");

  m_solver.flowMap(m_flow);

  // convert int -> flow

  const int * restrict int_f = int_flow.data();
  float * restrict float_f = float_flow;

  for (size_t e = 0, E = m_graph.num_arcs(); e < E; ++e) {
    float_f[e] = iscale * int_f[e];
  }
}

//----( opaque wrappers )-----------------------------------------------------

UniformDigraph::UniformDigraph (
    uint16_t num_nodes,
    uint16_t out_degree,
    const Vector<uint16_t> & tails)
  : m_guts(new UniformDigraphGuts(num_nodes, out_degree, tails))
{}

UniformDigraph::~UniformDigraph ()
{
  delete m_guts;
}

NetworkFlowSolver::NetworkFlowSolver (
    const UniformDigraph & graph,
    const Vector<float> & cost)
  : m_guts(new NetworkFlowSolverGuts(graph.guts(), cost))
{}

NetworkFlowSolver::~NetworkFlowSolver ()
{
  delete m_guts;
}

void NetworkFlowSolver::solve (
    const Vector<float> & supply,
    Vector<float> & flow)
{
  m_guts->solve(supply, flow);
}

