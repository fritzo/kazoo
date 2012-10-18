#ifndef KAZOO_NETWORK_FLOW_H
#define KAZOO_NETWORK_FLOW_H

#include "common.h"
#include "vectors.h"

//----( uniform digraph )-----------------------------------------------------

class UniformDigraphGuts;
class UniformDigraph
{
  UniformDigraphGuts * m_guts;

public:

  UniformDigraph (
      uint16_t num_nodes,
      uint16_t out_degree,
      const Vector<uint16_t> & tails);
  ~UniformDigraph ();

  const UniformDigraphGuts & guts () const { return * m_guts; }
};

//----( network flow solver )-------------------------------------------------

class NetworkFlowSolverGuts;
class NetworkFlowSolver
{
  NetworkFlowSolverGuts * m_guts;

public:

  NetworkFlowSolver (
      const UniformDigraph & graph,
      const Vector<float> & cost);
  ~NetworkFlowSolver ();

  void solve (const Vector<float> & supply, Vector<float> & flow);
};

#endif // KAZOO_NETWORK_FLOW_H
