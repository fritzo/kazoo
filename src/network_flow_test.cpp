
#include "network_flow.h"
#include "args.h"

size_t next (size_t x, size_t X) { return (x + 1) % X; }
size_t prev (size_t x, size_t X) { return (x + X - 1) % X; }

void test_torus (Args & args)
{
  int width = args.pop(4);
  int height = args.pop(5);

  LOG("testing flow on a " << width << " x " << height << " torus");

  int num_nodes = width * height;
  int out_degree = 4;
  int num_arcs = num_nodes * out_degree;

  const size_t N = num_nodes;
  const size_t X = width;
  const size_t Y = height;

  LOG("building a digraph");

  Vector<uint16_t> tails(num_arcs);

  for (size_t x = 0; x < X; ++x) {
    for (size_t y = 0; y < Y; ++y) {
      size_t xy = Y * x + y;
      tails[4 * xy + 0] = Y * next(x,X) + y;
      tails[4 * xy + 1] = Y * prev(x,X) + y;
      tails[4 * xy + 2] = Y * x + next(y,Y);
      tails[4 * xy + 3] = Y * x + prev(y,Y);
    }
  }

  UniformDigraph graph(num_nodes, out_degree, tails);

  LOG("creating a solver");

  Vector<float> cost(num_arcs);
  cost.set(1.0f);

  NetworkFlowSolver solver(graph, cost);

  Vector<float> supply(num_nodes);
  for (size_t n = 0; n < N; ++n) {
    supply[n] = random_std();
  }

  LOG("solving");

  Vector<float> flow(num_arcs);
  solver.solve(supply, flow);
}

const char * help_message =
"Usage: network_flow_test COMMAND"
"\nCommands:"
"\n  torus"
;

int main (int argc, char ** argv)
{
  Args args(argc, argv, help_message);

  args
    .case_("torus", test_torus)
    .default_(test_torus);

  return 0;
}

