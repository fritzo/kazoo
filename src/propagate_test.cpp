
#include "propagate.h"
#include "array.h"
#include <vector>

using namespace Propagate;

//----( tracking problem )----------------------------------------------------

class TrackingProblem
{
  struct State
  {
    float x,y;

    void simulate ()
    {
      x = random_01();
      y = random_01();
    }
    void advance (float dt)
    {
      float std_dev = sqrtf(dt);
      x = fmodf(x + 1 + random_01() * std_dev, 1);
      y = fmodf(y + 1 + random_01() * std_dev, 1);
    }
  };
  typedef std::vector<State> States;

  struct Detection
  {
    Id id;
    State state;
  };
  typedef std::vector<Detection> Frame;
  typedef std::vector<Frame> Frames;

  struct Arc
  {
    Id head, tail;
    float energy;
  };
  typedef std::vector<Arc> Arcs;
  typedef std::vector<Arcs> Arcss;

  Id m_id_generator;
  Frames m_frames;
  Arcss m_arcss;

  Id new_id () { return m_id_generator++; }

public:

  void clear ()
  {
    m_id_generator = 0;
    m_frames.clear();
    m_arcss.clear();
  }

  void simulate (
    size_t num_frames,
    float expected_num_objects,
    float expected_lifetime = 100.0,
    float mixing_time = 100.0,
    float prob_md = 0.5,
    float prob_fa = 0.5);

  void gate ();

  void save (const char * filename);
  void load (const char * filename);
};

void TrackingProblem::simulate (
    size_t num_frames,
    float expected_num_objects,
    float expected_lifetime,
    float mixing_time,
    float prob_md,
    float prob_fa)
{
  float expected_num_fas = prob_fa / (1 - prob_fa)
                        * prob_md
                        * expected_num_objects;
  float expected_births = expected_num_objects / expected_lifetime;
  float prob_death = 1 / expected_lifetime;
  float process_noise = 1 / mixing_time;

  // initialize states
  size_t init_num_objects = random_poisson(expected_num_objects);
  States states(init_num_objects), new_states;
  for (size_t i = 0; i < states.size(); ++i) {
    states[i].simulate();
  }

  // simulate frames one at a time
  clear();
  m_frames.resize(num_frames);
  for (size_t t = 0; t < num_frames; ++t) {

    // advance objects
    new_states.resize(random_poisson(expected_births));
    for (size_t i = 0; i < new_states.size(); ++i) {
      new_states[i].simulate();
    }
    for (size_t i = 0; i < states.size(); ++i) {
      if (not random_bernoulli(prob_death)) {
        states[i].advance(process_noise);
        new_states.push_back(states[i]);
      }
    }
    std::swap(states, new_states);

    // observe objects
    Frame & frame = m_frames[t];
    frame.resize(random_poisson(expected_num_fas));
    for (size_t i = 0; i < frame.size(); ++i) {
      frame[i].id = new_id();
      frame[i].state.simulate();
    }
    for (size_t i = 0; i < states.size(); ++i) {
      if (not random_bernoulli(prob_md)) {
        frame.resize(frame.size() + 1);
        frame.back().id = new_id();
        frame.back().state = states[i];
      }
    }
  }
}

void TrackingProblem::gate ()
{
  TODO("do something reasonable here");
}

void TrackingProblem::save (const char * filename) { TODO("save problem"); }
void TrackingProblem::load (const char * filename) { TODO("load problem"); }

//----( testing )-------------------------------------------------------------

/*
void test_mfa (size_t num_iters = 10)
{
  MFAProblem<8> problem;

  TODO("test multi-frame assignment");

  problem.solve(num_iters);
}
*/

void test_factored_mfa (size_t num_iters = 10)
{
  FactoredMFAProblem<8> problem;

  TODO("test factored multi-frame assignment");

  problem.solve(num_iters);
}

void test_grid_matching (size_t num_iters = 10)
{
  GridMatchingProblem<1> problem;

  TODO("test grid matching");

  problem.solve(num_iters);
}

int main ()
{
  //test_mfa();
  test_factored_mfa();
  test_grid_matching();

  return 0;
}

