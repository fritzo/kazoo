#ifndef KAZOO_MATCHING_H
#define KAZOO_MATCHING_H

#include "common.h"
#include "hash_map.h"
#include <vector>
#include <set>

namespace Matching
{

const float TOL = 1e-6f;
const float BIG = 50;

typedef std::vector<Id> Ids;
typedef std::set<Id> IdSet;
typedef std::hash_map<Id, Id> IdMap;

struct Arc
{
  Id i,j;
  Arc () {}
  Arc (Id a_i, Id a_j) : i(a_i), j(a_j) {}
};
typedef std::vector<Arc> Arcs;

typedef std::vector<float> Probs;
typedef std::vector<float> Costs;
typedef std::vector<bool> Bools;

/** Optimal values (less is better).

  To determine cost relative to an optimum,
  we need to store the two best costs.
*/
struct Optimum : public std::pair<float,float>
{
  operator float () const { return first; }

  void init () { first = second = INFINITY; }
  void init (float x) { first = x; second = INFINITY; }
  void update (float x)
  {
    if (x < second) {
      if (x < first) {
        second = first;
        first = x;
      } else {
        second = x;
      }
    }
  }

  float best_alternative (float x) { return x == first ? second : first; }
};
typedef std::vector<Optimum> Optima;

//----( soft assignment solver )----------------------------------------------

/** Sparse 2D Soft Assignment Solver.

  Finds probabilistic matching among two sets of objects,
  allowing for non-association.

  On Input:
    prior_1_non      Prior nonassociation likelihood ratio
    prior_2_non      Prior nonassociation likelihood ratio
    prior_ass        Prior association likelihood ratio
                     0 or 1 entry per pair in set1 x set2

  On Output:
    post_1_non      Posterior nonassociation probability
    post_2_non      Posterior nonassociation probability
    post_1_ass      Posterior association probability,
                    0 or 1 entry per pair in set1 x set2
*/

class SoftMatching
{
  Arcs m_arcs;

  Probs m_prior_1_non;
  Probs m_prior_2_non;
  Probs m_prior_1_ass;
  Probs m_prior_2_ass;

  Probs m_post_1_non;
  Probs m_post_2_non;
  Probs m_post_1_ass;
  Probs m_post_2_ass;

  Probs m_scale_1;
  Probs m_scale_2;
  Probs m_total;
  Probs m_message;

public:

  SoftMatching () {}
  ~SoftMatching () {}

  //----( input interface )----

  void clear ();
  void add1 (float like = 1) { m_prior_1_non.push_back(like); }
  void add2 (float like = 1) { m_prior_2_non.push_back(like); }
  void add12 (Id i, Id j, float like = 1)
  {
    m_arcs.push_back(Arc(i,j));
    m_prior_1_ass.push_back(like);
    m_prior_2_ass.push_back(like);
  }

  //----( solving )----

  void validate_problem () const;
  void solve (size_t num_iters = 6);
  void validate_solution ();

  //----( output interface )----

  size_t size_1 () const { return m_prior_1_non.size(); }
  size_t size_2 () const { return m_prior_2_non.size(); }
  size_t size_arc () const { return m_prior_1_ass.size(); }

  const Arc & arc (size_t i) const { return m_arcs[i]; }

  // binary likelihood ratios
  float prior_1_non (size_t i) const { return m_prior_1_non[i]; }
  float prior_2_non (size_t j) const { return m_prior_2_non[j]; }
  float prior_1_ass (size_t ij) const { return m_prior_1_ass[ij]; }
  float prior_2_ass (size_t ij) const { return m_prior_2_ass[ij]; }
  float prior_ass   (size_t ij) const { return m_prior_1_ass[ij]; }

  // likelihoods among sets
  float post_1_non (size_t i) const { return m_post_1_non[i]; }
  float post_2_non (size_t j) const { return m_post_2_non[j]; }
  float post_1_ass (size_t ij) const { return m_post_1_ass[ij]; }
  float post_2_ass (size_t ij) const { return m_post_2_ass[ij]; }
  float post_ass   (size_t ij) const { return m_post_1_ass[ij]; }

  // debugging
  void print_prior () const;
  void print_post () const;

protected:

  void total_1 ();
  void total_2 ();

  void propagate_12 ();
  void propagate_21 ();

  void normalize ();
};

//----( hard assignment solver )----------------------------------------------

/** Sparse 2D Hard Assignment Solver.

  Finds minimum-cost matching among two sets of objects,
  allowing for non-association.

  On Input:
    prior_1_non      Prior nonassociation relative cost
    prior_2_non      Prior nonassociation relative cost
    prior_ass        Prior association relative cost
                     0 or 1 entry per pair in set1 x set2

  On Output:
*/

class HardMatching
{
  Arcs m_arcs;

  Costs m_prior_1_non;
  Costs m_prior_2_non;
  Costs m_prior_ass;

  Costs m_message_12;
  Costs m_message_21;
  Costs m_post_ass;

  Optima m_optima;

  Bools m_1_non;
  Bools m_2_non;
  Bools m_1_ass;
  Bools m_2_ass;

public:

  HardMatching () {}
  ~HardMatching () {}

  //----( input interface )----

  void clear ();
  void add1 (float cost = 0) { m_prior_1_non.push_back(cost); }
  void add2 (float cost = 0) { m_prior_2_non.push_back(cost); }
  void add12 (Id i, Id j, float cost = 0)
  {
    m_arcs.push_back(Arc(i,j));
    m_prior_ass.push_back(cost);
  }

  //----( solving )----

  void validate_problem () const;
  void solve (size_t num_iters = 20);
  void validate_solution ();

  //----( output interface )----

  size_t size_1 () const { return m_prior_1_non.size(); }
  size_t size_2 () const { return m_prior_2_non.size(); }
  size_t size_arc () const { return m_arcs.size(); }

  const Arc & arc (size_t ij) const { return m_arcs[ij]; }

  // binary relative costs
  float prior_1_non (size_t i)  const { return m_prior_1_non[i]; }
  float prior_2_non (size_t j)  const { return m_prior_2_non[j]; }
  float prior_ass   (size_t ij) const { return m_prior_ass[ij]; }

  // boolean decisions
  bool post_1_non (size_t i)  const { return m_1_non[i]; }
  bool post_2_non (size_t j)  const { return m_2_non[j]; }
  bool post_1_ass (size_t ij) const { return m_1_ass[ij]; }
  bool post_2_ass (size_t ij) const { return m_2_ass[ij]; }
  bool post_ass   (size_t ij) const { return m_1_ass[ij]; }

  // debugging
  void print_prior () const;
  void print_post () const;

protected:

  void optimize_1 ();
  void optimize_2 ();

  void propagate_12 ();
  void propagate_21 ();

  void normalize ();
};

} // namespace matching

#endif // KAZOO_MATCHING_H
