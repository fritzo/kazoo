
#include "histogram.h"
#include <vector>
#include <algorithm>

void Histogram::normalize ()
{
  double total = 0;
  for (Bins::const_iterator i = begin(), I = end(); i != I; ++i) {
    total += i->second;
  }

  ASSERT_LT(0, total);
  const double scale = 1.0 / total;
  for (Bins::iterator i = m_bins.begin(), I = m_bins.end(); i != I; ++i) {
    i->second *= scale;
  }
}

ostream & operator<< (ostream & o, const Histogram & h)
{
  std::vector<Histogram::value_type> vector(h.begin(), h.end());
  std::sort(vector.begin(), vector.end());

  o << "[\n";
  for (auto i = vector.begin(); i != vector.end(); ++i) {
    o << " (" << (h.binwidth() * i->first) << ", " << i->second << "),\n";
  }
  return o << "]\n";
}

