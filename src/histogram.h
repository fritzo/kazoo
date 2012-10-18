#ifndef KAZOO_HISTOGRAM_H
#define KAZOO_HISTOGRAM_H

#include "common.h"
#include "hash_map.h"

class Histogram
{
public:

  typedef int key_type;
  struct data_type
  {
    double mass;
    data_type () : mass(0) {}
    data_type (double c) : mass(c) {}
    operator double () const { return mass; }
    operator double & () { return mass; }
  };
  typedef std::pair<key_type, data_type> value_type;

  typedef std::hash_map<key_type, data_type> Bins;
  typedef Bins::const_iterator iterator;

protected:

  const double m_binwidth;
  Bins m_bins;

public:

  Histogram (double binwidth = 1) : m_binwidth(binwidth) {}

  void add (double x) { m_bins[roundi(x / m_binwidth)] += 1; }
  void add (double x, double mass)
  {
    if (mass > 0) m_bins[roundi(x / m_binwidth)] += mass;
  }

  void normalize ();

  double binwidth () const { return m_binwidth; }
  const Bins & bins () const { return m_bins; }
  size_t size () const { return m_bins.size(); }

  iterator begin () const { return m_bins.begin(); }
  iterator end () const { return m_bins.end(); }
};

ostream & operator<< (ostream & o, const Histogram & h);

#endif // KAZOO_HISTOGRAM_H
