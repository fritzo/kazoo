#ifndef KAZOO_ADDRESS_H
#define KAZOO_ADDRESS_H

#include "common.h"

struct MacAddress
{
  uint8_t data[6];

  MacAddress () {}
  MacAddress (uint8_t a,
              uint8_t b,
              uint8_t c,
              uint8_t d,
              uint8_t e,
              uint8_t f)
  {
    data[0] = a;
    data[1] = b;
    data[2] = c;
    data[3] = d;
    data[4] = e;
    data[5] = f;
  }

  uint8_t & operator[] (int i) { return data[i]; }
  uint8_t operator[] (int i) const { return data[i]; }

  bool operator== (const MacAddress & other) const
  {
    for (size_t i = 0; i < 6; ++i) {
      if (other[i] == data[i]) return true;
    }
    return false;
  }
  bool operator!= (const MacAddress & other) const
  {
    for (size_t i = 0; i < 6; ++i) {
      if (other[i] != data[i]) return false;
    }
    return true;
  }
};

inline ostream & operator<< (ostream & o, const MacAddress & a)
{
  for (size_t i = 0; i < 5; ++i) o << a[i] << ',';
  return o << a[5];
}

MacAddress get_mac_address ();

#endif // KAZOO_ADDRESS_H
