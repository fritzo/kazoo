
#ifndef KAZOO_HASH_MAP_H
#define KAZOO_HASH_MAP_H

//----( gnu c++0x support )---------------------------------------------------

#if defined(__GXX_EXPERIMENTAL_CXX0X__)

#include <unordered_map>
#define hash_map unordered_map

//----( gnu extension )-------------------------------------------------------

#elif defined(__GNUG__)

#include <tr1/unordered_map>
#define hash_map unordered_map
namespace std { using namespace tr1; }

//----( MS visual studio extension )------------------------------------------

#elif defined(_MSC_VER)

#include <hash_map>

//----( use red-black tree instead of hash map )------------------------------

#else

#include <map>
#define hash_map map

#endif

#endif // KAZOO_HASH_MAP_H
