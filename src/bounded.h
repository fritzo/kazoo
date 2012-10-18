
#ifndef KAZOO_BOUNDED_H
#define KAZOO_BOUNDED_H

#include "common.h"

template<class Value>
struct BoundedSet
{
  const size_t capacity;
  size_t size;
  Value * const values;

  void operator= (const BoundedSet<Value> & other)
  {
    ASSERT_LE(other.size, capacity);
    size = other.size;
    for (size_t i = 0; i < size; ++i) {
      values[i] = other.values[i];
    }
  }

  BoundedSet (size_t cap)
    : capacity(cap),
      size(0),
      values(new Value[capacity])
  {}
  BoundedSet (const BoundedSet<Value> & other)
    : capacity(other.capacity),
      size(other.size),
      values(new Value[capacity])
  {
    * this = other;
  }
  ~BoundedSet ()
  {
    delete[] values;
  }

  void resize (size_t new_size)
  {
    ASSERT_LE(new_size, capacity);
    size = new_size;
  }
  Value & add ()
  {
    ASSERT_LT(size, capacity);
    return values[size++];
  }
  void add (const Value & value)
  {
    ASSERT_LT(size, capacity);
    values[size++] = value;
  }

  void clear () { size = 0; }
  bool empty () const { return size == 0; }
  bool full () const { return size == capacity; }
};

template<class Key, class Value>
struct BoundedMap
{
  const size_t capacity;
  size_t size;
  Key * const keys;
  Value * const values;

  void operator= (const BoundedMap<Key, Value> & other)
  {
    ASSERT_LE(other.size, capacity);
    size = other.size;
    for (size_t i = 0; i < size; ++i) {
      keys[i] = other.keys[i];
      values[i] = other.values[i];
    }
  }

  BoundedMap (size_t cap)
    : capacity(cap),
      size(0),
      keys(new Key[capacity]),
      values(new Value[capacity])
  {}
  BoundedMap (const BoundedMap<Key, Value> & other)
    : capacity(other.capacity),
      size(other.size),
      keys(new Key[capacity]),
      values(new Value[capacity])
  {
    * this = other;
  }
  ~BoundedMap ()
  {
    delete[] keys;
    delete[] values;
  }

  void resize (size_t new_size)
  {
    ASSERT_LE(new_size, capacity);
    size = new_size;
  }
  Value & add (const Key & key)
  {
    ASSERT_LT(size, capacity);
    keys[size] = key;
    return values[size++];
  }
  void add (const Key & key, const Value & value)
  {
    ASSERT_LT(size, capacity);
    keys[size] = key;
    values[size++] = value;
  }

  void clear () { size = 0; }
  bool empty () const { return size == 0; }
  bool full () const { return size == capacity; }
};

#endif // KAZOO_BOUNDED_H

