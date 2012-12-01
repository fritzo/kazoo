#ifndef KAZOO_ALIGNED_ALLOCATOR_H
#define KAZOO_ALIGNED_ALLOCATOR_H

#include <cstdint>
#include <cstdlib>

#ifdef USE_EXCEPTIONS
#include <new>
#endif // USE_EXCEPTIONS

namespace nonstd
{

template<class T, int alignment = 32>
class aligned_allocator
{
public:

  typedef T value_type;
  typedef size_t size_type;
  typedef std::ptrdiff_t difference_type;

  typedef T * pointer;
  typedef const T * const_pointer;

  typedef T & reference;
  typedef const T & const_reference;

  template <class U>
  aligned_allocator(const aligned_allocator<U, alignment> &) throw() {}
  aligned_allocator (const aligned_allocator &) throw() {}
  aligned_allocator () throw() {}
  ~aligned_allocator () throw() {}

  /*
private:
  void operator= (const aligned_allocator<T> &); // undefined
public:
  */

  template<class U>
  struct rebind
  {
    typedef aligned_allocator<U, alignment> other;
  };

  pointer address (reference r) const
  {
    return & r;
  }

  const_pointer address (const_reference r) const
  {
    return & r;
  }

  pointer allocate (size_t n, const void * /* hint */ = 0)
  {
    void * result = NULL;
    if (posix_memalign(& result, alignment, n * sizeof(T))) {
#ifdef USE_EXCEPTIONS
      throw std::bad_alloc();
#endif // USE_EXCEPTIONS
    }
    return static_cast<pointer>(result);
  }

  void deallocate (pointer p, size_type /* count */ )
  {
    free(p);
  }

  void construct (pointer p, const T & val)
  {
    new (p) T(val);
  }

  void destroy (pointer p)
  {
    p->~T();
  }

  size_type max_size () const throw()
  {
    return SIZE_MAX / sizeof(T);
  }
};

template<class T1, class T2>
inline bool operator== (
    const aligned_allocator<T1> &,
    const aligned_allocator<T2> &) throw()
{
  return true;
}

template<class T1, class T2>
inline bool operator!= (
    const aligned_allocator<T1> &,
    const aligned_allocator<T2> &) throw()
{
  return false;
}

} // namespace nonstd

#endif // KAZOO_ALIGNED_ALLOCATOR_H
