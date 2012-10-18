
#ifndef KAZOO_THREADS_H
#define KAZOO_THREADS_H

/** A mish-mash of threading wrappers.

  We use:
  * Intel's ttb::spin_mutex for spin mutexes, waiting for it to be built-in.
  * SDL's SDL_Thread for threading, because the built-int std::thread
    requires rtti, and we want to compile with -fno-rtti
  * Mac OS X does not support timed threads, so <mutex> and <thread> fail.
*/

#include "common.h"

//#define USE_STL_THREAD_WHICH_REQUIRES_RTTI

//#define KAZOO_USING_TBB

//----( mutexes )-------------------------------------------------------------

#ifdef __GXX_EXPERIMENTAL_CXX0X__

#include <mutex>
#include <condition_variable>

typedef std::mutex Mutex;
typedef std::unique_lock<std::mutex> UniqueLock;
typedef std::condition_variable ConditionVariable;
typedef std::mutex BinarySemaphore;

#else // __GXX_EXPERIMENTAL_CXX0X__

#include <SDL/SDL_mutex.h>

class Mutex;
class UniqueLock;
class ConditionVariable;
class BinarySemaphore;

class Mutex
{
  friend class UniqueLock;

  SDL_mutex * m_mutex;

public:

  Mutex () : m_mutex(SDL_CreateMutex()) {}
  ~Mutex () { SDL_DestroyMutex(m_mutex); }

  void lock () { SDL_LockMutex(m_mutex); }
  void unlock () { SDL_UnlockMutex(m_mutex); }
};

class UniqueLock
{
  friend class ConditionVariable;

  SDL_mutex * m_mutex;

public:

  UniqueLock (Mutex & mutex) : m_mutex(mutex.m_mutex)
  {
    SDL_LockMutex(m_mutex);
  }

  void lock () { SDL_LockMutex(m_mutex); }
  void unlock () { SDL_UnlockMutex(m_mutex); }
};

class ConditionVariable
{
  SDL_cond * m_cond;

public:

  ConditionVariable () : m_cond(SDL_CreateCond()) {}
  ~ConditionVariable () { SDL_DestroyCond(m_cond); }

  void wait (UniqueLock & lock) { SDL_CondWait(m_cond, lock.m_mutex); }
  void notify_all () { SDL_CondBroadcast(m_cond); }
};

class BinarySemaphore
{
  SDL_sem * m_sem;

public:

  BinarySemaphore () : m_sem(SDL_CreateSemaphore(1)) {}
  ~BinarySemaphore () { SDL_DestroySemaphore(m_sem); }

  void lock () { SDL_SemWait(m_sem); }
  bool try_lock () { return not SDL_SemTryWait(m_sem); }
  void unlock () { SDL_SemPost(m_sem); }
};

#endif // __GXX_EXPERIMENTAL_CXX0X__

#ifdef KAZOO_USING_TBB

#include <tbb/spin_mutex.h>
typedef tbb::spin_mutex SpinMutex;

#endif // KAZOO_USING_TBB

//----( threads )-------------------------------------------------------------

#ifdef USE_STL_THREAD_WHICH_REQUIRES_RTTI

#include <thread>

class Thread
{
  std::thread * m_thread;

public:

  Thread () : m_thread(NULL) {}

  Thread & start ();
  Thread & wait ();

  void operator () () { run(); }

// morally protected
  virtual void run () = 0;
};

#else // USE_STL_THREAD_WHICH_REQUIRES_RTTI

#include <SDL/SDL_thread.h>

class Thread
{
  SDL_Thread * m_thread;

public:

  Thread () : m_thread(NULL) {}

  Thread & start ();
  Thread & wait ();

// morally protected
  virtual void run () = 0;
};

#endif // USE_STL_THREAD_WHICH_REQUIRES_RTTI

#endif // KAZOO_THREADS_H
