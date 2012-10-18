
#include "threads.h"

//----( threads )-------------------------------------------------------------

#ifdef USE_STL_THREAD_WHICH_REQUIRES_RTTI

Thread & Thread::start ()
{
  ASSERT(not m_thread, "tried to start a thread twice");
  m_thread = new std::thread(std::ref(*this));
  return * this;
}

Thread & Thread::wait ()
{
  ASSERT(m_thread, "tried to stop a thread twice");
  m_thread->join();
  m_thread = NULL;
  return * this;
}

#else // USE_STL_THREAD_WHICH_REQUIRES_RTTI

namespace
{
extern "C" int Thread_run (void * t)
{
  ((Thread *) t)->run();
  return 0;
}
}

Thread & Thread::start ()
{
  ASSERT(not m_thread, "tried to start a thread twice");
  m_thread = SDL_CreateThread(Thread_run, this);
  return * this;
}

Thread & Thread::wait ()
{
  ASSERT(m_thread, "tried to stop a thread twice");
  SDL_WaitThread(m_thread, NULL);
  m_thread = NULL;
  return * this;
}

#endif // USE_STL_THREAD_WHICH_REQUIRES_RTTI

