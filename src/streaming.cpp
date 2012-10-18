
#include "streaming.h"
#include "events.h"
#include <set>

namespace Streaming
{

//----( threaded systems )----------------------------------------------------

static std::set<Thread *> g_all_threads;

Thread::Thread () : m_running(false) { g_all_threads.insert(this); }

Thread::~Thread () { g_all_threads.erase(this); }

void Thread::start ()
{
  ASSERT(not m_running, "tried to start thread twice");
  m_running = true;
  ::Thread::start();
}

void Thread::stop ()
{
  ASSERT(m_running, "tried to stop thread twice");
  m_running = false;
}

void TimedThread::run ()
{
  const float dt = 1.0f / m_rate;
  int sleep_usec = int(1e6 * floor(dt / 10));

  // if sleeping for less than a timer tick, we might as well spin
  if (sleep_usec < 1000) sleep_usec = 0;

  Seconds time = Seconds::now();
  while (m_running) {
    if (Seconds::now() > time) {
      step();
      time += max(dt, Seconds::now() - time - dt);
    } else if (sleep_usec) {
      usleep(sleep_usec);
    }
  }
}

bool g_screen_exists = false;

void run ()
{
  LOG("starting " << g_all_threads.size() << " threads");
  typedef std::set<Thread *>::iterator Auto;
  for (Auto i = g_all_threads.begin(); i != g_all_threads.end(); ++i) {
    (*i)->start();
  }

  if (g_screen_exists) {

    LOG("waiting at screen");
    sdl_event_loop();

  } else {

    LOG("waiting at terminal");
    getchar();
  }

  LOG("stopping " << g_all_threads.size() << " threads");
  for (Auto i = g_all_threads.begin(); i != g_all_threads.end(); ++i) {
    (*i)->stop();
  }

  for (Auto i = g_all_threads.begin(); i != g_all_threads.end(); ++i) {
    (*i)->wait();
  }

  usleep(200000); // HACK give stragglers a chance to die
}

//----( basic data types )----------------------------------------------------

const float GlovesImage::num_channels = 1.5f;
const float Yuv420p8Image::num_channels = 1.5f;

} // namespace Streaming

