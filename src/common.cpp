
#include "common.h"
#include "config.h"
//#include <ctime>
#include <vector>
#include <cstdio>
#include <cstring>
#include <sys/time.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <errno.h>
#include <unistd.h>

const char * kazoo_logo =
"  _  __   _    ____  ___   ___\n"
" | |/ /  / \\  |_  / /   \\ /   \\  Video-Controlled Musical Instruments\n"
" |   (  / O \\  / /_|  O  |  O  |  copyright 2009-2012 Fritz Obermeyer\n"
" |_|\\_\\/__n__\\/____|\\___/ \\___/";

void chdir_kazoo ()
{
  const char * KAZOO = getenv("KAZOO");
  if (KAZOO) {
    ASSERTW(not chdir(KAZOO), "directory $KAZOO does not exist");
  } else {
    const char * HOME = getenv("HOME");
    if (HOME) {
      string KAZOO = string(HOME) + string("/kazoo");
      ASSERTW(not chdir(KAZOO.c_str()), "directory $HOME/kazoo does not exist");
    } else {
      WARN("HOME is not in environment");
    }
  }
}

//----( memory tools )--------------------------------------------------------

//TODO use ifdefs to use _aligned_malloc and _aligned_free in Visual C++

void * malloc_aligned (size_t size, size_t alignment)
{
  void * result;
  int info = posix_memalign(&result, alignment, size);

  switch (info) {
    case 0: break;

    case ENOMEM:
      ERROR("malloc_aligned(" << size << ", " << alignment << ") failed"
          " due to lack of memory");
      break;

    case EINVAL:
      WARN("malloc_aligned(" << size << ", " << alignment << ") failed"
          " to align memory -- cross your fingers");
      break;

    default:
      ERROR("malloc_aligned(" << size << ", " << alignment << ") failed"
          " for unkown reason.\n\terror code = " << info);
  }

#ifdef CHECK_ALIGNMENT
  size_t offset = reinterpret_cast<size_t>(result) % alignment;
  ASSERT(offset == 0,
         "malloc_aligned(..., " << alignment << ") had offset " << offset);
#endif // CHECK_ALIGNMENT

  return result;
}

void free_aligned (void * pointer) { free(pointer); }

void copy_float (
    const float * restrict source,
    float * restrict dest,
    size_t size)
{
  memcpy(dest, source, size * sizeof(float));
}
void copy_complex (
    const complex * restrict source,
    complex * restrict dest,
    size_t size)
{
  memcpy(dest, source, size * sizeof(complex));
}

void zero_float (float * x, size_t size)
{
  bzero(x, size * sizeof(float));
}
void zero_complex (complex * x, size_t size)
{
  bzero(x, size * sizeof(complex));
}
void zero_bytes (void * x, size_t size)
{
  bzero(x, size);
}

void print_float (const float * data, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    cout << data[i] << "\n";
  }
  cout << endl;
}
void print_complex (const complex * data, size_t size)
{
  for (size_t i = 0; i < size; ++i) {
    cout << data[i] << "\n";
  }
  cout << endl;
}

//----( math )----------------------------------------------------------------

int log2i (int x)
{
  ASSERT_LT(0, x);

  int result = 0;
  for (x >>= 1; x; x >>= 1) ++result;
  return result;
}

//----( random generators )---------------------------------------------------

bool random_bit ()
{
    static size_t buffer = 0, mask = 0;
    mask >>= 1;
    if (not mask) {
        buffer = lrand48();
        mask = 1<<30;
    }
    return buffer & mask;
}

unsigned random_poisson (float mean)
{
  if (not (mean > 0)) return 0;

  unsigned result = 0;
  while (mean > 0) {
    mean += logf(random_01());
    ++result;
  }
  return result - 1;
}

//----( time )----------------------------------------------------------------

// time measurement
timeval g_begin_time, g_current_time;
const int g_time_is_available(gettimeofday(&g_begin_time, NULL));
inline void update_time () { gettimeofday(&g_current_time, NULL); }
double get_elapsed_time ()
{
  update_time();
  return g_current_time.tv_sec - g_begin_time.tv_sec
    + 1e-6 * (g_current_time.tv_usec - g_begin_time.tv_usec);
}

string get_date (bool hour)
{
  const size_t size = 20; // fits e.g. 2007-05-17-11-33
  static char buff[size];

  time_t t = time(NULL);
  tm T;
  gmtime_r (&t,&T);
  if (hour) strftime(buff,size, "%Y-%m-%d-%H-%M", &T);
  else      strftime(buff,size, "%Y-%m-%d", &T);
  return buff;
}

//----( unix magic )----------------------------------------------------------

void daemonize (const char * logfilename)
{
  // fork from parent
  if (fork()) exit(0);

  // detach from parent's environment
  umask(0);
  ASSERT(setsid() >= 0, "failed to set sid");
  ASSERT(chdir("/") >= 0, "failed to chdir to /");

  // redirect streams
  FILE * logfile = fopen(logfilename, "a");
  ASSERT(logfile, "failed to open log file " << logfilename);
  stdout = stderr = logfile;
}

/*
#ifdef __GNUG__

#include <execinfo.h>

ostream & indented_cout ()
{
  // from http://tombarta.wordpress.com/2008/08/01/c-stack-traces-with-gcc/
  // alternatively, see http://stackoverflow.com/questions/582673/is-there-a-cheaper-way-to-find-the-depth-of-the-call-stack-than-using-backtrace

  const size_t max_spaces = 32;
  static const char * spaces = "                                ";
  static void * buffer[max_spaces];

  size_t stack_depth = backtrace(buffer, max_spaces);
  return cout << spaces[max_spaces - stack_depth];
}

#endif // __GNUG__
*/

/*
// from "So you want a stand-alone function that prints a stack trace"
// http://stackoverflow.com/questions/4636456/stack-trace-for-c-using-gcc/4732119#4732119

#include <stdio.h>
#include <stdlib.h>
#include <sys/wait.h>
#include <unistd.h>

void print_stack_trace ()
{
  char pid_buf[30];
  sprintf(pid_buf, "%d", getpid());

  char name_buf[512];
  name_buf[readlink("/proc/self/exe", name_buf, 511)]=0;

  int child_pid = fork();
  if (!child_pid) {

  dup2(2,1); // redirect output to stderr
  fprintf(stdout,"stack trace for %s pid=%s\n",name_buf,pid_buf);
  execlp(
      "gdb", "gdb", "--batch", "-n",
      "-ex", "thread",
      "-ex", "bt",
      name_buf, pid_buf, NULL);
  abort(); // If gdb failed to start

  } else {

    waitpid(child_pid,NULL,0);
  }
}
*/

