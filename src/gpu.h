#ifndef KAZOO_GPU_H
#define KAZOO_GPU_H

#include "common.h"

namespace Gpu
{

//----( general info )--------------------------------------------------------

bool using_cuda ();
void set_using_cuda (bool whether);

int cuda_device_count ();

void print_gpu_info ();

//----( memory )--------------------------------------------------------------

void * cuda_malloc (size_t size);
void cuda_free (void * data);
void cuda_memcpy_h2d (void * dst, const void * src, size_t size);
void cuda_memcpy_d2h (void * dst, const void * src, size_t size);
void cuda_bzero (void * dest, size_t size);

//----( cuda timer )----------------------------------------------------------

class Timer
{
  float m_time;

public:

  Timer () : m_time(get_elapsed_time()) {}

  void tick () { m_time = get_elapsed_time(); }
  float tock () const { return get_elapsed_time() - m_time; }
};

} // namespace Gpu

#endif // KAZOO_GPU_H
