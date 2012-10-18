#!/usr/bin/python

"""setup.py
Description:
  Builds transforms module for analysis and synthesis.
  Installs _kazoo and kazoo.

Usage:
  python setup_kazoo.py build
  - compiles the library from source
    (requires compiler version appropriate to this Python build;
    run with the python executable which will be used to run this extension)
  python setup_kazoo.py install
  - installs the library for use (building it first, if needed)

Required Libraries:
  FFTW3         - GPL/commercial - http://www.fftw.org
  PortAudio v19 - plain MIT      - http://www.portaudio.com
  SDL 1.2       - GNU LGPL       - http://www.libsdl.org
  LAPACK/BLAS   - Berkeley style - http://www.netlib.org/lapack
"""

# see http://docs.python.org/extending/

from distutils.core import setup, Extension

for line in file('src/__init__.py').readlines():
  if line.startswith('__version__'):
    exec(line.strip())

kazoo_module = Extension(
    '_kazoo',
    language = 'c++',
    sources = [
      'src/_kazoo.cpp',
      'src/common.cpp',
      'src/fft.cpp',
      'src/vectors.cpp',
      'src/threads.cpp',
      'src/splines.cpp',
      'src/spectrogram.cpp',
      'src/reassign.cpp',
      'src/synchrony.cpp',
      'src/transforms.cpp',
      'src/psycho.cpp',
      'src/audio.cpp',
      'src/animate.cpp',
      'src/images.cpp',
      'src/events.cpp',
      ],
    depends = [
      'src/__init__.py',
      'src/_kazoo.h',
      'src/common.h',
      'src/fft.h',
      'src/vectors.h',
      'src/window.h',
      'src/threads.h',
      'src/splines.h',
      'src/spectrogram.h',
      'src/reassign.h',
      'src/synchrony.h',
      'src/transforms.h',
      'src/psycho.h',
      'src/audio.h',
      'src/animate.h',
      'src/images.h',
      'src/array.h',
      'src/sym33.h',
      'src/events.h',
      ],
    library_dirs = [ '/usr/local/lib' ],
    libraries = [
      'm',
      #'blas',
      #'lapack',
      'fftw3f',
      'tbb',
      'portaudio',
      'SDL',
      ],
    define_macros = [
      #('KAZOO_NDEBUG', None),
      ('KAZOO_IMAGES_COLOR', None),
      ],
    extra_compile_args = [
      '-std=c++0x',
      '-Wall',
      #'-Wextra',
      '-Winit-self',
      '-Wno-write-strings',
      #'-Winline',
      #'-Wdisable-optimization',
      '-ggdb',
      '-O3',
      '-pipe',
      '-march=native',
      #'-fno-exceptions',
      #'-fno-rtti',
      '-ffast-math',
      '-funswitch-loops',
      #'-malign-double',
      #'-fomit-frame-pointer',
      #'-funsafe-math-optimizations',
      #'-fsingle-precision-constant',
      #'-m32',
      #'-mmmx',
      #'-msse',
      #'-msse2',
      #'-msse3',
      #'-mfpmath=sse',
      ],
    extra_link_args = [
      #'-rdynamic',
      ],

    )

setup(
    name = 'kazoo',
    version = __version__,
    url = 'http://obermeyer.svnrepository.com/svn/kazoo',
    description = 'Streaming audio transforms',
    packages = [ 'kazoo', 'kazoo.example' ],
    package_dir = {'kazoo' : 'src'},
    ext_modules = [kazoo_module],
    )

