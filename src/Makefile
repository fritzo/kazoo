
current_target: all

all: test kazoo learn toy player

#----( params )----------------------------------------------------------------

PLATFORM := $(shell uname)

COMPILE_TYPE = debug
#COMPILE_TYPE = release
#COMPILE_TYPE = profile
#COMPILE_TYPE = debugcuda
#COMPILE_TYPE = slowdebug

#----( compile flags )---------------------------------------------------------

LD_LIB_PATH = /usr/local/lib
OPENCV = -lcxcore -lcv -lhighgui
V4L2 = -lv4l2
JACK = -ljack
STK = -lstk
FFTW3F =  -lfftw3f
LEMON = -lemon
PORTAUDIO = -lportaudio
LIBSDL = `sdl-config --libs`
PNG = -lpng
FFMPEG = -lswscale -lavformat -lavcodec -lavutil
CUDA = -lcudart -lcusparse -lcublas
#SPARSE = -lamd -lcamd -lccolamd -lcholmod -lcolamd -lcxsparse -lblas
SPARSE = -lcholmod
LAPACK = -L$(LD_LIB_PATH)/atlas-base -llapack -lf77blas -lcblas -latlas
EIGEN = -I/usr/local/include/eigen3 -I/usr/local/include/eigen3/unsupported
TBB = -ltbb

# just about everything uses PortAudio and SDL
LFLAGS = -L$(LD_LIB_PATH) -lm -lpthread $(PORTAUDIO) $(LIBSDL)
#LFLAGS = -lsndfile -rdynamic -ltbb

CPPFLAGS = -DKAZOO_HACKS `sdl-config --cflags` \
	   -I/usr/include/suitesparse $(EIGEN)
CXXFLAGS = -Wall -Wextra -Winit-self -pedantic -Wno-unused-parameter \
	   -fno-exceptions -fno-rtti
NVCCFLAGS = -g -arch=sm_20 -use_fast_math \
	    -Xcompiler -Wall,-Wextra,-Winit-self,-Wno-unused-parameter \
	    -Xcompiler -fno-exceptions,-fno-rtti \
	    -Xcompiler -O3,-fopenmp,-pipe,-march=native,-ffast-math,-funsafe-math-optimizations,-ffinite-math-only,-fno-trapping-math,-mrecip,-mfpmath=sse,-fvect-cost-model

# XXX
CPPFLAGS += -DHACK_TO_LIMIT_GPU_POWER_USAGE

FAST_FLAGS = -O3 -fopenmp -pipe -march=native -ffast-math -funsafe-math-optimizations -ffinite-math-only -fno-trapping-math -mrecip -mfpmath=sse -fvect-cost-model
PROFILE_FLAGS = -O2 -march=native -fno-exceptions -fno-rtti -ffast-math -funsafe-math-optimizations -ffinite-math-only -fno-trapping-math -mrecip -mfpmath=sse -fvect-cost-model
#Additionial useful flags:
#  -ftree-vectorizer-verbose=2
STRIP_SYMBOLS = touch

# nvcc currently does not support gcc4.5 and later
CC = g++
CXX = g++
NVCC = nvcc
ifeq ($(PLATFORM), Cygwin)
	CC = g++-3
	CXX = g++-3
endif
ifeq ($(PLATFORM), Darwin)
	#CC = g++-mp-4.4
	#CXX = g++-mp-4.4
	LD_LIB_PATH = /opt/local/lib
	PORTAUDIO = $(LD_LIB_PATH)/libportaudio.a 
	LIBSDL = $(LD_LIB_PATH)/libSDLmain.a $(LD_LIB_PATH)/libSDL.a
        FRAMEWORKS = -framework Cocoa -framework CoreAudio -framework AudioUnit -framework AudioToolbox -framework Carbon -framework IOKit -framework OpenGL
        LFLAGS += -Wl,-search_paths_first $(FRAMEWORKS) -static-libgcc
endif

ifneq ($(MKLROOT),)
	CPPFLAGS += -DUSE_INTEL_MKL -I$(MKLROOT)/include
	LFLAGS += -Wl,--start-group $(MKLROOT)/lib/intel64/libmkl_intel_ilp64.a $(MKLROOT)/lib/intel64/libmkl_sequential.a $(MKLROOT)/lib/intel64/libmkl_core.a -Wl,--end-group
endif


ifneq ($(PLATFORM), Darwin)
	CPPFLAGS += -std=c++0x
	CXXFLAGS += -std=c++0x
	#NVCCFLAGS += -Xcompiler -std=c++0x # nvcc barfs with c++0x
endif

ifeq ($(COMPILE_TYPE), debug)
	CXXFLAGS += -ggdb $(FAST_FLAGS)
endif
ifeq ($(COMPILE_TYPE), debugcuda)
	CXXFLAGS += -ggdb $(FAST_FLAGS)
	NVCCFLAGS += -g -G
endif
ifeq ($(COMPILE_TYPE), slowdebug)
	CXXFLAGS += -ggdb
endif
ifeq ($(COMPILE_TYPE), release)
	CPPFLAGS += -DNDEBUG -DKAZOO_NDEBUG
	CXXFLAGS += -g3 $(FAST_FLAGS)
	LFLAGS += -Wl,-dead_strip
	STRIP_SYMBOLS = strip
endif
ifeq ($(COMPILE_TYPE), profile)
	CPPFLAGS += -DNDEBUG -DKAZOO_NDEBUG
	CXXFLAGS += -ggdb $(PROFILE_FLAGS)
	CXXFLAGS += -pg -ftest-coverage -fprofile-arcs
	LFLAGS   += -pg -ftest-coverage -fprofile-arcs
endif

#----( build rules )----------------------------------------------------------

# this hack fixes spurrious nvcc warning
NVCC = rm -f $@; nvcc
NVCCFILTER = 2>&1 | grep -v 'warning.*-fno-rtti'; test -e $@

.SUFFIXES:

%.cpp.o: %.cpp
	@echo compiling $<
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -c $< -o $@
%.cu.o: %.cu
	@echo compiling $<
	@$(NVCC) $(NVCCFLAGS) -c $< -o $@ $(NVCCFILTER)
%_test:
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) -o $@

network_flow.cpp.o: network_flow.cpp
	@echo compiling $<
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -frtti -fexceptions -c $< -o $@

cloud_kernels_tbb.cpp.o: cloud_kernels_tbb.cpp
	@echo compiling $<
	@$(CXX) $(CPPFLAGS) $(CXXFLAGS) -frtti -fexceptions -c $< -o $@

#----( compiler diagnostics )-------------------------------------------------

# Tip: to see what -march is doing, try
arch: FORCE arch.cpp
	@$(CXX) $(CXXFLAGS) arch.cpp -o $@
	@echo 'The following are equivalent:'
	@echo ' gcc -march=native'
	@gcc -march=native -E -v - </dev/null 2>&1 | sed -n 's/.* -v - / gcc /p'

# To diagnose sse optimization, try
sse: FORCE synchrony.cpp.o
	@objdump -d synchrony.cpp.o > temp.text
	@echo sse operations:
	@grep -o $@
		| sort -u  \
		| sed 's/^/ /'
	@#| sed 's/\(.*\)/ `grep -c \1 temp.text` \1/'
	@echo sse registers:
	@grep -o $@
		| sort -u \
		| sed 's/xmm//' \
		| sort -n \
		| sed 's/^/ xmm/'

#----( main targets )---------------------------------------------------------

kazoo: kazoo.cpp.o dirt.cpp.o table.cpp.o bucket.cpp.o ball.cpp.o sing.cpp.o rational.cpp.o tracker.cpp.o calibration.cpp.o playback.cpp.o streaming_devices.cpp.o streaming_camera.cpp.o streaming_video.cpp.o streaming_audio.cpp.o streaming_shared.cpp.o streaming.cpp.o matching.cpp.o flock.cpp.o particle_synthesis.cpp.o streaming_synthesis.cpp.o synthesis.cpp.o psycho.cpp.o synchrony.cpp.o regression.cpp.o linalg.cpp.o propagate.cpp.o filters.cpp.o probability.cpp.o optim.cpp.o splines.cpp.o camera.cpp.o audio.cpp.o animate.cpp.o events.cpp.o particles.cpp.o images.cpp.o gestures.cpp.o random.cpp.o vectors.cpp.o eigen.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(LAPACK) $(V4L2) $(PNG) $(FFMPEG) $(EIGEN) -o $@
	@$(STRIP_SYMBOLS) $@

learn: learn.cpp.o gloves.cpp.o voice.cpp.o raster_audio.cpp.o streaming_devices.cpp.o streaming_camera.cpp.o streaming_video.cpp.o streaming_audio.cpp.o streaming_shared.cpp.o streaming.cpp.o psycho.cpp.o synchrony.cpp.o streaming_clouds.cpp.o clouds.cpp.o cloud_persistence.cpp.o cloud_math.cpp.o cloud_points.cpp.o cloud_points_cpu.cpp.o cloud_points_gpu.cu.o cloud_kernels.cpp.o cloud_kernels_tbb.cpp.o cloud_kernels.cu.o cloud_flow.cpp.o cloud_map.cpp.o sparse.cpp.o linalg.cpp.o filters.cpp.o probability.cpp.o optim.cpp.o splines.cpp.o compress.cpp.o camera.cpp.o audio.cpp.o animate.cpp.o events.cpp.o images.cpp.o gestures.cpp.o random.cpp.o eigen_gpu.cpp.o gpu.cu.o histogram.cpp.o eigen.cpp.o vectors.cu.o vectors.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(LAPACK) $(V4L2) $(PNG) $(FFMPEG) $(CUDA) $(SPARSE) $(TBB) -o $@
	@$(STRIP_SYMBOLS) $@

toy: toy.cpp.o flock.cpp.o streaming_devices.cpp.o streaming_video.cpp.o streaming_audio.cpp.o streaming_shared.cpp.o streaming.cpp.o synthesis.cpp.o psycho.cpp.o synchrony.cpp.o regression.cpp.o linalg.cpp.o filters.cpp.o probability.cpp.o optim.cpp.o splines.cpp.o audio.cpp.o animate.cpp.o events.cpp.o particles.cpp.o images.cpp.o gestures.cpp.o random.cpp.o vectors.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(LAPACK) $(PNG) -o $@
	@$(STRIP_SYMBOLS) $@

player: player.cpp.o raster_audio.cpp.o streaming_devices.cpp.o streaming.cpp.o psycho.cpp.o synchrony.cpp.o animate.cpp.o audio.cpp.o images.cpp.o vectors.cpp.o events.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) -o $@
	@$(STRIP_SYMBOLS) $@

#----( data )-----------------------------------------------------------------

gv: FORCE
	./learn gloves new -n 1024 data/gloves/filtered-gloves-*.avi
	./learn voice new -n 4096 data/voice/*Amy*.avi
	./learn gv new

#----( testing )---------------------------------------------------------------

current_test: compress-test

test: limits_test misc_test random_test vectors_test speed_test lemon_test fftw_test fft_test warping_test splines_test transforms_test audio_test animate_test psycho_test camera_test avcodec_test compress_test matching_test network_flow_test array_test filters_test linalg_test sparse_test eigen_gpu_test calibration_test images_test synthesis_test synchrony_test raster_audio_test audio_alignment_test pde_test gpu_test aligned_allocator_test
run-test: misc-test vectors-test speed-test fftw-test fft-test warping-test psycho-test matching-test array-test filters-test linalg-test tracking-test aligned-allocator-test

limits_test: limits_test.cpp.o common.cpp.o
limits-test: limits_test FORCE
	./limits_test

random_test: random_test.cpp.o common.cpp.o
random-test: random_test FORCE
	./random_test

misc_test: misc_test.cpp.o common.cpp.o
misc-test: misc_test FORCE
	./misc_test

speed_test: speed_test.cpp.o images.cpp.o vectors.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) -o $@
speed_test.text: speed_test
	./speed_test | tee speed_test.text
speed-test: speed_test.text FORCE
	cat speed_test.text

vectors_test: vectors_test.cpp.o vectors.cpp.o args.cpp.o common.cpp.o
vectors-test: vectors_test FORCE
	./vectors_test

lemon_test.cpp.o:
	@echo compiling $<
	@$(CXX) $(CXXFLAGS) -frtti -fexceptions -c lemon_test.cpp -o $@
lemon_test: lemon_test.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) -frtti -fexceptions $^ $(LFLAGS) $(LEMON) -o $@
lemon-test: lemon_test FORCE
	./lemon_test

fftw_test: fftw_test.cpp.o vectors.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(FFTW3F) -o $@
fftw-test: fftw_test FORCE
	./fftw_test

fft_test: fft_test.cpp.o vectors.cpp.o common.cpp.o fft.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(FFTW3F) -o $@
fft-test: fft_test FORCE
	./fft_test

warping_test: warping_test.cpp.o vectors.cpp.o splines.cpp.o fft.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(FFTW3F) -o $@
warping-test: warping_test FORCE
	./warping_test

splines_test: splines_test.cpp.o vectors.cpp.o splines.cpp.o common.cpp.o
splines-test: splines_test FORCE
	./splines_test

TRANSFORMS_O = transforms.cpp.o threads.cpp.o splines.cpp.o vectors.cpp.o fft.cpp.o common.cpp.o
transforms_test: transforms_test.cpp.o $(TRANSFORMS_O)
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(FFTW3F) -o $@
transforms-test: transforms_test FORCE
	./transforms_test

audio_test: audio_test.cpp.o audio.cpp.o vectors.cpp.o common.cpp.o
audio-test: audio_test FORCE
	./audio_test

animate_test: animate_test.cpp.o animate.cpp.o events.cpp.o vectors.cpp.o threads.cpp.o common.cpp.o
animate-test: animate_test FORCE
	./animate_test

psycho_test: psycho_test.cpp.o psycho.cpp.o synchrony.cpp.o images.cpp.o splines.cpp.o vectors.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(FFTW3F) $(PNG) -o $@
psycho-test: psycho_test FORCE
	./psycho_test

camera_test: camera_test.cpp.o camera.cpp.o images.cpp.o filters.cpp.o animate.cpp.o events.cpp.o vectors.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(V4L2) $(PNG) -o $@
full: camera_test FORCE
	./camera_test full
crop: camera_test FORCE
	./camera_test crop
mask: camera_test FORCE
	./camera_test mask

avcodec_test: avcodec_test.cpp.o streaming.cpp.o events.cpp.o vectors.cpp.o threads.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(FFMPEG) -o $@
avcodec-test: avcodec_test
	./avcodec_test

compress_test: compress_test.cpp.o compress.cpp.o streaming_camera.cpp.o streaming_devices.cpp.o streaming.cpp.o camera.cpp.o animate.cpp.o audio.cpp.o events.cpp.o images.cpp.o vectors.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) $(FFMPEG) $(V4L2) -o $@
compress-test: compress_test
	./compress_test encode_mono_fifth data/test.avi
	./compress_test decode data/test.avi 0.5

images_test: images_test.cpp.o images.cpp.o camera.cpp.o animate.cpp.o events.cpp.o vectors.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) $(V4L2) -o $@
images-test: images_test FORCE
	./images_test camera cross
hdr: images_test FORCE
	./images_test camera hdr
dilate: images_test FORCE
	./images_test camera dilate

cellular_test: cellular_test.cpp.o cellular.cpp.o common.cpp.o vectors.cpp.o

matching_test: matching_test.cpp.o matching.cpp.o common.cpp.o
matching-test: matching_test FORCE
	./matching_test

network_flow_test: network_flow_test.cpp.o network_flow.cpp.o vectors.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(LEMON) -o $@
network-flow-test: network_flow_test FORCE
	./network_flow_test

propagate_test: propagate_test.cpp.o propagate.cpp.o common.cpp.o
propagate-test: propagate_test FORCE
	./propagate_test

linalg_test: linalg_test.cpp.o linalg.cpp.o images.cpp.o vectors.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(LAPACK) $(PNG) -o $@
linalg-test: linalg_test FORCE
	./linalg_test

sparse_test: sparse_test.cpp.o sparse.cpp.o linalg.cpp.o images.cpp.o vectors.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) $(LAPACK) $(SPARSE) -o $@
sparse-test: sparse_test FORCE
	./sparse_test ring

eigen_gpu_test: eigen_gpu_test.cpp.o eigen_gpu.cpp.o eigen.cpp.o gpu.cu.o vectors.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(CUDA) -o $@

regression_test: regression_test.cpp.o regression.cpp.o linalg.cpp.o vectors.cpp.o common.cpp.o
regression-test: regression_test FORCE
	./regression_test 1 2 4

calibration_test: calibration_test.cpp.o calibration.cpp.o regression.cpp.o linalg.cpp.o propagate.cpp.o splines.cpp.o images.cpp.o vectors.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) $(LAPACK) -o $@
data/background.im: camera_test FORCE
	./camera_test mask || \
	test -e data/background.im && \
	echo '+-----------------------------------------+' && \
	echo '| WARNING using cached data/background.im |' && \
	echo '+-----------------------------------------+'
calibration-test: camera_test images_test calibration_test data/background.im FORCE
	./calibration_test
	./images_test data/crosses.im wire &
	test/calibration.py show data/verts.text data/edges.text &
bp-test: camera_test images_test calibration_test FORCE
	./calibration_test 3 1 0 2 # radius em_steps nls_steps bp_steps
	@#./images_test data/crosses.im wire &
	test/calibration.py show data/verts.text data/edges.text &

synthesis_test: synthesis_test.cpp.o synthesis.cpp.o audio.cpp.o synchrony.cpp.o images.cpp.o splines.cpp.o particles.cpp.o random.cpp.o vectors.cpp.o threads.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) -o $@
synthesis-test: synthesis_test FORCE
	./synthesis_test sustain

synchrony_test: synchrony_test.cpp.o synchrony.cpp.o reassign.cpp.o animate.cpp.o events.cpp.o fft.cpp.o splines.cpp.o images.cpp.o vectors.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) $(FFTW3F) -o $@
synchrony-test: synchrony_test FORCE
	./synchrony_test phasor
	@#./synchrony_test fourier
	@#./synchrony_test loop

rational_test: rational_test.cpp.o rational.cpp.o synchrony.cpp.o images.cpp.o vectors.cpp.o eigen.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) $(LAPACK) $(EIGEN) -o $@
rational-test: rational_test FORCE
	./rational_test

raster_audio_test: raster_audio_test.cpp.o raster_audio.cpp.o streaming_devices.cpp.o streaming.cpp.o animate.cpp.o audio.cpp.o vectors.cpp.o events.cpp.o threads.cpp.o args.cpp.o common.cpp.o
raster-audio-test: raster_audio_test
	./raster_audio_test pitch

audio_alignment_test: audio_alignment_test.cpp.o audio_alignment.cpp.o raster_audio.cpp.o voice.cpp.o streaming_devices.cpp.o streaming.cpp.o animate.cpp.o audio.cpp.o psycho.cpp.o synchrony.cpp.o splines.cpp.o vectors.cpp.o images.cpp.o events.cpp.o threads.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(PNG) -o $@
audio-alignment-test: audio_alignment_test
	./audio_alignment_test path && ristretto data/path.png &

pde_test: pde_test.cpp.o splines.cpp.o vectors.cpp.o common.cpp.o
pde-test: pde_test FORCE
	./pde_test

address_test: address_test.cpp.o address.cpp.o common.cpp.o

array_test: array_test.cpp.o common.cpp.o
array-test: array_test FORCE
	./array_test

filters_test: filters_test.cpp.o filters.cpp.o common.cpp.o
filters-test: filters_test FORCE
	./filters_test

sse_test: sse_test.cpp.o args.cpp.o common.cpp.o FORCE
	objdump -d sse_test.cpp.o | grep addps

gpu_test: gpu_test.cpp.o gpu_test.cu.o gpu.cu.o cloud_kernels.cpp.o cloud_kernels_tbb.cpp.o cloud_kernels.cu.o vectors.cu.o vectors.cpp.o args.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) $(CUDA) $(TBB) -o $@
gpu-test: gpu_test
	./gpu_test info
	./gpu_test distances

clouds-test: learn
	./learn cloud new

voice-test: learn
	./learn voice convert
	mplayer -nofs -xy 3 -fps 1 data/test.raw.avi

aligned_allocator_test: aligned_allocator_test.cpp.o
aligned-allocator-test: aligned_allocator_test FORCE
	./aligned_allocator_test


#----( profiling )-------------------------------------------------------------

supergram_profile: supergram_profile.cpp.o reassign.cpp.o splines.cpp.o network.cpp.o vectors.cpp.o fft.cpp.o threads.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) ^$ $(LFLAGS) $(FFTW3F) -o $@
supergram-profile: supergram_profile FORCE
	./supergram_profile
	gprof -b supergram_profile > supergram_profile.prof
	gvim supergram_profile.prof &

synthesis_profile: synthesis_profile.cpp.o synthesis.cpp.o audio.cpp.o vectors.cpp.o threads.cpp.o common.cpp.o
	@echo linking $@
	@$(CXX) $(CXXFLAGS) $^ $(LFLAGS) -o $@
synthesis-profile: synthesis_profile FORCE
	./synthesis_profile
	gprof -b synthesis_profile > synthesis_profile.prof
	gvim synthesis_profile.prof &

VALGRIND = valgrind --leak-check=full --alignment=16
mem-test: synchrony_test speed_test
	$(VALGRIND) --log-file=synchrony_test.vlog ./synchrony_test && gvim synchrony_test.vlog &
	$(VALGRIND) --log-file=speed_test.vlog ./speed_test && gvim speed_test.vlog &

#----( misc )------------------------------------------------------------------

alsa: FORCE
	sudo /sbin/alsa force-reload && \
	alsactl init && \
	amixer set Master 100 unmute && \
	amixer set 'Input Source' Line
jackd: FORCE
	@killall -q -w jackd && echo 'restarting jackd' || echo 'starting jackd'
	@/usr/bin/jackd --realtime --realtime-priority=50 --timeout=500 --nozombies \
	       	--driver=alsa --rate=48000 --period=64 --nperiods=2 --softmode &

# see http://sourceforge.net/apps/trac/fluidsynth/wiki/Documentation
fluidsynth: FORCE
	killall -q -w fluidsynth \
		&& echo 'starting fluidsynth' \
		|| echo 'restarting fluidsynth'
	/usr/bin/fluidsynth -C0 -R0 -a alsa -l -r 48000 -o audio.alsa.device=hw:0

todo:
	@echo '\033[1m-------- Top 40 of' \
		"$$(cat *.h *.cpp *.cu *.py | grep -c TODO)" \
		'tasks --------\033[0m'
	@ls -t \
	| grep '.\(h\|cpp\|cu\|py\)$$' \
	| xargs grep 'TODO' --color=yes 2>/dev/null\
	| head -n 40

style:
	cpplint.py --filter=-whitespace/parens,-whitespace/braces *.h *.cpp 1&> temp.style || gvim temp.style

lclean: FORCE
	rm -f kazoo learn toy arch *_test *_profile *.exe
oclean: FORCE
	rm -f *.o

clean: lclean oclean FORCE
	rm -f Makefile.bak
	rm -f temp.diff temp.style
	rm -f *.pyc *.pyo */*.pyc */*.pyo
	rm -f gmon.out *.gcda *.gcno *.prof
	rm -f core *.core *.core.* *.stackdump
	rm -f vgcore.* *.vlog valgrind.log
	rm -rf temp
	rm -f *.png *.jpg *.jpeg *.mp4 *.pdf
	rm -f speed_test.text

cleaner: clean
	rm -f *.wav *.mp3 *.raw *.pd *.im *.text
	rm -f temp*

depend:
	@makedepend -D__GNUG__ -Y *.cpp -o.cpp.o -fcpp.depend 2> makedepend.log
	@makedepend -D__GNUG__ -Y *.cu -o.cu.o -fcu.depend 2>> makedepend.log
	@! grep -i error makedepend.log

include cpp.depend
include cu.depend

FORCE:

