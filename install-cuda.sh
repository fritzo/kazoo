
# setup for Ubuntu 10.10, as per
# http://samiux.blogspot.com/2011/04/howto-nvidia-cuda-40-rc-on-ubuntu-1010.html

# Add the CUDA 4.0 PPA.
sudo add-apt-repository ppa:aaron-haviland/cuda-4.0
sudo apt-get update
sudo apt-get upgrade

# install cuda drivers & toolkit
sudo apt-get install \
	nvidia-cuda-gdb \
	nvidia-cuda-toolkit \
	nvidia-compute-profiler \
	libnpp4 \
	nvidia-cuda-doc \
	libcudart4 \
	libcublas4 \
	libcufft4 \
	libcusparse4 \
	libcurand4 \
	nvidia-current \
	nvidia-opencl-dev \
	nvidia-current-dev \
	nvidia-cuda-dev \
	opencl-headers

sudo apt-get install nvidia-kernel-common
sudo apt-get install libvdpau-doc
sudo apt-get install nvidia-current-modaliases

# setup nvidia drivers, if your system has not already done so
# WARNING this breaks systems without nvidia gpus
#sudo nvidia-xconfig

# install sample code
#...

