#!/bin/sh

# This script installs a realtime kernel in which to run the kazoo project.

sudo -i

# Install 2.6.33-rt kernel, as described in
#   www.pubbs.net/201006/linuxaudio/28168-lau-lucid-lynx-and-rt-kernels.html
# This kernel has Richard Kaswy's patch merged into mainline,
# and allows PS3 eye parameters to be set via V4L.
echo 'deb http://ppa.launchpad.net/abogani/ppa/ubuntu natty main' >> /etc/apt/sources.list
echo 'deb-src http://ppa.launchpad.net/abogani/ppa/ubuntu natty main' >> /etc/apt/sources.list
add-apt-repository ppa:abogani/ppa
apt-get update
apt-get install linux-lowlatency

exit #------------------------------------------------------------------------

# configure kernel:
# * compile in ext2 support when using ext2 on SSD drives
# * multimedia -> v4l -> usb camera -> gspca -> ov534
# * fully preemptible kernel
# * timer frequency = 1000
# * disable kernel tracing and frame pointers
# * optimize for size
cd /usr/src
apt-get source linux-image-2.6.38-8-lowlatency
cd linux-lowlatency-2.6.38
make defconfig
make menuconfig # ...

# build kernel
make -j 5 all

# install kernel
make modules_install
make install
update-initramfs -c -k 2.6.33.7-rt29-atom

# notify grub2
update-grub

# Note: in Ubuntu 10.01, one can install the natty kernel, as described in
# http://ubuntuforums.org/showpost.php?p=10072326&postcount=10

