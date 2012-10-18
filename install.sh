#!/bin/sh

# This script installs build tools for the kazoo project.
# Requires: ubuntu 9.10 -- 10.10

i='sudo apt-get install -y'

# install developer tools
$i libtbb-dev
$i makedepend || $i xutils-dev
$i g++ liblapack-dev libsdl-dev
$i libpng-dev
$i libv4l-dev
$i portaudio19-dev libstk0-dev # these conflicts with jack2
$i libcvaux-dev libcv-dev libhighgui-dev
$i python-numpy python-scipy python-matplotlib python-dev
$i libsuitesparse-dev
$i mplayer

# install jack
$i jackd1 qjackctl madplay
sudo echo '@audio  -  rtprio  80' >> /etc/security/limits.conf
sudo echo '@audio  -  nice    -10' >> /etc/security/limits.conf
sudo echo '@audio  -  memlock  760974' >> /etc/security/limits.conf

# install non-packaged tools
./install-fftwf.sh
./install-lemon.sh
./install-cuda.sh

# make data directory
./make-data.sh

# install documentation tools
if [ 0 ]; then
	$i texlive xfig
fi

# install fluidsynth & soundfonts for midi synthesis
#$i fluidsynth fluid-soundfont-gm

