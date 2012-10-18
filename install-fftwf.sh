
# build single-precision version of FFTW3
# modify Makefiles to compile with CFLAGS += -fPIC as recommended by
#  http://forums.fedoraforum.org/showthread.php?t=232607
fftw=fftw-3.2.2
test -e ~/src || mkdir ~/src && cd ~/src
test -e $fftw.tar.gz || wget http://www.fftw.org/$fftw.tar.gz
tar -xzf $fftw.tar.gz
cd $fftw
./configure --enable-single --enable-sse && \
(find . | grep 'Makefile$' \
        | xargs sed -i 's/^CFLAGS = /CFLAGS = -fPIC /g') && \
make clean && \
make && \
sudo make install \
|| echo "ERROR failed to install fftw"

