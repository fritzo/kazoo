
# install LEMON graph library
# http://lemon.cs.elte.hu/trac/lemon/wiki/Downloads
lemon=lemon-1.2.2
test -e ~/src || mkdir ~/src && cd ~/src
test -e $lemon.tar.gz || wget http://lemon.cs.elte.hu/pub/sources/$lemon.tar.gz
tar -xzf $lemon.tar.gz
cd $lemon
./configure && \
make clean && \
make && \
make check && \
sudo make install \
|| echo "ERROR failed to install lemon"


lemondoc=lemon-doc-1.2.2
test -e ~/src || mkdir ~/src && cd ~/src
test -e $lemondoc.tar.gz || wget http://lemon.cs.elte.hu/pub/sources/$lemondoc.tar.gz
tar -xzf $lemondoc.tar.gz

