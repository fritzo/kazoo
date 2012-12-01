
sudo apt-get install cmake libatlas-base-dev
# TODO maybe just sudo apt-get install libeigen3-dev

# install eigen3
eigen=3.0.2 # see http://eigen.tuxfamily.org for latest version
test -e ~/src || mkdir ~/src && cd ~/src
test -e $eigen.tar.bz2 || wget http://bitbucket.org/eigen/eigen/get/$eigen.tar.bz2

rm -rf eigen-eigen-*
tar -xjf $eigen.tar.bz2
mv eigen-eigen-* eigen-eigen-$eigen

source_dir=~/src/eigen-eigen-$eigen
build_dir=~/src/eigen-$eigen-build
test -e $build_dir || mkdir $build_dir
cd $build_dir
cmake $source_dir && \
sudo make install

