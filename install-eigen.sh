
sudo apt-get install cmake

# install eigen3
eigen=3.0.2
test -e ~/src || mkdir ~/src && cd ~/src
test -e $eigen.tar.bz2 || wget http://bitbucket.org/eigen/eigen/get/$eigen.tar.bz2
tar -xjf $eigen.tar.bz2

build_dir=~/src/eigen-$eigen-build
source_dir=~/src/eigen-eigen-$eigen
test -e $build_dir || mkdir $build_dir
cd $build_dir
cmake $source_dir && \
sudo make install

