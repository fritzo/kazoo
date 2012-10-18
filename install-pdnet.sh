
sudo apt-get install gfortran

# install pdnet
test -e ~/src || mkdir ~/src && cd ~/src
test -e src-pdnet.tar.gz || wget http://www2.research.att.com/~mgcr/pdnet/src-pdnet.tar.gz
tar -xzf src-pdnet.tar.gz
cd src-pdnet
test -e Makefile.bak || sed -i.bak 's/\<f77\>/gfortran/g' Makefile
make cdriver

