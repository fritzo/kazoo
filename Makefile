
current-target: build

#----( installing )------------------------------------------------------------

build: FORCE
	python setup.py build
	$(MAKE) -C src all

install:
	python setup.py build &&\
	sudo python setup.py install
cyg-install:
	python setup.py install
win-install:
	/cygdrive/c/Python26/python.exe setup.py install

#----( misc )------------------------------------------------------------------

all: build
	$(MAKE) -C src all
	$(MAKE) -C data all

doc: FORCE
	$(MAKE) -C doc all

todo:
	@grep '.*(T[0-9]\+\(.[0-9]\+\)*) ' *.py */*.text */*.tex */*.py */*.h */*.cpp --color=yes

clean:
	rm -f *.tar.gz
	rm -rf build
	#rm -f bin/kazoo bin/toy bin/learn
	$(MAKE) -C doc clean
	$(MAKE) -C src clean
	$(MAKE) -C test clean
	$(MAKE) -C data cleaner

cleaner: clean
	$(MAKE) -C src cleaner
	$(MAKE) -C data cleaner

FORCE:

