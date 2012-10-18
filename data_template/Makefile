
all: pull test.raw

#----( rsync )-----------------------------------------------------------------

DATA = fritzoor@fritzo.org:kazoo_data
SYNC = rsync -avuz --exclude-from=rsyncignore --skip-compress=mp3/png/avi -e ssh

pull: FORCE
	$(SYNC) $(DATA)/ ./

push: FORCE
	$(SYNC) ./ $(DATA)/

#----( derived files )---------------------------------------------------------

test.raw: test.mp3
	madplay --stereo -o test.raw test.mp3

mono.raw: test.mp3
	madplay --mono -o mono.raw test.mp3

#----( misc )------------------------------------------------------------------

clean: FORCE
	rm -f temp.*
	$(MAKE) -C raster_audio clean
	$(MAKE) -C gloves clean
	$(MAKE) -C voice clean
	$(MAKE) -C gg clean
	$(MAKE) -C gv clean

cleaner: clean
	rm -rf *.*
	$(MAKE) -C raster_audio cleaner
	$(MAKE) -C gloves cleaner
	$(MAKE) -C voice cleaner
	$(MAKE) -C gg cleaner
	$(MAKE) -C gv cleaner

cloud-format-change: FORCE
	$(MAKE) -C gloves cloud-format-change
	$(MAKE) -C voice cloud-format-change
	$(MAKE) -C gg cloud-format-change
	$(MAKE) -C gv cloud-format-change

FORCE:

