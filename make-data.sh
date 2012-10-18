
# TODO figure out how to replace all of this script with a call to rsync

test -e ~/kazoo_data || mkdir ~/kazoo_data
ln -sf ~/kazoo_data data
cp data_template/* data/

for d in raster_audio gloves voice gg gv
do
	test -d ~/kazoo_data/$d || mkdir ~/kazoo_data/$d
	cp data_template/$d/* data/$d/
done

