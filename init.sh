#!/bin/sh

# to install,
if [ 0 ]; then
	sudo cp init.sh /etc/init.d/kazoo
	sudo chmod 700 /etc/init.d/kazoo
	sudo update-rc.d kazoo defaults
fi

# kill existing jackd instances, so we get the right settings
killall -q -w jackd

/usr/bin/jackd --realtime --realtime-priority=50 --timeout=500 --nozombies \
  --driver=alsa --rate=48000 --period=128 --nperiods=2 --softmode && \
  /home/fritz/kazoo/src/play sand &

