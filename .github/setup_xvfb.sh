#!/bin/sh
set -x
sudo apt-get update && sudo apt-get install libgl1-mesa-glx xvfb -y
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=True
which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
# give xvfb some time to start
sleep 3
set +x


##!/bin/bash -ef
#
## this is only relevant for GitHub Actions, but it avoids
## https://github.com/actions/virtual-environments/issues/323
## via
## https://github.community/t/ubuntu-latest-apt-repository-list-issues/17182/10#M4501
#for apt_file in `grep -lr microsoft /etc/apt/sources.list.d/`; do
#    echo "Removing $apt_file"
#    rm $apt_file
#done
#
## This also includes the libraries necessary for PyQt5/PyQt6
#apt-get update
#apt-get install -yqq xvfb libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libopengl0 libegl1 libosmesa6 mesa-utils libxcb-shape0 libxcb-cursor0
#/sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset
