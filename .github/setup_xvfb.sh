#!/bin/sh
# set up the headless display for pyvista off-screen rendering on example plots
set -x
sudo apt-get update && sudo apt-get install libgl1-mesa-glx xvfb -y
sudo apt-get install libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libopengl0 libegl1 libosmesa6 mesa-utils libxcb-shape0 libx
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=True
which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
# give xvfb some time to start
sleep 3
set +x

# This also includes the libraries necessary for PyQt5/PyQt6
sudo apt-get install xvfb libxkbcommon-x11-0 libxcb-icccm4 libxcb-image0 libxcb-keysyms1 libxcb-randr0 libxcb-render-util0 libxcb-xinerama0 libxcb-xfixes0 libopengl0 libegl1 libosmesa6 mesa-utils libxcb-shape0 libxcb-cursor0 -y
/sbin/start-stop-daemon --start --quiet --pidfile /tmp/custom_xvfb_99.pid --make-pidfile --background --exec /usr/bin/Xvfb -- :99 -screen 0 1400x900x24 -ac +extension GLX +render -noreset
