#!/bin/sh
# set up the headless display for pyvista off-screen rendering on example plots
set -x
export DISPLAY=:99.0
export PYVISTA_OFF_SCREEN=True
which Xvfb
Xvfb :99 -screen 0 1024x768x24 > /dev/null 2>&1 &
# give xvfb some time to start
sleep 3
set +x
