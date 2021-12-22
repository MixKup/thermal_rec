#!/bin/bash
cd /home/fm/cameraservice-jetson_nano/eventstreamclient/samples/thermal-raw;
tmux new-session -d -s matrix_session 'while sleep 1; do /home/fm/cameraservice-jetson_nano/eventstreamclient/samples/thermal-raw/build/thermal-raw 10.0.7.117:80; done'
