#!/bin/bash
sudo modprobe peak_pci

sudo ip link set can1 type can bitrate 1000000
sudo ip link set up can1