#!/bin/bash

#the shebang at the top is pulled from bing copilot, "which scripting language should I use for linux?". It makes sure
#that the script is run with bash, which is default shell language for most linux distributions

#now i also added something so it can run itself with sudo automatically, got from copilot bing:
# "having bash script run itself with sudo priviledges"
# Check if the script is running as root; if not, re-execute itself with sudo
[ "$EUID" -ne 0 ] && exec sudo "$0" "$@"

#install core isolation tools (cpuset)
apt install cpuset
Y

#install frequency governor tools (linux-cpupower)
apt install linux-cpupower
Y
