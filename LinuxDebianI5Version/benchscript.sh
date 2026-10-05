#!/bin/bash

#the shebang at the top is pulled from bing copilot, "which scripting language should I use for linux?". It makes sure
#that the script is run with bash, which is default shell language for most linux distributions

#now i also added something so it can run itself with sudo automatically, got from copilot bing:
# "having bash script run itself with sudo priviledges"
# Check if the script is running as root; if not, re-execute itself with sudo
[ "$EUID" -ne 0 ] && exec sudo "$0" "$@"

#Set core isolation to core 0 and pin program to that core
#cset shield --cpu 0 --kthread=on

echo "Note: you must still disable SMT/Turbo boost yourself in BIOS before running this script"
echo "Configuration notes: This script will only work on Intel CPUs, only tested with debian. Also, working set sizes for caching must be customized to the CPU model."
echo "Configuration notes: This script is configured with AVX-512 enabled, if your CPU doesn't support it, you must change it to AVX2"

touch compilation.txt
touch scalarCompilation.txt
touch vectorCompilation.txt
touch scalarResults.txt
touch vectorResults.txt
touch results.txt

#chmod rw compilation.txt
#chmod rw scalarCompilation.txt
#chmod rw vectorCompilation.txt
#chmod rw scalarResults.txt
#chmod rw vectorResults.txt
#chmod rw results.txt

#everything below this was assisted by gemini and cpupower online documentation. Definitely heavily modified

CORE_END=$(($(nproc) - 1)) #this gets the number of cores on the system and subtracts 1 to get the last core index (since core indexing starts at 0)

#push live processes off core 0
systemctl set-property -- -.slice AllowedCPUs="1-$CORE_END"
systemctl set-property -- user.slice AllowedCPUs="1-$CORE_END"
#force systemd daemon to respect the dynamic property changes
systemctl daemon-reload

#set intel pstate to passive mode, allows for direct frequency change instead of preset governors
echo passive | tee /sys/devices/system/cpu/intel_pstate/status

#Setting governor to userspace (allows for direct frequency locking)
cpupower -c 0 frequency-set -g userspace #this sets core 0 to userspace mode, which allows for pinning exact frequency

#disable turbo boost
echo 1 | tee /sys/devices/system/cpu/intel_pstate/no_turbo #this disables turbo boost on all cores

#disable all sleep states
cpupower idle-set -D 0 #forces C0 active state by limiting it to 0 microseconds allowed in sleep state

#lock frequency to a fixed value (1.4GHz for this case, can be changed)
cpupower -c 0 frequency-set -f 1400000kHz #this sets core 0 to 1.4GHz (unit is in KHz by default)
# Clear explicit frequency constraints
cpupower -c 0 frequency-set -u 1400MHz -d 1400MHz
cpupower -c 0 frequency-info #this prints out the current frequency info for core 0, which should now be in userspace mode
#cpupower -c 0 monitor

#compiling program (deliverables 2-5) with SIMD flags enabled
gcc -O3 -mavx512vl project1.cpp -lstdc++ -lm -o project1.exe | tee compilation.txt #SIMD flags

#compiling program (deliverable 1) 2 times, one with scalar flags and one with auto-vectorization flags enabled
gcc -O3 -fno-tree-vectorize project1Scalar.cpp -lstdc++ -lm -o project1Scalar.exe | tee scalarCompilation.txt #scalar flags
gcc -O3 project1Vector.cpp -lstdc++ -lm -o project1Vector.exe | tee vectorCompilation.txt #auto-vectorization flags

#run the program (deliverable 1) with the core isolation and frequency settings applied 2 times, one with scalar flags and the other with auto-vectorization flags
#cset shield --exec -- ./project1Scalar.exe | tee scalarResults.txt
#cset shield --exec -- ./project1Vector.exe | tee vectorResults.txt

#run it on that core
systemd-run --scope -p AllowedCPUs=0 ./project1Scalar.exe | tee scalarResults.txt
systemd-run --scope -p AllowedCPUs=0 ./project1Vector.exe | tee vectorResults.txt

#run the program (deliverables 2-5) with the core isolation and frequency settings applied
#cset shield --exec -- ./project1.exe | tee results.txt #this will be done near the end of the script

#run it on that core
systemd-run --scope -p AllowedCPUs=0 ./project1.exe | tee results.txt #this will be done near the end of the script

#reset parts of the script
#reseting core isolation
#cset shield --reset
#re-enable all cores for the system and users
systemctl set-property -- -.slice AllowedCPUs="0-$CORE_END"
systemctl set-property -- user.slice AllowedCPUs="0-$CORE_END"
#reset systemd to its default configuration state
systemctl daemon-reload
#puts intel pstate back to active mode, which allows for dynamic frequency scaling
echo active | tee /sys/devices/system/cpu/intel_pstate/status
cpupower -c 0 frequency-set -g powersave
# Clear explicit frequency constraints
cpupower -c 0 frequency-set -u 100% -d 0%
# 1. Re-enable C-states and clear latency constraints
cpupower idle-set -D 0ms
cpupower idle-set -E
#re-enable turbo boost
echo 0 | tee /sys/devices/system/cpu/intel_pstate/no_turbo
#then restart your computer to undo the rest

#note: you can live monitor the frequency by using: watch -n 0.5 "grep 'cpu MHz' /proc/cpuinfo"

#gemini also recommended this for monitoring performance for SIMD: perf stat -e cycles,instructions,cache-misses,L1-dcache-load-misses taskset -c 0 ./project1test.exe