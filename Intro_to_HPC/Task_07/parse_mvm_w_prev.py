"""
STREAM version: 5.10
gcc -O1
gcc (GCC) 11.3.0
Run: Node: jrc0667; ntasks: 1
=====================================
        64        8192       98304      32.768       0.003
       128       32768      393216      28.087       0.014
       256      131072     1572864      28.598       0.055
"""

import sys

import matplotlib.pyplot as plt

filename = sys.argv[1]
prev_filename = sys.argv[2]
pic_name = sys.argv[3]

with open(filename, 'r') as f:
    lines = f.readlines()

vec_size = []
flops = []
bytes_n = []
bw = []
rtime = []

for line in lines[5:]:
    vec_size.append(int(line.split()[0]))
    flops.append(float(line.split()[1]))
    bytes_n.append(float(line.split()[2]) / 1024 / 1024)
    bw.append(float(line.split()[3]))
    rtime.append(float(line.split()[4]))

# previous

with open(prev_filename, 'r') as f:
    lines = f.readlines()

prev_vec_size = []
prev_flops = []
prev_bytes_n = []
prev_bw = []
prev_rtime = []

for line in lines[5:]:
    prev_vec_size.append(int(line.split()[0]))
    prev_flops.append(float(line.split()[1]))
    prev_bytes_n.append(float(line.split()[2]) / 1024 / 1024)
    prev_bw.append(float(line.split()[3]))
    prev_rtime.append(float(line.split()[4]))

# Produce a dual-plot (two plots side-by-side)
# to show the memory bandwidth (in GB/s) vs. memory footprint (in MB),
# and the runtime (msec) vs. memory footprint (in MB).

fig, axs = plt.subplots(1, 2, figsize=(10, 5))

axs[0].plot(bytes_n, bw, '-o', label='bw')
axs[0].plot(prev_bytes_n, prev_bw, '-o', label='prev_bw')

axs[0].set_xlabel("Memory footprint [MB]")
axs[0].set_ylabel("Memory bandwidth [GB/s]")

axs[0].set_xscale("log")
axs[0].set_yscale("log")

# add vertical lines for 4MiB, 64MiB, 512MiB - L1D, L2, L3
axs[0].axvline(4 * 1024 * 1024, color='k', linestyle='--')
axs[0].axvline(64 * 1024 * 1024, color='k', linestyle='--')
axs[0].axvline(512 * 1024 * 1024, color='k', linestyle='--')

axs[0].legend()
axs[0].set_title("Memory bandwidth vs. memory footprint")

axs[1].plot(bytes_n, rtime, '-o', label='rtime')
axs[1].plot(prev_bytes_n, prev_rtime, '-o', label='prev_rtime')

axs[1].set_xlabel("Memory footprint [MB]")
axs[1].set_ylabel("Runtime [msec]")

axs[1].set_xscale("log")
axs[1].set_yscale("log")

# add vertical lines for 4MiB, 64MiB, 512MiB - L1D, L2, L3
axs[1].axvline(4 * 1024 * 1024, color='k', linestyle='--')
axs[1].axvline(64 * 1024 * 1024, color='k', linestyle='--')
axs[1].axvline(512 * 1024 * 1024, color='k', linestyle='--')

axs[1].legend()
axs[1].set_title("Runtime vs. memory footprint")

plt.savefig(pic_name)
