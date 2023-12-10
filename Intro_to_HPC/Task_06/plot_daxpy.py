# Write a script that plots the memory bandwidth usage in GB/s vs the memory footprint of daxpy.
# Hint: the memory footprint is the amount of memory used by the function, which you can calculate from the vector size and amount of reads or writes performed per element.
# You can neglect the impact of the scalar.
# Store the plot as daxpy.png, in your solutions folder.

#       Size      BW[GB/s]      Runtime[ms]
#       1000      39.669       0.001
#      10000      40.609       0.006
#     100000      40.451       0.059
#    1000000      24.448       0.982
#   10000000      23.122      10.380
#  100000000      23.109     103.854
# 1000000000      23.022    1042.488

# get arguments from command line
import sys

args = sys.argv

if len(args) != 2:
    print("Usage: python3 plot_daxpy.py <filename>")
    sys.exit(1)

with open(f"solutions/{args[1]}", 'r') as f:
    lines = f.readlines()

sizes = []
bandwidths = []
runtimes = []
footprints = []

for line in lines[7:]:
    size, bandwidth, runtime = line.split()
    sizes.append(int(size))
    bandwidths.append(float(bandwidth))
    runtimes.append(float(runtime))

    footprint = 3 * int(size) * 8
    footprints.append(footprint)

import matplotlib.pyplot as plt

plt.plot(footprints, bandwidths, 'o')
plt.xlabel("Memory footprint [bytes]")
plt.ylabel("Memory bandwidth [GB/s]")

plt.xscale("log")
plt.yscale("log")

# add vertical lines for 4MiB, 64MiB, 512MiB - L1D, L2, L3
plt.axvline(4 * 1024 * 1024, color='k', linestyle='--')
plt.axvline(64 * 1024 * 1024, color='k', linestyle='--')
plt.axvline(512 * 1024 * 1024, color='k', linestyle='--')

# add horizontal line for 204.8 GB/s - peak memory bandwidth
plt.axhline(204.8, color='b', linestyle='--')

plt.savefig(f"solutions/{args[1][:-3]}.png")
