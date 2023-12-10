# vec_size copy_rate scale_rate add_rate
# 10000 26843.5 31956.6 83886.1
# 100000 27169.6 27169.6 58525.2
# 1000000 19000.2 22133.5 24116.7
# 10000000 18476.1 18430.9 23573.4
# 100000000 18504.8 18057.5 23005.2

# open file with filename from command line, parse the lines, and store the results in a dictionary

with open('solutions/stream_bw.txt', 'r') as f:
    lines = f.readlines()

vec_size = []
copy_rate = []
scale_rate = []
add_rate = []
footprint = []

for line in lines[1:]:
    vec_size.append(int(line.split()[0]))
    copy_rate.append(float(line.split()[1]) / 1024)
    scale_rate.append(float(line.split()[2]) / 1024)
    add_rate.append(float(line.split()[3]) / 1024)
    footprint.append(4 * int(line.split()[0]) * 8)

# plot the results

import matplotlib.pyplot as plt

plt.plot(footprint, copy_rate, '-o', label='copy')
plt.plot(footprint, scale_rate, '-o', label='scale')
plt.plot(footprint, add_rate, '-o', label='add')

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

plt.savefig(f"solutions/stream.png")
