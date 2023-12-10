# get all txt files in solutions/outputs_stream_omp

import matplotlib.pyplot as plt

n_threads = [2 ** i for i in range(9)]
for n_th in n_threads:
    with open(f'solutions/outputs_stream_omp/{n_th}/stream_bw.txt', 'r') as f:
        lines = f.readlines()

    vec_size = []
    triad_rate = []
    footprint = []

    for line in lines[1:]:
        vec_size.append(int(line.split()[0]))
        triad_rate.append(float(line.split()[4]) / 1024)
        footprint.append(4 * int(line.split()[0]) * 8)

    plt.plot(footprint, triad_rate, '-o', label=f'{n_th} threads')

plt.title("Triad rate")

plt.xlabel("Memory footprint [bytes]")
plt.ylabel("Memory bandwidth [GB/s]")

plt.xscale("log")
plt.yscale("log")

plt.savefig('solutions/triad_omp.png')
