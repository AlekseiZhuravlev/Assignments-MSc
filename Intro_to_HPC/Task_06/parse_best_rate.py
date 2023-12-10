base_dir = 'solutions/outputs_stream_omp'

# for each directory in base_dir
import os

import numpy as np

for omp_threads in os.listdir(base_dir):
    out_file = f'solutions/outputs_stream_omp/{omp_threads}/stream_bw.txt'

    # parse the file, format:
    # vec_size copy_rate scale_rate add_rate
    # 10000 95.4 95.1 142.9
    # 100000 931.9 929.7 1364.4
    # 1000000 6267.2 6397.4 7756.5
    # 10000000 14858.9 14812.4 18298.4
    # 100000000 19116.1 18790.4 21878.4

    with open(out_file, 'r') as f:
        lines = f.readlines()

    vec_size = []
    copy_rate = []
    scale_rate = []
    add_rate = []

    for line in lines[1:]:
        vec_size.append(int(line.split()[0]))
        copy_rate.append(float(line.split()[1]))
        scale_rate.append(float(line.split()[2]))
        add_rate.append(float(line.split()[3]))

    # get the best rates for each operation
    best_copy_rate = np.max(copy_rate)
    best_scale_rate = np.max(scale_rate)
    best_add_rate = np.max(add_rate)

    with open(f'solutions/outputs_stream_omp/stream_omp{omp_threads}.txt', 'w') as f:
        f.write('copy_rate scale_rate add_rate\n')
        f.write(f'{best_copy_rate} {best_scale_rate} {best_add_rate}\n')
