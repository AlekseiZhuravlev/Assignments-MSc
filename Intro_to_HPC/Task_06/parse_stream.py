# open file with filename from command line
import sys

filename = sys.argv[1]
vec_size = int(sys.argv[2])

with open(filename, 'r') as f:
    lines = f.readlines()

copy_line = lines[29]
scale_line = lines[30]
add_line = lines[31]
triad_line = lines[32]

copy_rate = float(copy_line.split()[1])
scale_rate = float(scale_line.split()[1])
add_rate = float(add_line.split()[1])
triad_rate = float(triad_line.split()[1])

# open file solutions/stream_bw.txt for appending
import os

# check if file 'solutions/stream_bw.txt' exists
if not os.path.exists('solutions/stream_bw.txt'):
    # if not, create it and write header
    with open('solutions/stream_bw.txt', 'w') as f:
        f.write('vec_size copy_rate scale_rate add_rate triad_rate\n')

with open('solutions/stream_bw.txt', 'a') as f:
    f.write(f'{vec_size} {copy_rate} {scale_rate} {add_rate} {triad_rate}\n')
