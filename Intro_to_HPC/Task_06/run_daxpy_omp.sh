#!/bin/bash

#SBATCH --account=training2325
#SBATCH --time=00:05:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --partition=dc-cpu-devel


# --- TODO: Set vector length limits
min=1000
max=100000000



# --- TODO: Print run conditions in the header of your output file: solutions/daxpy_omp.txt
# echo "DAXPY daxpy.c version:2023" > solutions/daxpy_omp.txt 
# echo "gcc −O1 −fopt−info" > solutions/daxpy_omp.txt
# echo "gcc (GCC) 11.3.0" > solutions/daxpy_omp.txt
# echo "OpenMP Version" > solutions/daxpy_omp.txt
# echo "Run: Node: jrc0667; ntasks: 1" > solutions/daxpy_omp.txt
# echo "=====================================" > solutions/daxpy_omp.txt
# echo "      Size      BW[GB/s]      Runtime[ms]" > solutions/daxpy_omp.txt

# run daxpy for increasing number of OpenMP threads, starting by 1 and ending by the maximum number of hardware threads supported by the CPU


# make directory solutions/$j if it does not exist
mkdir -p solutions

mkdir -p solutions/omp



for ((j=1; j<=128; j=j*2))
do
export OMP_NUM_THREADS=$j

# create file daxpy_omp<ThreadNum>.txt
echo "DAXPY daxpy.c version:2023" > solutions/daxpy_omp$j.txt
echo "gcc −O1 −fopt−info" > solutions/daxpy_omp$j.txt
echo "gcc (GCC) 11.3.0" > solutions/daxpy_omp$j.txt
echo "OpenMP Version" > solutions/daxpy_omp$j.txt
echo "Run: Node: jrc0667; ntasks: 1" > solutions/daxpy_omp$j.txt
echo "=====================================" > solutions/daxpy_omp$j.txt

for ((i=$min; i<=$max; i=i*10))
# run bin/daxpy for increasing vector lengths
do



srun -n 1 bin/daxpy $i > solutions/daxpy_omp$j.txt
python solutions/plot_daxpy.py daxpy_omp$j.txt
done

echo "Done with $j threads"
done

# --- TODO: Use srun to run daxpy for increasing vector lengths 
#  and store its output in the outputfile: solutions/daxpy_omp.txt
  
