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



# --- TODO: Print run conditions in the header of your output file: solutions/daxpy.txt
echo "DAXPY daxpy.c version:2023" > solutions/daxpy.txt 
echo "gcc −O1 −fopt−info" > solutions/daxpy.txt
echo "gcc (GCC) 11.3.0" > solutions/daxpy.txt
echo "Scalar Version (no OpenMP or MPI)" > solutions/daxpy.txt
echo "Run: Node: jrc0667; ntasks: 1" > solutions/daxpy.txt
echo "=====================================" > solutions/daxpy.txt
echo "      Size      BW[GB/s]      Runtime[ms]" > solutions/daxpy.txt

for ((i=$min; i<=$max; i=i*10))
# run bin/daxpy for increasing vector lengths
do
srun -n 1 bin/daxpy $i >> solutions/daxpy.txt
done

# --- TODO: Use srun to run daxpy for increasing vector lengths 
#  and store its output in the outputfile: solutions/daxpy.txt
  
