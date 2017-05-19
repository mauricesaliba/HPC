#!/bin/sh
#
# Your job name
#$ -N nbody_01
#
# Use current working directory
#$ -cwd
#
#
# Run job through bash shell
#$ -S /bin/bash
#$ -o myFile.txt



# If modules are needed, source modules environment:
. /etc/profile.d/modules.sh

# Add any modules you might require:
module add shared openmpi/gcc/64/1.8.8
module add shared mpich2/hydra/gcc/3.2

# The following output will show in the output file
echo "Got $NSLOTS processors."

# Run your application 
OMP_NUM_THREADS=32 ./exec_01.o > output_fileOpenMp.txt
