#!/bin/sh
#
# Your job name
#$ -N MPI_Job
#
# Use current working directory
#$ -cwd
#
# pe (Parallel environment) request. Set your number of processors here.
#$ -pe openmpi 4
#
# Run job through bash shell
#$ -S /bin/bash

# If modules are needed, source modules environment:
. /etc/profile.d/modules.sh

# Add any modules you might require:
module add shared openmpi/gcc/64/1.8.8

# The following output will show in the output file
echo "Got $NSLOTS processors."

# Run your application
mpirun -np $NSLOTS ./mpi_nbody.o > output_mpi.txt
