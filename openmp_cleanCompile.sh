#!/bin/sh

rm -f exec_01.o
rm -f  nbody_01.*
rm -f output_executable.log  
rm -f qsub_job.log
g++ -fopenmp nbody_shared.cpp -o exec_01.o