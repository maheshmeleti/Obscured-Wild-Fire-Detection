#!/bin/bash
#
#PBS -N trainer
#PBS -l select=1:ncpus=16:mpiprocs=16:ngpus=1:gpu_model=a100:mem=64gb:interconnect=hdr,walltime=10:00:00
#PBS -o infer_logs.txt
#PBS -j oe

nvidia-smi
cd $PBS_O_WORKDIR
module load anaconda3/2022.05-gcc/9.5.0
source activate pytorch-p100
python infer.py