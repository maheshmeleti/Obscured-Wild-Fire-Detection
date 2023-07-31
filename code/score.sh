#!/bin/bash
#
#PBS -N score_cal
#PBS -l select=1:ncpus=16:mem=64gb:ngpus=1:gpu_model=p100:interconnect=10ge,walltime=2:00:00
#PBS -o SCORE.txt
#PBS -j oe

nvidia-smi
cd $PBS_O_WORKDIR
module load anaconda3/2022.05-gcc/9.5.0
source activate pytorch-p100
python score_cal.py