#!/bin/bash
#
#SBATCH -G 1
#SBATCH -t 2-0
#SBATCH -p gpu,serc
#SBATCH --mem 50GB

cd /home/groups/kmaher/timdai/bandit-segmentation/
source activate cs131
which python

$1
