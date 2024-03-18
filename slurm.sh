#!/bin/bash
#
#SBATCH -t 2-0
#SBATCH -p kmaher,normal

cd /home/groups/kmaher/timdai/bandit-segmentation/
source activate cs131
which python

$1
