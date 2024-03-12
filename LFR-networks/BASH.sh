#!/bin/bash
#SBATCH --job-name="newt"
#SBATCH --partition genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time 1:00:00 #runtime
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=192
set -x
cp -r $HOME/Neutral_theory/scratch-shared/
cp $HOME/netural_theory/master_iter.py/scratch-shared/Netural_theory
cp $HOME/Neutral_theory/one-run.py/scratch-shared/Neutral_theory

cd /scratch-shared/Neutral_theory
module load 2022
module load Python/3.10.4-GCCorde-11.3.0

pip install --user networkx # load required packages not available on Snellius

python3 master_iter.py
cp res* $HOME/Neutral_theory/results/
