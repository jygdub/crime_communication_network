#!/bin/bash
#SBATCH --job-name="fixed"
#SBATCH --partition=rome
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time 1:00:00 #runtime
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=128


cp $HOME/Graph-Atlas/data-GraphAtlas.tsv /scratch-shared/Graph-Atlas
cp -r $HOME/Graph-Atlas/graphs /scratch-shared/Graph-Atlas
cp $HOME/Graph-Atlas/adjusted_parallel_run.py /scratch-shared/Graph-Atlas
cp $HOME/Graph-Atlas/adjusted_dynamics_vectorized.py /scratch-shared/Graph-Atlas

cd /scratch-shared/Graph-Atlas
module load 2023 
module load Python/3.11.3-GCCcore-12.3.0 # python version

# load required packages not available on Snellius
# pip install --user itertools
pip install --user numpy
pip install --user pandas

python3 adjusted_parallel_run.py # run script