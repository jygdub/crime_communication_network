#!/bin/bash
#SBATCH --job-name="sim-a50-b50"
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time 15:00:00 #runtime
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=192

cp $HOME/Graph-Atlas/data-GraphAtlas.tsv /scratch-shared/Graph-Atlas
cp -r $HOME/Graph-Atlas/graphs /scratch-shared/Graph-Atlas
cp $HOME/Graph-Atlas/run_vectorized_parallel_Atlas.py /scratch-shared/Graph-Atlas
cp $HOME/Graph-Atlas/dynamics_vectorized_Atlas.py /scratch-shared/Graph-Atlas

cd /scratch-shared/Graph-Atlas
module load 2023 
module load Python/3.11.3-GCCcore-12.3.0 # python version

# load required packages not available on Snellius
# pip install --user itertools
pip install --user numpy
pip install --user pandas

python3 run_vectorized_parallel_Atlas.py # run script
# cd results
# zip -r alpha0_50-beta0_50.zip alpha0_50-beta0_50
# cp -r alpha1_00-beta0_50.zip $HOME/Graph-Atlas/results # copy result files in pre-existing 