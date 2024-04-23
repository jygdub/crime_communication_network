#!/bin/bash
#SBATCH --job-name="graph-trans"
#SBATCH --partition=genoa
#SBATCH --nodes=1
#SBATCH --exclusive
#SBATCH --time 1:00:00 #runtime
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=192

cp $HOME/Graph-Atlas/data-GraphAtlas.tsv /scratch-shared/Graph-Atlas
cp -r $HOME/Graph-Atlas/graphs /scratch-shared/Graph-Atlas
cp $HOME/Graph-Atlas/run_transitions_parallel_Atlas.py /scratch-shared/Graph-Atlas

cd /scratch-shared/Graph-Atlas
module load 2023 
module load Python/3.11.3-GCCcore-12.3.0 # python version

# load required packages not available on Snellius
pip install --user numpy
pip install --user pandas
pip install --user scipy

python3 run_transitions_parallel_Atlas.py # run script

cp from_graph_n=7.tsv $HOME/Graph-Atlas/
cp to_graph_n=7.tsv $HOME/Graph-Atlas/