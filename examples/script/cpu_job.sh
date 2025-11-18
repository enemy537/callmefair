#!/bin/bash
#SBATCH --job-name=cpu_callmefair
#SBATCH --nodes=1
#SBATCH --cpus-per-task=192
#SBATCH --time=0-00:20:00
#SBATCH --output=cpu_test_%j.out

cd $SLURM_SUBMIT_DIR

# Print CPU info for debugging
echo "Allocated CPUs: ${SLURM_CPUS_PER_TASK}"
echo "Node: ${SLURMD_NODENAME}"
lscpu | grep -E "^CPU\(s\)|Thread|Core|Socket"

module load StdEnv/2023 python/3.12 openblas

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1

virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index --no-cache-dir -r callmefair/requirements.txt
pip install --no-index --no-cache-dir --find-links=./local-wheels aif360[Reductions,inFairness]

echo "Starting bias search with ${SLURM_CPUS_PER_TASK} threads..."

python callmefair/examples/script/bias_search_example.py \
    --dataset callmefair/examples/data/diabetes_pre/diabetes.csv \
    --label readmitted \
    --attributes age,gender,race \
    --n_threads ${SLURM_CPUS_PER_TASK:-192}