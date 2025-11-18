#!/bin/bash
#SBATCH --job-name=multinode_ml_training
#SBATCH --nodes=2                    # Number of nodes
#SBATCH --ntasks-per-node=6          # MPI ranks per node (adjust to your cluster)
#SBATCH --cpus-per-task=32           # CPU cores per rank (adjust to node core count)
#SBATCH --time=0-01:00:00
#SBATCH --output=mnode_test_%j.out
#SBATCH --error=multinode_training_%j.err

cd $SLURM_SUBMIT_DIR
# Create a temporary writable directory for matplotlib config
# Environment fixes
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
export FONTCONFIG_PATH=$SLURM_TMPDIR/fontconfig
export TF_ENABLE_ONEDNN_OPTS=0
mkdir -p $MPLCONFIGDIR $FONTCONFIG_PATH

module load StdEnv/2023 python/3.12 openblas mpi4py

virtualenv --no-download ENV
source ENV/bin/activate
pip install --no-index --upgrade pip
pip install --no-index --no-cache-dir -r callmefair/requirements.txt
pip install --no-index --no-cache-dir --find-links=./local-wheels aif360[Reductions,inFairness] BlackBoxAuditing

# Set up the environment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Print job info
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"

# Create a hostfile for MPI (optional, for debugging)
scontrol show hostnames "$SLURM_JOB_NODELIST" > hostfile_$SLURM_JOB_ID

# Set Python path and other environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

# Run the distributed training script
# Bind each rank to distinct cores and propagate thread limits
srun --cpu-bind=cores --kill-on-bad-exit=1 \
     --export=ALL,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK,OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK,MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK,NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK,VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK \
     python callmefair/examples/script/hfgsearch_mpi.py --dataset callmefair/examples/data/diabetes_pre/diabetes.csv \
     --label readmitted --sensitive age --models all

# Cleanup
rm -f hostfile_$SLURM_JOB_ID

echo "Training completed on $(date)"