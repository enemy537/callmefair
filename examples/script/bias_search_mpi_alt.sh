#!/bin/bash
#SBATCH --job-name=bias_search_mpi_alt
#SBATCH --nodes=4                    # 4 nodes as requested
#SBATCH --ntasks-per-node=16         # 16 MPI ranks per node (64 total ranks)
#SBATCH --cpus-per-task=12           # 12 CPU cores per rank (16*12=192 cores per node)
#SBATCH --time=0-04:00:00            # 4 hours max runtime
#SBATCH --output=bias_search_mpi_alt_%j.out
#SBATCH --error=bias_search_mpi_alt_%j.err
#SBATCH --mem=0                      # Use all available memory per node
#SBATCH --exclusive                  # Exclusive node access for optimal performance

# Alternative configuration with more MPI ranks and fewer threads per rank
# This configuration may be better for workloads with many independent models
# or when memory per rank is a limiting factor

# Change to submission directory
cd $SLURM_SUBMIT_DIR

# Create temporary directories for matplotlib and other configs
export MPLCONFIGDIR=$SLURM_TMPDIR/matplotlib
export FONTCONFIG_PATH=$SLURM_TMPDIR/fontconfig
export TF_ENABLE_ONEDNN_OPTS=0
mkdir -p $MPLCONFIGDIR $FONTCONFIG_PATH

# Load required modules (adjust based on your cluster)
module load StdEnv/2023 python/3.12 openblas mpi4py

# Create and activate virtual environment
virtualenv --no-download ENV
source ENV/bin/activate

# Install dependencies
pip install --no-index --upgrade pip
pip install --no-index --no-cache-dir -r callmefair/requirements.txt
pip install --no-index --no-cache-dir --find-links=./local-wheels aif360[Reductions,inFairness] BlackBoxAuditing

# Set up MPI environment
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=29500
export WORLD_SIZE=$SLURM_NTASKS
export RANK=$SLURM_PROCID

# Print job information
echo "========================================="
echo "SLURM Job Information (Alternative Config)"
echo "========================================="
echo "Job ID: $SLURM_JOB_ID"
echo "Number of nodes: $SLURM_JOB_NUM_NODES"
echo "Node list: $SLURM_JOB_NODELIST"
echo "Tasks per node: $SLURM_NTASKS_PER_NODE"
echo "CPUs per task: $SLURM_CPUS_PER_TASK"
echo "Total tasks: $SLURM_NTASKS"
echo "Total CPUs: $((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))"
echo "Master node: $MASTER_ADDR"
echo "World size: $WORLD_SIZE"
echo "Configuration: More MPI ranks, fewer threads per rank"
echo "========================================="

# Create a hostfile for MPI
scontrol show hostnames "$SLURM_JOB_NODELIST" > hostfile_$SLURM_JOB_ID

# Set Python path and threading environment variables
export PYTHONPATH=$PYTHONPATH:$(pwd)
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK
export NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK
export VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK

# Disable CPU frequency scaling for consistent performance
export OMP_PROC_BIND=true
export OMP_PLACES=cores

# Default parameters
DATASET=${DATASET:-"callmefair/examples/data/diabetes_pre/diabetes.csv"}
LABEL=${LABEL:-"readmitted"}
ATTRIBUTES=${ATTRIBUTES:-"age,gender,race"}
MODELS=${MODELS:-"lr,mlp,xgb,cat,rf"}  # Added random forest for more models
ITERATIONS=${ITERATIONS:-20}
OUTPUT_DIR=${OUTPUT_DIR:-"./mpi_results_alt_$SLURM_JOB_ID"}

echo "Bias Search Parameters (Alternative):"
echo "Dataset: $DATASET"
echo "Label: $LABEL"
echo "Attributes: $ATTRIBUTES"
echo "Models: $MODELS"
echo "Iterations: $ITERATIONS"
echo "Output Directory: $OUTPUT_DIR"
echo "Threads per rank: $SLURM_CPUS_PER_TASK"
echo "========================================="

# Create output directory
mkdir -p $OUTPUT_DIR

# Run the MPI bias search script
echo "Starting Alternative MPI Bias Search at $(date)"
echo "========================================="

srun --cpu-bind=cores --kill-on-bad-exit=1 \
     --export=ALL,OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK,OPENBLAS_NUM_THREADS=$SLURM_CPUS_PER_TASK,MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK,NUMEXPR_NUM_THREADS=$SLURM_CPUS_PER_TASK,VECLIB_MAXIMUM_THREADS=$SLURM_CPUS_PER_TASK \
     python callmefair/examples/script/bias_search_example_mpi.py \
     --dataset "$DATASET" \
     --label "$LABEL" \
     --attributes "$ATTRIBUTES" \
     --models "$MODELS" \
     --iterations $ITERATIONS \
     --threads_per_rank $SLURM_CPUS_PER_TASK \
     --output_dir "$OUTPUT_DIR" \
     --combinations_model "lr" \
     --log_file "$OUTPUT_DIR/bias_search_mpi_alt_$SLURM_JOB_ID.log"

echo "========================================="
echo "Alternative MPI Bias Search completed at $(date)"

# Performance summary
echo "========================================="
echo "Performance Summary (Alternative Config):"
echo "Total nodes used: $SLURM_JOB_NUM_NODES"
echo "Total MPI ranks: $SLURM_NTASKS"
echo "Total CPU cores: $((SLURM_JOB_NUM_NODES * SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))"
echo "Cores per node: $((SLURM_NTASKS_PER_NODE * SLURM_CPUS_PER_TASK))"
echo "MPI ranks per node: $SLURM_NTASKS_PER_NODE"
echo "Threads per MPI rank: $SLURM_CPUS_PER_TASK"
echo "Strategy: More ranks for better model distribution"
echo "========================================="

# List output files
echo "Output files generated:"
ls -la $OUTPUT_DIR/

# Cleanup
rm -f hostfile_$SLURM_JOB_ID

echo "Alternative configuration job completed successfully!"