#!/bin/bash
#SBATCH --job-name=qutrit_rb
#SBATCH --output=qutrit_rb_%A_%a.out
#SBATCH --error=qutrit_rb_%A_%a.err
#SBATCH --array=0-20
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=64GB

# Load necessary modules
module load python

# OpenMP settings for Python
export OMP_NUM_THREADS=32
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Set up output directory
CHECKPOINT_DIR="qutrit_rb_results"
mkdir -p $CHECKPOINT_DIR
SIM_SCRIPT="/global/homes/s/sohan100/qutrit_simulation/"\
"qutrit_logical_rb_logical_noise_sim.py"

# Define error rates array
error_rates=(
    1.00000000e-07
    7.84759970e-06
    3.35981829e-05
    6.15848211e-04
    1.12883789e-02
    2.06130785e-02
    2.33572147e-02
    3.11537409e-02
    3.62021775e-02
    4.20687089e-02
    4.83293024e-02
    5.47144504e-02
    6.35808794e-02
    7.38841056e-02
    8.58569606e-02
    9.25524149e-02
    1.00000000e-01
    1.43844989e-01
    2.06913808e-01
    3.35981829e-01
)

# Print job information
echo "Starting job array ${SLURM_ARRAY_JOB_ID}, task ${SLURM_ARRAY_TASK_ID}"
echo "Running on $(hostname)"
echo "Current working directory: $(pwd)"

# Get the error rate corresponding to this array index
if [ "${SLURM_ARRAY_TASK_ID}" -lt "${#error_rates[@]}" ]; then
    p_val=${error_rates[${SLURM_ARRAY_TASK_ID}]}
    
    echo "Processing error rate: ${p_val}"
    
    # Run the simulation for this error rate and checkpoint directory.
    srun python ${SIM_SCRIPT} \
        --error-rate ${p_val} \
        --checkpoint-dir ${CHECKPOINT_DIR} \
        --reps 10000
    
    echo "Simulation completed for error rate ${p_val}"
else
    # If all individual simulations are done, generate plots from all results
    echo "Generating combined plots from all results"
    srun python ${SIM_SCRIPT} \
        --plot-only \
        --checkpoint-dir ${CHECKPOINT_DIR}
    
    echo "Plot generation completed"
fi

echo "Job completed successfully"
