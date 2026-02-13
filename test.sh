#!/bin/bash
#SBATCH --job-name=qutrit_test
#SBATCH --output=qutrit_test_%A_%a.out
#SBATCH --error=qutrit_test_%A_%a.err
#SBATCH --array=0
#SBATCH -C cpu
#SBATCH -q regular  
#SBATCH -t 1:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=8GB

# Load necessary modules
module load python

# Set up output directory
CHECKPOINT_DIR="qutrit_test_results"
mkdir -p $CHECKPOINT_DIR
SIM_SCRIPT="/global/homes/s/sohan100/qutrit_simulation/"\
"qutrit_logical_rb_logical_noise_sim.py"

# Select a single, low error rate for testing
p_val=1.00000000e-07

echo "Starting test job for error rate: ${p_val}"
echo "Running on $(hostname)"
echo "Current working directory: $(pwd)"

# Run the simulation with minimal parameters
srun python ${SIM_SCRIPT} \
    --error-rate ${p_val} \
    --checkpoint-dir ${CHECKPOINT_DIR} \
    --reps 100

echo "Test simulation completed"
