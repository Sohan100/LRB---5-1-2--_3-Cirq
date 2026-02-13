#!/bin/bash
#SBATCH --job-name=qutrit_rb
#SBATCH --output=qutrit_rb_%j.out
#SBATCH --error=qutrit_rb_%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00
#SBATCH --ntasks=19
#SBATCH --cpus-per-task=1

# Load necessary modules
module load python

# OpenMP settings for Python - using single thread
export OMP_NUM_THREADS=1
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

# Set up output directory
CHECKPOINT_DIR="qutrit_rb_results_alternate"
mkdir -p "${CHECKPOINT_DIR}"
SIM_SCRIPT="/global/homes/s/sohan100/qutrit_simulation/"\
"qutrit_logical_rb_logical_noise_sim.py"

# Print job information
echo "Starting job ${SLURM_JOB_ID}"
echo "Running on $(hostname)"
echo "Current working directory: $(pwd)"

launch_rate() {
    local p_val="$1"
    echo "Launching simulation for error rate: ${p_val}"
    srun --ntasks=1 --cpus-per-task=1 \
        python "${SIM_SCRIPT}" \
        --error-rate "${p_val}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --reps 10000 \
        > "${CHECKPOINT_DIR}/simulation_${p_val}.log" 2>&1 &
}

# Launch all simulations explicitly in parallel
launch_rate "3.35981829e-05"
launch_rate "6.15848211e-04"
launch_rate "1.12883789e-02"
launch_rate "2.06130785e-02"
launch_rate "2.33572147e-02"
launch_rate "3.11537409e-02"
launch_rate "3.62021775e-02"
launch_rate "4.20687089e-02"
launch_rate "4.83293024e-02"
launch_rate "5.47144504e-02"
launch_rate "6.35808794e-02"
launch_rate "7.38841056e-02"
launch_rate "8.58569606e-02"
launch_rate "9.25524149e-02"
launch_rate "1.00000000e-01"
launch_rate "1.43844989e-01"
launch_rate "2.06913808e-01"
launch_rate "3.35981829e-01"

# Wait for all simulations to complete
echo "Waiting for all simulations to complete..."
wait
echo "All simulations have completed."

# Generate plots from results (after all simulations complete)
echo "Generating combined plots from all results"
srun --ntasks=1 --cpus-per-task=1 \
    python "${SIM_SCRIPT}" \
    --plot-only \
    --checkpoint-dir "${CHECKPOINT_DIR}"

echo "Job completed successfully"
