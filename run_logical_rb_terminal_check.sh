#!/bin/bash
#SBATCH --job-name=qutrit_rb_terminal_check_parallel
#SBATCH --output=qutrit_rb_terminal_check_parallel_%j.out
#SBATCH --error=qutrit_rb_terminal_check_parallel_%j.err
#SBATCH -C cpu
#SBATCH -q regular
#SBATCH -t 47:30:00

# --- BEGIN USER CONFIGURABLE SECTION ---
#SBATCH --nodes=20
#SBATCH --ntasks=20
#SBATCH --cpus-per-task=32

CHECKPOINT_DIR="qutrit_rb_results_terminal_check_local_noise"
SCRIPTS_DIR="/global/homes/s/sohan100/qutrit_simulation"
SIM_SCRIPT="${SCRIPTS_DIR}/qutrit_logical_rb_terminal_check_sim.py"
REPS=10000
# --- END USER CONFIGURABLE SECTION ---

PROBABILITIES=(
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
NUM_PROBS=${#PROBABILITIES[@]}

mkdir -p "${CHECKPOINT_DIR}"
LOG_DIR="${CHECKPOINT_DIR}/logs_job_${SLURM_JOB_ID}"
mkdir -p "${LOG_DIR}"

# Load necessary modules
module load python

# OpenMP settings for Python
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OMP_PLACES=threads
export OMP_PROC_BIND=spread

echo "-----------------------------------------------------"
echo "Job Configuration:"
echo "Checkpoint dir: ${CHECKPOINT_DIR}"
echo "SIM script: ${SIM_SCRIPT}"
echo "Probabilities: ${PROBABILITIES[*]}"
echo "Number of probabilities: ${NUM_PROBS}"
echo "Slurm tasks requested: ${SLURM_NTASKS}"
echo "CPUs per srun task: ${SLURM_CPUS_PER_TASK}"
echo "Log directory: ${LOG_DIR}"
echo "-----------------------------------------------------"

echo "Starting job ${SLURM_JOB_ID} on $(hostname)"
echo "Current working directory: $(pwd)"

for idx in $(seq 0 $((${NUM_PROBS} - 1))); do
    p_val="${PROBABILITIES[idx]}"
    echo "Launching simulation for p=${p_val} (index ${idx})"

    srun --exclusive --nodes=1 --ntasks=1 \
        --cpus-per-task=${SLURM_CPUS_PER_TASK} \
        python "${SIM_SCRIPT}" \
        --error-rate "${p_val}" \
        --checkpoint-dir "${CHECKPOINT_DIR}" \
        --reps "${REPS}" \
        > "${LOG_DIR}/simulation_p${p_val}.log" 2>&1 &
done

echo "Waiting for all ${NUM_PROBS} simulations to complete..."
wait
echo "All simulations have completed."
