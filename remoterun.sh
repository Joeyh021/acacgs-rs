#!/bin/sh
cargo build --release
echo "#!/bin/sh

#SBATCH --job-name=$( whoami )-$1
#SBATCH --partition=cs257
#SBATCH --account=cs257users
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00
#SBATCH --output=%j/output_%j.out
#SBATCH --error=%j/error_%j.err


echo
" > tmp


echo "echo ===== RUNNING acacgs-rs $@ =====
srun target/release/acacgs-rs $@" >> tmp

BATCH=$( sbatch tmp )
BATCHNO=$( echo $BATCH | sed 's/[^0-9]//g' )

rm tmp

mkdir $BATCHNO
echo "Running acacgs-rs $@..."

echo "===== Job $BATCHNO has been submitted! ====="

echo "===== My jobs ====="
echo "Note: Don't worry if the job is PENDING, the job will be ran as soon as possible."
squeue -u $( whoami ) -o "%.8i %.20j %.10T %.5M %.20R %.20e"
echo $OMP_DISPLAY_ENV
