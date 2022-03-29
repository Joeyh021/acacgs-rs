#!/bin/sh

#This script requires at lease a singular parameter, an executable. 
#This should be the first parameter. 
#Any further parameters should be the ones required by the executable itself,
#  as these will get passed to the program when ran by the compute node.

if [ $# -lt 1 ]; then
	echo "Please provide a path to the executable you want to run"
	exit 1
fi

if [ -x "$1" ]; then
	echo "Program: $@"
else
	echo "The file provided is not an executable. Please provide an executable"
	exit 1
fi

echo "#!/bin/sh

#SBATCH --job-name=$( whoami )-$1
#SBATCH --partition=cs257
#SBATCH --account=cs257users
#SBATCH --cpus-per-task=6
#SBATCH --time=10:00
#SBATCH --mail-type=END,FAIL,TIME_LIMIT_80
#SBATCH --output=%j/cs257_output_%j.out
#SBATCH --error=%j/cs257_error_%j.err
echo ===== ENVIRONMENT =====
. /etc/profile.d/modules.sh
module load gcc9
module load cs257-silo

lscpu

export OMP_NUM_THREADS=6
export OMP_PLACES=cores
export OMP_PROC_BIND=true
export OMP_WAIT_POLICY=active

echo
" > tmp

if [ -f "Makefile" ]; then
	echo "echo ===== COMPILING Makefile IN $( pwd ) =====
make clean
make

echo
" >> tmp
fi

echo "echo ===== RUNNING $@ =====
srun $@" >> tmp

BATCH=$( sbatch tmp )
BATCHNO=$( echo $BATCH | sed 's/[^0-9]//g' )

rm tmp

mkdir $BATCHNO

echo "===== Job $BATCHNO has been submitted! ====="

echo "===== My jobs ====="
echo "Note: Don't worry if the job is PENDING, the job will be ran as soon as possible."
squeue -u $( whoami ) -o "%.8i %.20j %.10T %.5M %.20R %.20e"
echo $OMP_DISPLAY_ENV
#ENDTIME=$( squeue -o "%i %e" | grep $BATCHNO | cut -d " " -f 2 )

#echo "===== Job $BATCHNO has been submitted, should be finished by $ENDTIME ====="
