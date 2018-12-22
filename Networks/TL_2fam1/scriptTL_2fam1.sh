#!/bin/bash -l
#SBATCH --account cadmos
#SBATCH --mail-type ALL
#SBATCH --mail-user leonard.jequier@unil.ch
#SBATCH --workdir ./
#SBATCH --job-name TL_2fam_t1
#SBATCH --output TL_2fam_t1.out
#SBATCH --partition serial 
#SBATCH --nodes 1
#SBATCH --ntasks 1
#SBATCH --cpus-per-task 8
#SBATCH --mem 10G
#SBATCH --time 64:00:00

module load gcc/7.3.0 python/3.6.5

source $HOME/PyTorch_CPU_Py3/bin/activate

srun python TL_net.py
deactivate
