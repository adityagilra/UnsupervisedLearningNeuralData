#!/bin/bash
#
#----------------------------------
# single GPU + single CPU example
#----------------------------------
#
#SBATCH --job-name=nengo_DL
# only error gets redirected here somehow -- so redirecting stdout on command line below
#SBATCH --output=nohups/nohup_gpu_%a.txt
#
#number of CPUs to be used
#SBATCH --ntasks=1
#
#Define the number of hours the job should run. 
#Maximum runtime is limited to 10 days, ie. 240 hours
#SBATCH --time=240:00:00
#
#Define the amount of system RAM used by your job in GigaBytes
##SBATCH --mem=5G
#SBATCH --mem=24G
#
##Send emails when a job starts, it is finished or it exits
##SBATCH --mail-user=YourEmail@ist.ac.at
##SBATCH --mail-type=ALL
#
#Pick whether you prefer requeue or not. If you use the --requeue
#option, the requeued job script will start from the beginning, 
#potentially overwriting your previous progress, so be careful.
#For some people the --requeue option might be desired if their
#application will continue from the last state.
#Do not requeue the job in the case it fails.
#SBATCH --no-requeue
#
#Define the "gpu" partition for GPU-accelerated jobs
##SBATCH --partition=gpu
##SBATCH --partition=gpu10cards
#SBATCH --partition=defaultp
##SBATCH --partition=bigmem
#
##Define the number of GPUs used by your job
##SBATCH --gres=gpu:1
#
## Define the GPU architecture (GTX980 in the example, other options are GTX1080Ti, K40)
##  actually gpu10cards partition has GTX2080Ti-s, but I don't know their constraint name,
##   sometimes I've got 2080Ti if partition gpu10cards is chosen and maybe 1080Ti-s are full?
##SBATCH --constraint=GTX1080Ti
#
#Do not export the local environment to the compute nodes
#SBATCH --export=NONE
unset SLURM_EXPORT_ENV
#
#for single-CPU jobs make sure that they use a single thread
#export OMP_NUM_THREADS=1
#
#load a CUDA software module

# below is the tf_gpu conda environment that has nengo, nengo-dl and tensorflow-gpu:
module load cuda/10.0                   # don't use 10.1 as my pip installed tensorflow-gpu wants 10.0
module load cudnn/7.5
#module load python                      # has nengo and nengo-dl -- latest default 3.6 versions don't have
#module load python/3.5-gpu              # has nengo and nengo-dl -- doesn't import nengolib as scipy.misc.pade gives import error due to new scipy
module load python/2.7.13-gpu           # has nengo and nengo-dl, and imports nengolib (after pip install --user nengolib)
module load boost                       # loads v1.70.0, needed for boost::python interface to EMBasins.cpp, see Makefile
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/mnt/nfs/clustersw/Debian/stretch/boost/1.70.0/lib/
                                        # older Ubuntu-default boost library is used otherwise
export PYTHONPATH=~/:$PYTHONPATH        # set where you have TreeHMM repo cloned and compiled
#/nfs/scistore12/gaspgrp/agilra/anaconda3/bin/conda activate tf_gpu
                                        # on remote node, must point to my local conda
                                        # this worked once, but now gives earlier error to first do `conda init bash`

# any of the three modules below have nengo, nengo-dl and tensorflow-gpu:
#module load python/2.7.13-gpu
#module load python/3.5-gpu
#module load tensorflow/python3/1.13.1

#
## print out the list of GPUs before the job is started
#/usr/bin/nvidia-smi
#
# run with `sbatch -p gpu10cards submit_L2L.sbatch` or `sbatch -p gpu submit_L2L.sbatch`
# actually gpu partition is already defined above! dunno if commandline one overrides it?
# `sbatch submit_L2L.sbatch` is enough
# squeue -u agilra                       # list my jobs, status SD means scheduled, R means running
# squeue -p gpu10cards                   # list jobs running in the specified partition
# scontrol show job <jobid>              # give details above jobid
#
#run your CUDA binary through SLURM's srun
#srun --cpu_bind=verbose `python test.py`
# still need to give output redirects, slurm/sbatch uses #SBATCH --output=... above only for warnings/errors.
srun --cpu_bind=verbose `python EMBasins_sbatch.py $SLURM_ARRAY_TASK_ID > nohups/nohup$SLURM_ARRAY_TASK_ID.txt`
