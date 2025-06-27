#!/bin/bash -l
#SBATCH --job-name gk_tcv_3x2v_dbg
#SBATCH --qos debug
#SBATCH --nodes 1
#SBATCH --tasks-per-node=4
#SBATCH --time 00:30:00
#SBATCH --constraint gpu
#SBATCH --gpus 4
#SBATCH --mail-user=ahoffman@pppl.gov
#SBATCH --mail-type=END,FAIL,REQUEUE
#SBATCH --output ../history/output_sf_0.out
#SBATCH --error ../history/error_sf_0.out
#SBATCH --account m2116
module load PrgEnv-gnu/8.5.0 craype-accel-nvidia80 cray-mpich/8.1.28 cudatoolkit/12.0 nccl/2.18.3-cu12
export MPICH_GPU_SUPPORT_ENABLED=0
export DVS_MAXNODES=32_
export MPICH_MPIIO_DVS_MAXNODES=32
srun -u -n 4 ./g0 -g -M -c 1 -d 1 -e 4 
exit 0
