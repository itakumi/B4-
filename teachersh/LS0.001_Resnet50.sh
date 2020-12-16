#!/bin/bash
#$ -cwd
#$ -l rt_F=32
#$ -j y
#$ -l h_rt=05:00:00
#$ -o output/o.$JOB_ID

# ======== Pyenv/ ========
export PYENV_ROOT=$HOME/.pyenv
export PATH=$PYENV_ROOT/bin:$PATH
eval "$(pyenv init -)"

# ======== Modules ========
source /etc/profile.d/modules.sh
module load cuda/10.2/10.2.89 cudnn openmpi nccl/2.7/2.7.8-1

export LD_LIBRARY_PATH=~/.pyenv/versions/pytorch_distributed_tutorials/lib/python3.8/site-packages/torch/lib/:$LD_LIBRARY_PATH
# For example: ~/.pyenv/versions/hoge/lib/python3.8/site-packages/torch/lib/:$LD_LIBRARY_PATH

# For ABCI
export MASTER_ADDR=$(/usr/sbin/ip a show dev bond0 | grep inet | cut -d " " -f 6 | cut -d "/" -f 1)
export ROOT=/groups2/gaa50004/data/ILSVRC2012

export NGPUS=128
mpirun -npernode 4 -np $NGPUS \
    -x LD_LIBRARY_PATH \
python LabelSmoothing.py --epsilon 0.001\
  --dist-url $MASTER_ADDR \
  -a resnet50 \
  $ROOT
