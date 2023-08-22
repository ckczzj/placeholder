#!/bin/bash
# This script is for 1) Install dependencies; 2) Align internal cluster with standard practice

set +e

# If the cluster is in US, then proxy is not needed.
if [[ $ARNOLD_MONITOR_CLUSTER != cloudnative-maliva ]]; then
    export http_proxy=http://sys-proxy-rd-relay.byted.org:3128 https_proxy=http://sys-proxy-rd-relay.byted.org:8118 no_proxy=.byted.org
fi

if [ ! -d "$HOME/miniconda3" ]; then
  # use miniconda installer
  cp ./miniconda3.sh $HOME/miniconda3.sh

  # Run Anaconda installer. -b flag means run in batch mode (silent mode), -p flag sets installation location
  bash $HOME/miniconda3.sh -b -p $HOME/miniconda3

  # Delete the installer file
  rm $HOME/miniconda3.sh

  $HOME/miniconda3/bin/conda init bash

  # Add miniconda to PATH manually for the current session
  export PATH="$HOME/miniconda3/bin:$PATH"

  # Activate conda
  source $HOME/miniconda3/bin/activate

  # Update conda
  conda update -y -n base -c defaults conda

  # To make the changes permanent, add the export command to your .bashrc file
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> $HOME/.bashrc

  #shellcheck source="/home/tiger/.bashrc"
  source $HOME/.bashrc

  echo 'Successfully installed miniconda...'
  echo -n 'Conda version: '
  conda --version
  echo -e '\n'
fi

if [[ $ARNOLD_MONITOR_CLUSTER != cloudnative-maliva ]]; then
    unset http_proxy && unset https_proxy && unset no_proxy
fi


# ----------------------------------------------------------------------------------------
# setup environment variables
# disable TF verbose logging
TF_CPP_MIN_LOG_LEVEL=2
# fix known issues for pytorch-1.5.1 accroding to
# https://blog.exxactcorp.com/pytorch-1-5-1-bug-fix-release/
MKL_THREADING_LAYER=GNU
# set NCCL envs for disributed communication
NCCL_IB_GID_INDEX=3
NCCL_IB_DISABLE=0
NCCL_DEBUG=INFO
ARNOLD_FRAMEWORK=pytorch
# get distributed training parameters
METIS_WORKER_0_HOST=${METIS_WORKER_0_HOST:-"127.0.0.1"}
NV_GPUS=$(nvidia-smi --query-gpu=name --format=csv,noheader | wc -l)
ARNOLD_WORKER_GPU=${ARNOLD_WORKER_GPU:-$NV_GPUS}
ARNOLD_WORKER_NUM=${ARNOLD_WORKER_NUM:-1}
ARNOLD_ID=${ARNOLD_ID:-0}
ARNOLD_PORT=${METIS_WORKER_0_PORT:-3343}


export NNODES=$ARNOLD_WORKER_NUM
export NODE_RANK=$ARNOLD_ID
export MASTER_ADDR=$METIS_WORKER_0_HOST
export MASTER_PORT=$ARNOLD_PORT
export GPUS=$ARNOLD_WORKER_GPU
