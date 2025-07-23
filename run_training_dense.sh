#!/bin/bash
# Needs to be run from the AutoencoderTraining directory

PROJECT_ROOT="$(pwd -P)"
PARENT_DIR="$(dirname "$PROJECT_ROOT")"

singularity exec --nv \
  --bind "$PROJECT_ROOT" \
  --bind "$(readlink -f $HOME)" \
  --bind "$(readlink -f ${HOME}/nobackup/)" \
  --bind /cvmfs \
  /cvmfs/unpacked.cern.ch/registry.hub.docker.com/fnallpc/fnallpc-docker:tensorflow-2.12.0-gpu-singularity \
  bash -c "cd $PROJECT_ROOT && pwd -P &&export PYTHONPATH=$PARENT_DIR:\$PYTHONPATH && python training/dense_autoencoder/train.py"
