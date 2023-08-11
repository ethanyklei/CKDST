#!/usr/bin/env bash

set -u 
set -eo pipefail

CONFIG_NAME=$1
dataset=mustc
SRC=en
TGT=$2

REPO_ROOT=/path/to/repo
USER_DIR=${REPO_ROOT}/knowledge_distillation
CONFIG_DIR=${USER_DIR}/config/st/${dataset}-${SRC}-${TGT}

DATA_ROOT=/path/to/mustc/data
SAVE_DIR=/path/to/save/model
SAVE_DIR=${SAVE_DIR}/"\${task._name}-\${model._name}-\${criterion._name}-\${now:%Y-%m-%d}-\${now:%H-%M-%S}"

LOG_DIR=${USER_DIR}/log/starfire/`date +%Y-%m-%d-%H-%M-%S`
mkdir -p ${LOG_DIR}

wav2vec_path=/path/to/wav2vec/model


fairseq-hydra-train \
    common.user_dir=${USER_DIR} \
    checkpoint.save_dir=${SAVE_DIR} \
    task.data=${DATA_ROOT} \
    model.w2v_cfg.w2v_path=${wav2vec_path} \
    hydra.run.dir=${LOG_DIR} \
    --config-dir ${CONFIG_DIR} \
    --config-name ${CONFIG_NAME} 2>&1 | tee ${LOG_DIR}/train.log