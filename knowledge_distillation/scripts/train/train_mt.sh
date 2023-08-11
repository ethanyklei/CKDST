#!/usr/bin/env bash

set -u 
set -eo pipefail

CONFIG_NAME=$1
src=en
tgt=$2

REPO_ROOT=/path/to/repo
USER_DIR=${REPO_ROOT}/knowledge_distillation
DATA_ROOT=/path/to/mt/bin/data
CONFIG_DIR=${USER_DIR}/config/mt

save_dir=/path/to/save/model
save_dir=${save_dir}/"\${task._name}-\${model._name}-\${criterion._name}-\${optimization.update_freq}-\${optimization.lr}-\${now:%Y-%m-%d}-\${now:%H-%M-%S}"

fairseq-hydra-train \
    task.data=${data_root} \
    checkpoint.save_dir=${save_dir} \
    --config-dir ${config_root} \
    --config-name ${CONFIG_NAME}