# !/usr/bin/env bash 

set -u 
set -eo pipefail

REPO_ROOT=/path/to/repo
USER_DIR=${REPO_ROOT}/knowledge_distillation

EXP_NAME=$1

DATASET=$2
src=$3
tgt=$4

BIN_ROOT=/path/to/mt/bin/data
MODEL_ROOT=/path/to/mt/model
SPM_MODEL=/path/to/spm/model
LOG_DIR=${USER_DIR}/log/test/mt/${DATASET}-${src}-${tgt}/${EXP_NAME}

mkdir -p ${LOG_DIR}

gen_subset=test

upper_bound=$5
average_num=$6
if [[ ${upper_bound} -gt 1000 ]]; then 
    average_type=update
else
    average_type=epoch
fi 

cp_name=average_${average_type}_${average_num}_${upper_bound}

if [[ ${average_type} == 'update' ]]; then 
    ave_args="--num-update-checkpoints ${average_num}"
elif [[ ${average_type} == 'epoch' ]]; then 
    ave_args="--num-epoch-checkpoints ${average_num}"
else
    echo `date`: unkow average type ${average_type}
fi 

if [[ -f ${MODEL_ROOT}/${cp_name}.pt ]]; then 
    echo $cp_name.pt exist, skip averaging checkpoints
elif [[ ${average_num} -gt 0 ]]; then
    python3 ${USER_DIR}/scripts/average_checkpoints.py \
            --inputs ${MODEL_ROOT} \
            ${ave_args} \
            --checkpoint-upper-bound ${upper_bound} \
            --output ${MODEL_ROOT}/${cp_name}.pt
else
    cp_name=checkpoint_best
fi


task=translation
beam=5
lenpen=1.0
log_prefix=${cp_name}-${lenpen}-${beam}
echo "##################################"
echo $cp_name
echo $gen_subset
echo $log_prefix
echo

fairseq-generate ${BIN_ROOT} \
    --gen-subset ${gen_subset} \
    --seed 2022 \
    --task ${task} \
    --max-tokens 4096 --max-source-positions 1024 \
    --lenpen ${lenpen} --beam ${beam} \
    --path ${MODEL_ROOT}/${cp_name}.pt \
    --bpe sentencepiece --sentencepiece-model ${SPM_MODEL} \
    --scoring sacrebleu > ${LOG_DIR}/${log_prefix}.result

grep -P "^T" ${LOG_DIR}/${log_prefix}.result | cut -f 2- > ${LOG_DIR}/${log_prefix}.ref
grep -P "^D" ${LOG_DIR}/${log_prefix}.result | cut -f 3- > ${LOG_DIR}/${log_prefix}.sys 
sacrebleu ${LOG_DIR}/${log_prefix}.ref < ${LOG_DIR}/${log_prefix}.sys