# !/usr/bin/env bash 

set -u 
set -eo pipefail

REPO_ROOT=/path/to/repo/
USER_DIR=${REPO_ROOT}/knowledge_distillation

EXP_NAME=$1
DATASET=mustc
src=en
tgt=$2
langpair=${src}-${tgt}

config_yaml=config_st_audio_bi
gen_subset=tst-COMMON_st_audio
task_args="--config-yaml ${config_yaml}.yaml --langpair ${src}-${tgt} --gen-subset ${gen_subset}"

DATA_ROOT=/path/to/mustc/test/tsv
MODEL_ROOT=/path/to/model/dir
LOG_DIR=${USER_DIR}/log/test/st/${DATASET}-${langpair}/${EXP_NAME}/

mkdir -p ${LOG_DIR}

# [update  epoch]
average_type=update
average_num=$3
upper_bound=$4

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

# # cp_name=checkpoint_best
# # gen_subset=dev
echo ########### averge ckpt ###############
echo $cp_name
echo $gen_subset
echo

beam=10
lenpen=1.0

mode_args="--mode speech --max-tokens 10000000 --max-source-positions 10000000"


log_prefix=${cp_name}-${lenpen}-${beam}

echo ########### inference ###############
echo ${log_prefix}
echo

fairseq-generate ${DATA_ROOT} \
    --user-dir ${USER_DIR} \
    \
    --task speech_to_text_joint_with_extra_mt \
    ${task_args} \
    \
    ${mode_args} \
    \
    --lenpen ${lenpen} --beam ${beam} \
    \
    --path ${MODEL_ROOT}/${cp_name}.pt \
    --model-overrides ${model_overrides} \
    --scoring sacrebleu > ${LOG_DIR}/${log_prefix}.result

grep -P "^T|^D" ${LOG_DIR}/${log_prefix}.result | sed -e "s/\t-[0-9].*\t/\t/g"> ${LOG_DIR}/${log_prefix}.align
grep -P "^T" ${LOG_DIR}/${log_prefix}.result | cut -f 2- > ${LOG_DIR}/${log_prefix}.ref
grep -P "^D" ${LOG_DIR}/${log_prefix}.result | cut -f 3- > ${LOG_DIR}/${log_prefix}.sys 
sacrebleu -l ${langpair} ${LOG_DIR}/${log_prefix}.ref < ${LOG_DIR}/${log_prefix}.sys 

python ${USER_DIR}/scripts/test/parse_result.py ${DATA_ROOT}/${gen_subset}.tsv ${LOG_DIR}/${log_prefix}.result /path/to/comet/model