#!/usr/bin/env bash 

set -u 
set -eo pipefail

REPO_ROOT=/path/to/repo/
USER_DIR=${REPO_ROOT}/knowledge_distillation

DATA_ROOT=/path/to/mustc/raw/data
data_type=audio
tgt=$1

LOG_DIR=${REPO_ROOT}/knowledge_distillation/log/preprocess/`date +%F `
mkdir -p ${LOG_DIR}

echo ${LOG_DIR}/`date +%H-%M-%S`.log

if [[ ${data_type} == "audio" ]]
then 
    DATA_ARGS="--use-audio-input --min-n-frames 1000 --max-n-frames 480000"
elif [[ ${data_type} == "fbank" ]]
then
    DATA_ARGS="--cmvn-type utterance --min-n-frames 3 --max-n-frames 3000"
else
    echo `date`: unknow data_type ${data_type}
    exit 
fi 

nohup python ${USER_DIR}/preprocess/st/prep_mustc_data.py \
    --data-root ${DATA_ROOT} \
    --vocab-type unigram \
    --vocab-size 10000 \
    --task st \
    --tgt-lang ${tgt} \
    ${DATA_ARGS} \
    > ${LOG_DIR}/`date +%H-%M-%S`.log 2>&1 &