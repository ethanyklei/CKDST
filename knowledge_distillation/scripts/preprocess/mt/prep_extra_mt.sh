set -u 
set -eo pipefail

REPO_ROOT=/path/to/repo/

src=en
tgt=$1

langpair=${src}-${tgt}

if [[ ${langpair} == "en-de" ]];
then 
version="wmt16"
FILES=(
    "commoncrawl.de-en"
    "training/europarl-v7.de-en"
    "training-parallel-nc-v11/news-commentary-v11.de-en"
)
elif [[ ${langpair} == "en-es" ]];
then 
version="wmt13"
FILES=(
    "commoncrawl.es-en"
    "training/europarl-v7.es-en"
    "training/news-commentary-v8.es-en"
    "un/undoc.2000.es-en"
)
elif [[ ${langpair} == "en-fr" ]];
then 
version="wmt14"
FILES=(
    "commoncrawl.fr-en"
    "training/europarl-v7.fr-en"
    "training/news-commentary-v9.fr-en"
    "un/undoc.2000.fr-en"
    "giga-fren.release2.fixed"
)
elif [[ ${langpair} == "en-ru" ]];
then 
version="wmt16"
FILES=(
    "corpus.en_ru.1m"
    "commoncrawl.ru-en"
    "training-parallel-nc-v11/news-commentary-v11.ru-en"
    "wiki/ru-en/wiki.ru-en"
)
else
    echo unknow langpair ${langpair}
fi 

DATA_ROOT=/path/to/mt/data/
SAVE_ROOT=/path/to/save/preprocessed/mt/data
ORIG_ROOT=${SAVE_ROOT}/orig
PREP_ROOT=${SAVE_ROOT}/prep
BIN_ROOT=${SAVE_ROOT}/bin

MUSTC_ROOT=/MUSTC/RAW/DATA/ROOT
test_set=tst-COMMON

mkdir -p ${ORIG_ROOT} ${PREP_ROOT} ${BIN_ROOT}

spm_dict_path=/path/to/spm/dict/
spm_mode_path=/path/to/spm/model/


echo "pre-processing train data..."
for l in $src $tgt; do
    train_file=${ORIG_ROOT}/train.$l
    if [[ -f ${train_file} ]]; 
    then 
        rm $train_file
    fi 
    for f in "${FILES[@]}"; do
        if [[ $f != "null" ]]; then
            echo containing: ${DATA_ROOT}/$f.$l
            tr -d "\015" < ${DATA_ROOT}/$f.$l >> $train_file
        fi
    done
    echo containing: ${MUSTC_ROOT}/data/train/txt/train.${l}
    cat ${MUSTC_ROOT}/data/train/txt/train.${l} >> $train_file
    echo ${train_file}: `wc -l ${train_file}`
done



echo "pre-processing MUST-C dev data..."
for l in $src $tgt; do
    dev_file=${ORIG_ROOT}/valid_mustc.${l}
    if [[ -f ${dev_file} ]]; then 
        rm ${dev_file}
    fi 
    cat ${MUSTC_ROOT}/data/dev/txt/dev.${l} > ${dev_file}
done

echo "pre-processing original newstest2013 data..."
for l in $src $tgt; do
    dev_file=${ORIG_ROOT}/valid.$l
    if [[ -f ${dev_file} ]]; then 
        rm ${dev_file}
    fi 
    tr -d "\015" < ${DATA_ROOT}/dev/newstest2013.$l > ${dev_file}
done

echo "pre-processing MUST-C test data..."
for l in $src $tgt; do
    cp ${MUSTC_ROOT}/data/${test_set}/txt/${test_set}.$l ${ORIG_ROOT}/test_mustc.$l
done


for split in train valid_mustc valid test_mustc; do
    for l in $src $tgt; do
        echo apply_spm on ${ORIG_ROOT}/${split}.$l
        python ${REPO_ROOT}/knowledge_distillation/scripts/preprocess/mt/apply_bpe.py \
                --input-file ${ORIG_ROOT}/${split}.$l \
                --output-file ${PREP_ROOT}/${split}.bpe.$l \
                --bpe-args "{'bpe':'sentencepiece', 'sentencepiece_model': '${spm_mode_path}'}"
    done 
done

fairseq-preprocess \
    --source-lang ${src} --target-lang ${tgt} \
    --trainpref ${PREP_ROOT}/train.bpe \
    --validpref ${PREP_ROOT}/valid.bpe,${PREP_ROOT}/valid_mustc.bpe \
    --testpref ${PREP_ROOT}/test_mustc.bpe \
    --destdir ${BIN_ROOT} --thresholdtgt 0 --thresholdsrc 0 \
    --srcdict ${spm_dict_path} --tgtdict ${spm_dict_path} \
    --workers 32