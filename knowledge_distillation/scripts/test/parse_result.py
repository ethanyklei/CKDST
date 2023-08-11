# Standard Library
import csv
import os
import sys
from pathlib import Path

import pandas as pd
import sacrebleu
from comet import load_from_checkpoint


def load_df_from_csv(path):
    df = pd.read_csv(
        path,
        sep="\t",
        header=0,
        encoding="utf-8",
        escapechar="\\",
        quoting=csv.QUOTE_NONE,
        na_filter=False,
    )
    return df 

csv_path = Path(sys.argv[1])
result_path = Path(sys.argv[2])
comet_model_path = Path(sys.argv[3])

df = load_df_from_csv(csv_path)
with open(result_path, 'r') as fin:
    indexes = []
    results = []
    for line in fin.readlines():
        if line.startswith("D-"):
            idx, _, res = line.split("\t")
            indexes.append(int(idx.split("D-")[1]))
            results.append(res.strip())

src_texts = []
ref_texts = []
for idx in indexes:
    src_texts.append(df['src_text'][idx])
    ref_texts.append(df['tgt_text'][idx])


bleu_sig = sacrebleu.corpus_bleu(results, [ref_texts])

print(bleu_sig)

chrf_sig = sacrebleu.corpus_chrf(results, [ref_texts])

print(chrf_sig)

ter_sig = sacrebleu.corpus_ter(results, [ref_texts])

print(ter_sig)


model = load_from_checkpoint(comet_model_path)

model_input = []
for src, mt, ref in zip(src_texts, results, ref_texts):
    model_input.append(
        {
            "src": src,
            "mt": mt,
            "ref": ref
        }
    )

model_output = model.predict(model_input, batch_size=8, gpus=1)

print(f"COMET Score: {model_output[-1]}")