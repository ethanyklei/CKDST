# Standard Library
import argparse
import json
from argparse import Namespace

import tqdm

# My Stuff
from fairseq.data import encoders

parser = argparse.ArgumentParser()
parser.add_argument("--input-file", type=str, required=True)
parser.add_argument("--output-file", type=str, required=True)
parser.add_argument("--bpe-args", type=str, required=True)

args = parser.parse_args()
bpe_args = Namespace(**json.loads(args.bpe_args.replace("\'", "\"")))
print(bpe_args)
bpe_tokenizer = encoders.build_bpe(
    bpe_args
)

with open(args.input_file, 'rb') as input_file:
    with open(args.output_file, 'w') as output_file:
        for line in tqdm.tqdm(input_file):
            line = str(line.decode("utf-8")).strip()
            encoded_line = bpe_tokenizer.encode(line).strip()
            output_file.write(encoded_line + '\n')