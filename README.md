# CKDST: Comprehensively and Effectively Distill Knowledge from Machine Translation to End-to-End Speech Translation
This is the offical source code for ACL2023-findings paper "CKDST: Comprehensively and Effectively Distill Knowledge from Machine Translation to End-to-End Speech Translation".

## Requirements and Installation
To make sure you have the following requirements ready.
* [PyTorch](http://pytorch.org/) version >= 1.13.1
* Python version >= 3.8
* [Fairseq](https://github.com/facebookresearch/fairseq) >= 0.12.1

Next you can install the project according to the following commands.
```bash
git clone git@github.com:ethanyklei/CKDST.git
git submodule update --init --recursive
cd fairseq
pip install -e .
```

## Data Preprocess
### MUST-C Data
You can find more detail in [here](knowledge_distillation/scripts/preprocess/st/README.md) for how to preprocess MUST-C data.

### Extra-MT
You can find more detail in [here](knowledge_distillation/scripts/preprocess/mt/README.md) for how to preprocess Extra MT data.

## Training
You can find more detail in [here](knowledge_distillation/scripts/train/README.md) for how to training.

## Evaluation
You can find more detail in [here](knowledge_distillation/scripts/test/README.md) for how to evaluate you model.


## Citation
Please cite as:
``` bibtex
@inproceedings{lei-etal-2023-ckdst,
    title = "{CKDST}: Comprehensively and Effectively Distill Knowledge from Machine Translation to End-to-End Speech Translation",
    author = "Lei, Yikun  and
      Xue, Zhengshan  and
      Zhao, Xiaohu  and
      Sun, Haoran  and
      Zhu, Shaolin  and
      Lin, Xiaodong  and
      Xiong, Deyi",
    booktitle = "Findings of the Association for Computational Linguistics: ACL 2023",
    month = jul,
    year = "2023",
    address = "Toronto, Canada",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.findings-acl.195",
    pages = "3123--3137",
}
```
