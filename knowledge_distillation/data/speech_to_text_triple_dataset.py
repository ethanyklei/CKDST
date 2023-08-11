# Standard Library
import csv
import io
import logging
import re
from dataclasses import dataclass
from email.mime import audio
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import torch
import torch.nn.functional as F

# My Stuff
from fairseq.data import ConcatDataset, Dictionary, ResamplingDataset
from fairseq.data import data_utils as fairseq_data_utils
from fairseq.data.audio.data_cfg import S2TDataConfig
from fairseq.data.audio.speech_to_text_dataset import (
    SpeechToTextDataset,
    SpeechToTextDatasetCreator,
    _collate_frames
)
from fairseq.data.encoders.gpt2_bpe import GPT2BPE

logger = logging.getLogger(__name__)

class S2TTripleDataConfig(S2TDataConfig):

    @property
    def prepend_src_lang_tag(self) -> bool:
        """Prepend target lang ID token as the target BOS (e.g. for to-many
        multilingual setting). During inference, this requires `--prefix-size 1`
        to force BOS to be lang ID token."""
        return self.config.get("prepend_src_lang_tag", False)

    @property
    def src_vocab_filename(self):
        """fairseq vocabulary file under data root"""
        return self.config.get("src_vocab_filename", "src_dict.txt")

    @property
    def src_pre_tokenizer(self) -> Dict:
        """Pre-tokenizer to apply before subword tokenization. Returning
        a dictionary with `tokenizer` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_pre_tokenizer", {"tokenizer": None})

    @property
    def src_bpe_tokenizer(self) -> Dict:
        """Subword tokenizer to apply on source text after pre-tokenization.
        Returning a dictionary with `bpe` providing the tokenizer name and
        the other items providing the tokenizer-specific arguments.
        Tokenizers are defined in `fairseq.data.encoders.*`"""
        return self.config.get("src_bpe_tokenizer", {"bpe": None})

@dataclass
class SpeechToTextTripleDatasetItem(object):
    index: int
    source: torch.Tensor
    target: Optional[torch.Tensor] = None
    src_txt_tokens: Optional[torch.Tensor] = None


class SpeechToTextTripleDataset(SpeechToTextDataset):
    LANG_TAG_TEMPLATE = "<lang:{}>"

    def __init__(
        self,
        split: str,
        is_train_split: bool,
        cfg: S2TTripleDataConfig,
        audio_paths: List[str],
        n_frames: List[int],
        src_texts: Optional[List[str]] = None,
        tgt_texts: Optional[List[str]] = None,
        speakers: Optional[List[str]] = None,
        src_langs: Optional[List[str]] = None,
        tgt_langs: Optional[List[str]] = None,
        ids: Optional[List[str]] = None,
        tgt_dict: Optional[Dictionary] = None,
        pre_tokenizer=None,
        bpe_tokenizer=None,
        src_dict: Optional[Dictionary] = None,
        src_pre_tokenizer=None,
        src_bpe_tokenizer=None,
        append_eos=True,
        mode="speech",
    ):
        super().__init__(
            split=split,
            is_train_split=is_train_split,
            cfg=cfg, 
            audio_paths=audio_paths, 
            n_frames=n_frames,
            src_texts=src_texts, 
            tgt_texts=tgt_texts,
            speakers=speakers, 
            src_langs=src_langs, 
            tgt_langs=tgt_langs,
            ids=ids, 
            tgt_dict=tgt_dict, 
            pre_tokenizer=pre_tokenizer, 
            bpe_tokenizer=bpe_tokenizer, 
            append_eos=append_eos,
        )
        self.check_src_lang_tag()
        self.src_dict = src_dict
        self.src_pre_tokenizer = src_pre_tokenizer
        self.src_bpe_tokenizer = src_bpe_tokenizer
        self.src_lens = self.get_src_lens_and_check_oov()
        self.mode = mode
    
    def get_src_lens_and_check_oov(self):
        if self.src_texts is None:
            return [0 for _ in range(self.n_samples)]
        src_lens = []
        n_tokens, n_oov_tokens = 0, 0
        for i in range(self.n_samples):
            tokenized = self.get_tokenized_src_text(i).split(" ")
            oov_tokens = [
                t
                for t in tokenized
                if self.src_dict.index(t) == self.src_dict.unk_index
            ]
            n_tokens += len(tokenized)
            n_oov_tokens += len(oov_tokens)
            src_lens.append(len(tokenized))
        logger.info(f"'{self.split}' has {n_oov_tokens / n_tokens * 100:.2f}% OOV")
        return src_lens

    def check_src_lang_tag(self):
        if self.cfg.config.get("prepend_src_lang_tag", False):
            assert self.src_langs is not None and self.tgt_dict is not None
            src_lang_tags = [
                self.LANG_TAG_TEMPLATE.format(t) for t in set(self.src_langs)
            ]
            assert all(t in self.tgt_dict for t in src_lang_tags)

    def get_tokenized_src_text(self, index: int):
        text = self.tokenize(self.src_pre_tokenizer, self.src_texts[index])
        text = self.tokenize(self.src_bpe_tokenizer, text)
        return text

    def __getitem__(self, index: int) -> SpeechToTextTripleDatasetItem:
        if self.mode == "speech":
            source = self._get_source_audio(index)
            source = self.pack_frames(source)
        else:
            source = None

        target = None
        if self.tgt_texts is not None:
            tokenized = self.get_tokenized_tgt_text(index)
            target = self.tgt_dict.encode_line(
                tokenized, add_if_not_exist=False, append_eos=self.append_eos
            ).long()
            if self.cfg.prepend_tgt_lang_tag:
                lang_tag_idx = self.get_lang_tag_idx(
                    self.tgt_langs[index], self.tgt_dict
                )
                target = torch.cat((torch.LongTensor([lang_tag_idx]), target), 0)

        src_txt_tokens = None
        if self.src_texts is not None and self.src_dict is not None:
            if self.src_dict.is_roberta:
                src_txt_tokens = "<s> " + self.src_bpe_tokenizer.encode(self.src_texts[index]) + " </s>"
                src_txt_tokens = self.src_dict.encode_line(
                src_txt_tokens, add_if_not_exist=False, append_eos=False
            ).long()
            else:
                src_txt_tokens = self.get_tokenized_src_text(index)
                src_txt_tokens = self.src_dict.encode_line(
                    src_txt_tokens, add_if_not_exist=False, append_eos=self.append_eos
                ).long()
                if self.cfg.config.get("prepend_src_lang_tag", False):
                    lang_tag = self.LANG_TAG_TEMPLATE.format(self.src_langs[index])
                    lang_tag_idx = self.src_dict.index(lang_tag)
                    assert lang_tag_idx != self.src_dict.unk()
                    src_txt_tokens = torch.cat([torch.LongTensor([lang_tag_idx]), src_txt_tokens], 0)

        return SpeechToTextTripleDatasetItem(
            index=index, 
            source=source,
            target=target,
            src_txt_tokens=src_txt_tokens, 
        )
    
    def __len__(self):
        return self.n_samples

    def collater(
        self, samples: List[SpeechToTextTripleDatasetItem], return_order: bool = False
    ) -> Dict:
        if len(samples) == 0:
            return {}
        
        indices = torch.tensor([x.index for x in samples], dtype=torch.long)

        if self.mode == "speech":
            frames = _collate_frames([x.source for x in samples], self.cfg.use_audio_input)
            # sort samples by descending number of frames
            n_frames = torch.tensor([x.source.size(0) for x in samples], dtype=torch.long)
            n_frames, order = n_frames.sort(descending=True)
            frames = frames.index_select(0, order)
        else:
            order = None        
        
        if self.src_texts is not None:
            src_txt_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_tokens for x in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False
            )
            src_txt_lenghts = torch.tensor(
                [x.src_txt_tokens.size(0) for x in samples], dtype=torch.long
            )

            prev_output_src_tokens = fairseq_data_utils.collate_tokens(
                [x.src_txt_tokens for x in samples],
                self.src_dict.pad(),
                self.src_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=True
            )

            if order == None:
                src_txt_lenghts, order = src_txt_lenghts.sort(descending=True)
            else:
                src_txt_lenghts = src_txt_lenghts.index_select(0, order)

            src_txt_tokens = src_txt_tokens.index_select(0, order)
            prev_output_src_tokens = prev_output_src_tokens.index_select(0, order)
            src_ntokens = sum(x.src_txt_tokens.size(0) for x in samples)

        indices = indices.index_select(0, order)

        target, target_lengths = None, None
        prev_output_tokens = None
        ntokens = None
        if self.tgt_texts is not None:
            target = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                self.tgt_dict.eos(),
                left_pad=False,
                move_eos_to_beginning=False,
            )
            target = target.index_select(0, order)
            target_lengths = torch.tensor(
                [x.target.size(0) for x in samples], dtype=torch.long
            ).index_select(0, order)
            prev_output_tokens = fairseq_data_utils.collate_tokens(
                [x.target for x in samples],
                self.tgt_dict.pad(),
                eos_idx=None,
                left_pad=False,
                move_eos_to_beginning=True,
            )
            prev_output_tokens = prev_output_tokens.index_select(0, order)
            ntokens = sum(x.target.size(0) for x in samples)

        if self.mode == "speech":
            net_input = {
                "src_tokens": frames,
                "src_lengths": n_frames,
                "prev_output_tokens": prev_output_tokens,
                "mode": self.mode,
                "add_lang_tag": self.cfg.config.get("prepend_src_lang_tag", False)
            }
            out = {
                "id": indices,
                "net_input": net_input,
                "target": target,
                "target_lengths": target_lengths,
                "ntokens": ntokens,
                "nsentences": len(samples),
                "src_txt_tokens": src_txt_tokens,
                "src_txt_lengths": src_txt_lenghts,
                "prev_output_src_tokens": prev_output_src_tokens,
                "src_ntokens": src_ntokens,
            }
        elif self.mode == "text":
            net_input = {
                "src_tokens": src_txt_tokens,
                "src_lengths": src_txt_lenghts,
                "prev_output_tokens": prev_output_tokens,
                "mode": self.mode
            }
            out = {
                "id": indices,
                "net_input": net_input,
                "target": target,
                "target_lengths": target_lengths,
                "ntokens": ntokens,
                "nsentences": len(samples),
            }
        return out

    def num_tokens(self, index):
        # return self.n_frames[index]
        if self.mode == "speech":
            return self.n_frames[index]
        elif self.mode == "text":
            return max(
                self.src_lens[index],
                self.tgt_lens[index]
            )

    def size(self, index):
        # return self.n_frames[index], self.tgt_lens[index]
        if self.mode == "speech":
            return self.n_frames[index], self.tgt_lens[index]
        elif self.mode == "text":
            return self.src_lens[index], self.tgt_lens[index]

    def ordered_indices(self):
        if self.shuffle:
            order = [np.random.permutation(len(self))]
        else:
            order = [np.arange(len(self))]
        # first by descending order of # of frames then by original/random order
        # order.append([-n for n in self.n_frames])
        if self.mode == "speech":
            order.append([-n for n in self.n_frames])
        elif self.mode == "text":
            order.append([-n for n in self.src_lens])
        return np.lexsort(order)

    @property
    def sizes(self):
        # return np.array(self.n_frames)
        
        if self.mode == "speech":
            return np.array(self.n_frames)
        elif self.mode == "text":
            return np.array(self.src_lens)
        

class SpeechToTextTripleDatasetCreator(SpeechToTextDatasetCreator):

    @classmethod
    def _from_list(
        cls,
        split_name: str,
        is_train_split,
        samples: List[Dict],
        cfg: S2TTripleDataConfig,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        src_dict,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        mode,

    ) -> SpeechToTextTripleDataset:
        audio_root = Path(cfg.audio_root)
        ids = [s[cls.KEY_ID] for s in samples]
        audio_paths = [(audio_root / s[cls.KEY_AUDIO]).as_posix() for s in samples]
        n_frames = [int(s[cls.KEY_N_FRAMES]) for s in samples]
        tgt_texts = [s[cls.KEY_TGT_TEXT] for s in samples]
        src_texts = [s.get(cls.KEY_SRC_TEXT, cls.DEFAULT_SRC_TEXT) for s in samples]
        src_langs = [s.get(cls.KEY_SRC_LANG, cls.DEFAULT_LANG) for s in samples]
        tgt_langs = [s.get(cls.KEY_TGT_LANG, cls.DEFAULT_LANG) for s in samples]
        return SpeechToTextTripleDataset(
            split_name,
            is_train_split,
            cfg,
            audio_paths,
            n_frames,
            src_texts=src_texts,
            tgt_texts=tgt_texts,
            src_langs=src_langs,
            tgt_langs=tgt_langs,
            ids=ids,
            tgt_dict=tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            src_dict=src_dict,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            mode=mode,
        )

    @classmethod
    def _from_tsv(
        cls,
        root: str,
        cfg: S2TTripleDataConfig,
        split: str,
        tgt_dict,
        is_train_split: bool,
        pre_tokenizer,
        bpe_tokenizer,
        src_dict,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        mode,
    ) -> SpeechToTextTripleDataset:
        samples = cls._load_samples_from_tsv(root, split)
        return cls._from_list(
            split,
            is_train_split,
            samples,
            cfg,
            tgt_dict,
            pre_tokenizer,
            bpe_tokenizer,
            src_dict,
            src_pre_tokenizer,
            src_bpe_tokenizer,
            mode,
        )

    @classmethod
    def from_tsv(
        cls,
        root: str,
        cfg: S2TTripleDataConfig,
        splits: str,
        tgt_dict,
        pre_tokenizer,
        bpe_tokenizer,
        is_train_split: bool,
        epoch: int,
        seed: int,
        src_dict,
        src_pre_tokenizer,
        src_bpe_tokenizer,
        mode,
    ) -> SpeechToTextTripleDataset:
        datasets = [
            cls._from_tsv(
                root,
                cfg,
                split,
                tgt_dict,
                is_train_split,
                pre_tokenizer,
                bpe_tokenizer,
                src_dict,
                src_pre_tokenizer,
                src_bpe_tokenizer,
                mode,
            )
            for split in splits.split(",")
        ]

        if is_train_split and len(datasets) > 1 and cfg.sampling_alpha != 1.0:
            # temperature-based sampling
            size_ratios = cls.get_size_ratios(datasets, alpha=cfg.sampling_alpha)
            datasets = [
                ResamplingDataset(
                    d, size_ratio=r, seed=seed, epoch=epoch, replace=(r >= 1.0)
                )
                for r, d in zip(size_ratios, datasets)
            ]

        return ConcatDataset(datasets) if len(datasets) > 1 else datasets[0]