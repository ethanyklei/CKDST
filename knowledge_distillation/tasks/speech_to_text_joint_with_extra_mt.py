# Standard Library
import json
import logging
import os
from argparse import Namespace
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import II

# My Stuff
from fairseq import metrics, utils
from fairseq.data import (
    ConcatDataset,
    Dictionary,
    ResamplingDataset,
    TransformEosLangPairDataset,
    encoders
)
from fairseq.data.audio.multi_modality_dataset import (
    ModalityDatasetItem,
    MultiModalityDataset
)
from fairseq.data.iterators import GroupedEpochBatchIterator
from fairseq.dataclass.configs import FairseqDataclass
from fairseq.optim.amp_optimizer import AMPOptimizer
from fairseq.tasks import register_task
from fairseq.tasks.speech_to_text import SpeechToTextTask
from fairseq.tasks.translation import load_langpair_dataset
# user custome module
from knowledge_distillation.data.speech_to_text_triple_dataset import (
    S2TTripleDataConfig,
    SpeechToTextTripleDatasetCreator
)

logger = logging.getLogger(__name__)

LANG_TAG_TEMPLATE = "<lang:{}>"
EVAL_BLEU_ORDER=4

@dataclass
class SpeechToTextJointWithExtraMTConfig(FairseqDataclass):

    # default args
    data: Optional[str] = field(
        default=None,
        metadata={"help": "manifest root path"}
    )
    config_yaml: str = field(
        default="config.yaml",
        metadata={"help": "Configuration YAML filename (under manifest root)"}
    )
    max_source_positions: int = field(
        default=6000,
        metadata={"help": "max number of tokens in the source sequence"}
    )
    max_target_positions: int = field(
        default=1024,
        metadata={"help": "max number of tokens in the target sequence"}
    )
    prefix_size: int = field(
        default=0,
        metadata={"help": ""}
    )
    mode: str = field(
        default="speech_to_text",
    )
    train_subset: str = II("dataset.train_subset")
    seed: int = II("common.seed")
    input_feat_per_channel: Optional[int] = field(
        default=80
    )
    input_channels: Optional[int] = field(
        default=1
    )
    speaker_to_id: Optional[str] = field(
        default=None
    )

    max_tokens: Optional[int] = II("dataset.max_tokens")
    batch_size: Optional[int] = II("dataset.batch_size")
    
    # extra mt args
    parallel_text_data: Optional[str] = field(
        default="",
    )
    max_text_tokens: Optional[int] = field(
        default=8096,
    )
    max_text_positions: Optional[int] = field(
        default=512
    )
    langpair: Optional[str] = field(
        default="en-de",
    )
    speech_sample_ratio: Optional[float] = field(
        default=1,
    )
    text_sample_ratio: Optional[float] = field(
        default=1
    )

    roberta_dict_root: Optional[str] = field(
        default=None
    )

    # eval bleu args 
    eval_bleu: bool = field(
        default=False, metadata={"help": "evaluation with BLEU scores"}
    )
    eval_bleu_args: Optional[str] = field(
        default="{}",
        metadata={
            "help": 'generation args for BLUE scoring, e.g., \'{"beam": 4, "lenpen": 0.6}\', as JSON string'
        },
    )
    eval_bleu_detok: str = field(
        default="space",
        metadata={
            "help": "detokenize before computing BLEU (e.g., 'moses'); required if using --eval-bleu; "
            "use 'space' to disable detokenization; see fairseq.data.encoders for other options"
        },
    )
    eval_bleu_detok_args: Optional[str] = field(
        default="{}",
        metadata={"help": "args for building the tokenizer, if needed, as JSON string"},
    )
    eval_tokenized_bleu: bool = field(
        default=False, metadata={"help": "compute tokenized BLEU instead of sacrebleu"}
    )
    eval_bleu_remove_bpe: Optional[str] = field(
        default=None,
        metadata={
            "help": "remove BPE before computing BLEU",
            "argparse_const": "@@ ",
        },
    )
    eval_bleu_print_samples: bool = field(
        default=False, metadata={"help": "print sample generations during validation"}
    )

    micro_batch_size: int = field(
        default=-1
    )


@register_task("speech_to_text_joint_with_extra_mt", dataclass=SpeechToTextJointWithExtraMTConfig)
class SpeechToTextJointWithExtraMTTask(SpeechToTextTask):

    def __init__(self, args, src_dict, tgt_dict, mode="speech"):
        super().__init__(args, tgt_dict)
        self.src_dict = src_dict
        self.data_cfg = S2TTripleDataConfig(Path(args.data) / args.config_yaml)
        assert self.tgt_dict.pad() == self.src_dict.pad()
        assert self.tgt_dict.eos() == self.src_dict.eos()
        self.mode = mode
        self.src_lang, self.tgt_lang = args.langpair.split("-")

    @classmethod
    def add_args(cls, parser):
        super(SpeechToTextTask, cls).add_args(parser)

    @classmethod
    def setup_task(cls, args, **kwargs):
        """Setup the task (e.g., load dictionaries)."""
        data_cfg = S2TTripleDataConfig(Path(args.data) / args.config_yaml)
        tgt_dict_path = Path(args.data) / data_cfg.vocab_filename
        if args.roberta_dict_root is not None:
            src_dict_path = Path(args.roberta_dict_root) / "dict.txt"
        else:
            src_dict_path = Path(args.data) / data_cfg.src_vocab_filename
        if (not os.path.isfile(src_dict_path)) or (not os.path.isfile(tgt_dict_path)):
            raise FileNotFoundError("Dict not found: {}".format(args.data))
        
        src_dict = Dictionary.load(src_dict_path.as_posix())
        tgt_dict = Dictionary.load(tgt_dict_path.as_posix())

        if args.roberta_dict_root is not None:
            src_dict.add_symbol("<mask>")
            src_dict.is_roberta = True
        else:
            src_dict.is_roberta = False

        logger.info("| src dictionary: {} types".format(len(src_dict)))
        logger.info("| tgt dictionary: {} types".format(len(tgt_dict)))

        if args.parallel_text_data != "":
            if not os.path.isabs(args.parallel_text_data):
                args.parallel_text_data = os.path.join(
                    args.data, args.parallel_text_data
                )

            if args.langpair is None:
                raise Exception(
                    "Could not infer language pair, please provide it explicitly"
                )

        # the mode is a controller to swith the primary task (st or mt)
        assert args.mode in ['speech', 'text']

        return cls(args, src_dict, tgt_dict, mode=args.mode)

    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
            )

    def build_src_tokenizer(self, args):
        logger.info(f"src-pre-tokenizer: {self.data_cfg.src_pre_tokenizer}")
        return encoders.build_tokenizer(Namespace(**self.data_cfg.src_pre_tokenizer))

    def build_src_bpe(self, args):
        logger.info(f"src-tokenizer: {self.data_cfg.src_bpe_tokenizer}")
        return encoders.build_bpe(Namespace(**self.data_cfg.src_bpe_tokenizer))

    def load_langpair_dataset(
        self, prepend_tgt_lang_tag=False, sampling_alpha=1.0, epoch=0
    ):
        text_dataset = None
        split = "train"
        # if using lang tag, please add lang tag to the raw text data
        text_dataset = load_langpair_dataset(
            self.args.parallel_text_data,
            split,
            self.src_lang,
            self.src_dict,
            self.tgt_lang,
            self.tgt_dict,
            combine=True,
            dataset_impl=None,
            upsample_primary=1,
            left_pad_source=False,
            left_pad_target=False,
            max_source_positions=self.args.max_text_positions,
            max_target_positions=self.args.max_target_positions,
            load_alignments=False,
            truncate_source=False,
        )
        return text_dataset

    def load_dataset(self, split, epoch=1, combine=False, **kwargs):
        """Load a given dataset split.

        Args:
            split (str): name of the split (e.g., train, valid, test)
        """
        is_train_split = split.startswith("train")
        pre_tokenizer = self.build_tokenizer(self.args)
        bpe_tokenizer = self.build_bpe(self.args)
        src_pre_tokenizer = self.build_src_tokenizer(self.args)
        src_bpe_tokenizer = self.build_src_bpe(self.args)
        ast_dataset = SpeechToTextTripleDatasetCreator.from_tsv(
            self.args.data,
            self.data_cfg,
            split,
            self.tgt_dict,
            pre_tokenizer=pre_tokenizer,
            bpe_tokenizer=bpe_tokenizer,
            is_train_split=is_train_split,
            epoch=epoch,
            seed=self.args.seed,
            src_dict=self.src_dict,
            src_pre_tokenizer=src_pre_tokenizer,
            src_bpe_tokenizer=src_bpe_tokenizer,
            mode=self.mode,
        )
        
        text_dataset = None
        if self.args.parallel_text_data != "" and is_train_split:
            text_dataset = self.load_langpair_dataset(
                self.data_cfg.prepend_tgt_lang_tag, 1.0, epoch=epoch,
            )

        if text_dataset is not None:
            mdsets = [
                ModalityDatasetItem(
                    "speech",
                    ast_dataset,
                    (self.args.max_source_positions, self.args.max_target_positions),
                    self.args.max_tokens,
                    self.args.batch_size,
                ),
                ModalityDatasetItem(
                    "text",
                    text_dataset,
                    (self.args.max_text_positions, self.args.max_target_positions),
                    self.args.max_text_tokens
                    if self.args.max_text_tokens is not None
                    else self.args.max_tokens,
                    self.args.batch_size,
                ),
            ]
            ast_dataset = MultiModalityDataset(mdsets)
        self.datasets[split] = ast_dataset

    
    @property
    def target_dictionary(self):
        """Return the :class:`~fairseq.data.Dictionary` for the language
        model."""
        return self.tgt_dict

    @property
    def source_dictionary(self):
        """Return the source :class:`~fairseq.data.Dictionary` (if applicable
        for this task)."""
        return None if self.mode == "speech" else self.src_dict

    @property
    def text_source_dictionary(self):
        return self.src_dict

    def get_batch_iterator(
        self,
        dataset,
        max_tokens=None,
        max_sentences=None,
        max_positions=None,
        ignore_invalid_inputs=False,
        required_batch_size_multiple=1,
        seed=1,
        num_shards=1,
        shard_id=0,
        num_workers=0,
        epoch=0,
        data_buffer_size=0,
        disable_iterator_cache=False,
        skip_remainder_batch=False,
        grouped_shuffling=False,
        update_epoch_batch_itr=False,
    ):

        if not isinstance(dataset, MultiModalityDataset):
            return super(SpeechToTextJointWithExtraMTTask, self).get_batch_iterator(
                dataset,
                max_tokens,
                max_sentences,
                max_positions,
                ignore_invalid_inputs,
                required_batch_size_multiple,
                seed,
                num_shards,
                shard_id,
                num_workers,
                epoch,
                data_buffer_size,
                disable_iterator_cache,
                skip_remainder_batch=skip_remainder_batch,
                update_epoch_batch_itr=update_epoch_batch_itr,
            )

        mult_ratio = [self.args.speech_sample_ratio, self.args.text_sample_ratio]
        assert len(dataset.datasets) == 2

        # initialize the dataset with the correct starting epoch
        dataset.set_epoch(epoch)

        batch_samplers = dataset.get_batch_samplers(
            mult_ratio, required_batch_size_multiple, seed
        )

        # return a reusable, sharded iterator
        epoch_iter = GroupedEpochBatchIterator(
            dataset=dataset,
            collate_fn=dataset.collater,
            batch_samplers=batch_samplers,
            seed=seed,
            num_shards=num_shards,
            shard_id=shard_id,
            num_workers=num_workers,
            epoch=epoch,
            mult_rate=1,
            # mult_rate=1 if self.args.update_mix_data else max(self.args.update_freq),
            buffer_size=data_buffer_size,
            skip_remainder_batch=skip_remainder_batch,
        )
        self.dataset_to_epoch_iter[dataset] = {}  # refresh it every epoch
        return epoch_iter

    def build_model(self, cfg: FairseqDataclass, from_checkpoint=False):
        # My Stuff
        from fairseq import models, quantization_utils

        model = models.build_model(cfg, self, from_checkpoint)
        model = quantization_utils.quantize_model_scalar(model, cfg)

        if self.args.eval_bleu:
            detok_args = json.loads(self.args.eval_bleu_detok_args)
            self.tokenizer = encoders.build_tokenizer(
                Namespace(tokenizer=self.args.eval_bleu_detok, **detok_args)
            )

            gen_args = json.loads(self.args.eval_bleu_args)
            self.sequence_generator = self.build_generator(
                [model], Namespace(**gen_args)
            )
        return model

    def valid_step(self, sample, model, criterion):
        loss, sample_size, logging_output = super().valid_step(sample, model, criterion)
        if self.args.eval_bleu:
            bleu = self._inference_with_bleu(self.sequence_generator, sample, model)
            logging_output['_bleu_sys_len'] = bleu.sys_len
            logging_output["_bleu_ref_len"] = bleu.ref_len
            # we split counts into separate entries so that they can be
            # summed efficiently across workers using fast-stat-sync
            assert len(bleu.counts) == EVAL_BLEU_ORDER
            for i in range(EVAL_BLEU_ORDER):
                logging_output["_bleu_counts_" + str(i)] = bleu.counts[i]
                logging_output["_bleu_totals_" + str(i)] = bleu.totals[i]
        return loss, sample_size, logging_output

    def _inference_with_bleu(self, generator, sample, model):

        import sacrebleu

        def get_symbols_to_strip_from_output(generator):
            if hasattr(generator, "symbols_to_strip_from_output"):
                return generator.symbols_to_strip_from_output
            else:
                return {generator.eos}

        def decode(toks, escape_unk=False):
            s = self.tgt_dict.string(
                toks.int().cpu(),
                self.args.eval_bleu_remove_bpe,
                # The default unknown string in fairseq is `<unk>`, but
                # this is tokenized by sacrebleu as `< unk >`, inflating
                # BLEU scores. Instead, we use a somewhat more verbose
                # alternative that is unlikely to appear in the real
                # reference, but doesn't get split into multiple tokens.
                unk_string=("UNKNOWNTOKENINREF" if escape_unk else "UNKNOWNTOKENINHYP"),
                extra_symbols_to_ignore=get_symbols_to_strip_from_output(generator)
            )
            if self.tokenizer:
                s = self.tokenizer.decode(s)
            return s
        
        prefix_tokens = None
        if self.data_cfg.prepend_tgt_lang_tag:
            prefix_tokens = sample['target'][:, : 1]
        
        gen_out = self.inference_step(generator, [model], sample, prefix_tokens=prefix_tokens)

        hyps, refs = [], []
        for i in range(len(gen_out)):
            hyps.append(decode(gen_out[i][0]['tokens']))
            refs.append(
                decode(
                    utils.strip_pad(sample["target"][i], self.tgt_dict.pad()),
                    escape_unk=True,
                )
            )

        if self.args.eval_bleu_print_samples:
            logger.info("example hypothesis: " + hyps[0])
            logger.info("example reference: " + refs[0])
        if self.args.eval_tokenized_bleu:
            return sacrebleu.corpus_bleu(hyps, [refs], tokenize="none")
        else:
            if self.tgt_lang == "zh":
                return sacrebleu.corpus_bleu(hyps, [refs], tokenize="zh")
            else:
                return sacrebleu.corpus_bleu(hyps, [refs])

    def reduce_metrics(self, logging_outputs, criterion):
        super().reduce_metrics(logging_outputs, criterion)
        if self.args.eval_bleu:
            def sum_logs(key):
                if key in logging_outputs[0]:
                    if isinstance(logging_outputs[0][key], int):
                        return sum(log[key] for log in logging_outputs)
                    else:
                        return sum(log[key].cpu().numpy() for log in logging_outputs)
                return sum(log.get(key, 0) for log in logging_outputs)

            counts, totals = [], []
            for i in range(EVAL_BLEU_ORDER):
                counts.append(sum_logs("_bleu_counts_" + str(i)))
                totals.append(sum_logs("_bleu_totals_" + str(i)))

            if max(totals) > 0:
                # log counts as numpy arrays -- log_scalar will sum them correctly
                metrics.log_scalar("_bleu_counts", np.array(counts))
                metrics.log_scalar("_bleu_totals", np.array(totals))
                metrics.log_scalar("_bleu_sys_len", sum_logs("_bleu_sys_len"))
                metrics.log_scalar("_bleu_ref_len", sum_logs("_bleu_ref_len"))

                def compute_bleu(meters):
                    # Standard Library
                    import inspect

                    import sacrebleu

                    fn_sig = inspect.getfullargspec(sacrebleu.BLEU.compute_bleu)[0]
                    if "smooth_method" in fn_sig:
                        smooth = {"smooth_method": "exp"}
                    else:
                        smooth = {"smooth": "exp"}
                    bleu = sacrebleu.BLEU.compute_bleu(
                        correct=meters["_bleu_counts"].sum,
                        total=meters["_bleu_totals"].sum,
                        sys_len=int(meters["_bleu_sys_len"].sum),
                        ref_len=int(meters["_bleu_ref_len"].sum),
                        **smooth
                    )
                    return round(bleu.score, 2)

                metrics.log_derived("bleu", compute_bleu)

    def train_step(
        self, sample, model, criterion, optimizer, update_num, ignore_grad=False
    ):
        model.train()
        model.set_num_updates(update_num)
        bsz = sample['net_input']['src_tokens'].size(0)
        micro_bsz = self.args.micro_batch_size
        if micro_bsz > 0:
            if bsz % micro_bsz == 0:
                micro_batch_num = bsz // micro_bsz
            else:
                micro_batch_num = bsz // micro_bsz + 1
        else:
            micro_batch_num = 1
        loss = None
        sample_size = None
        logging_output = None
        for micro_idx in range(micro_batch_num):
            start_idx = micro_bsz * micro_idx
            end_idx = micro_bsz * (micro_idx + 1) if micro_idx != (micro_batch_num - 1) else bsz
            
            micro_sample = {
                "id": sample['id'][start_idx: end_idx],
                "net_input": {
                    "src_tokens": sample['net_input']['src_tokens'][start_idx: end_idx],
                    "src_lengths": sample['net_input']['src_lengths'][start_idx: end_idx],
                    "prev_output_tokens": sample['net_input']['prev_output_tokens'][start_idx: end_idx],
                    "mode": sample['net_input']['mode'],
                },
                "target": sample['target'][start_idx: end_idx],
                "target_lengths": sample['target_lengths'][start_idx: end_idx],
                "src_txt_tokens": sample['src_txt_tokens'][start_idx: end_idx],
                "src_txt_lengths": sample['src_txt_lengths'][start_idx: end_idx],
                "prev_output_src_tokens": sample['prev_output_src_tokens'][start_idx: end_idx],
                "ntokens": sum(sample['target_lengths'][start_idx: end_idx]),
                "src_ntokens": sum(sample['src_txt_lengths'][start_idx: end_idx]),
            } 

            with torch.autograd.profiler.record_function("forward"):
                with torch.cuda.amp.autocast(enabled=(isinstance(optimizer, AMPOptimizer))):
                    micro_loss, micro_sample_size, micro_logging_output = criterion(model, micro_sample)
            
            if ignore_grad:
                micro_loss *= 0
            with torch.autograd.profiler.record_function("backward"):
                optimizer.backward(micro_loss)

            if micro_idx == 0:
                loss = micro_loss
                sample_size = micro_sample_size
                logging_output = micro_logging_output
            else:
                loss += micro_loss
                sample_size += micro_sample_size
                for key in micro_logging_output:
                    logging_output[key] += micro_logging_output[key]

        return loss, sample_size, logging_output