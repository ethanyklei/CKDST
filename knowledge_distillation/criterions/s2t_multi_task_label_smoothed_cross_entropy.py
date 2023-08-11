# Standard Library
import math
from dataclasses import dataclass, field

import torch
from omegaconf import II

# My Stuff
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss
)


@dataclass 
class S2TMultiTaskLabelSmoothedCrossEntropyCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    asr_weight: float = field(
        default=1.0
    )
    mt_weight: float = field(
        default=1.0
    )

@register_criterion(
    "s2t_multi_task_label_smoothed_cross_entropy",
    dataclass=S2TMultiTaskLabelSmoothedCrossEntropyCriterionConfig
)
class S2TMultiTaskLabelSmoothedCrossEntropyCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, 
                task, 
                sentence_avg, 
                label_smoothing, 
                ignore_prefix_size=0, 
                report_accuracy=False,
                asr_weight=1.0,
                mt_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.asr_weight = asr_weight
        self.mt_weight = mt_weight


    def forward(self, model, sample, reduce=True):
        
        st_loss = torch.tensor(0.0)
        asr_loss = torch.tensor(0.0)
        mt_loss = torch.tensor(0.0)
        
        is_st_dataset = (sample["net_input"]["mode"] == "speech") # st mode dataset

        net_output = model(**sample['net_input'])

        if is_st_dataset:
            st_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
            
        else:
            # extra mt loss
            mt_loss, nll_loss = self.compute_loss(model, net_output, sample)

        if self.training and is_st_dataset:
            if self.asr_weight > 0.0:
                asr_loss, _ = self.compute_asr_loss(model, net_output, sample, reduce=reduce)
                asr_loss *= self.asr_weight

            if self.mt_weight > 0.0:
                mt_loss, _ = self.compute_mt_loss(model, sample, reduce=reduce)
                mt_loss *= self.mt_weight
        
        loss = st_loss + mt_loss + asr_loss
        
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        if is_st_dataset:
            st_ntokens = sample['ntokens']
            st_nsentences = sample['target'].size(0)
            src_ntokens = sample["src_ntokens"]
        else:
            st_ntokens = 0.0
            st_nsentences = 0.0
            src_ntokens = 0.0

        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "st_loss": utils.item(st_loss.data),
            "asr_loss": utils.item(asr_loss.data),
            "mt_loss": utils.item(mt_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "st_ntokens": st_ntokens,
            "st_nsentences": st_nsentences,
            "src_ntokens": src_ntokens
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_asr_loss(self, model, net_ouput, sample, reduce=True):
        asr_net_outut = model.decoder(
                            prev_output_tokens=sample['prev_output_src_tokens'],
                            encoder_out=net_ouput[1]['encoder_out']
                        )
        lprobs = model.get_normalized_probs(asr_net_outut, log_probs=True)
        target = sample['src_txt_tokens']
        if self.ignore_prefix_size > 0:
            if getattr(lprobs, "batch_first", False):
                lprobs = lprobs[:, self.ignore_prefix_size:, :].contiguous()
                target = target[:, self.ignore_prefix_size:].contiguous()
            else:
                lprobs = lprobs[self.ignore_prefix_size:, :, :].contiguous()
                target = target[self.ignore_prefix_size:, :].contiguous()
        lprobs = lprobs.view(-1, lprobs.size(-1))
        target = target.view(-1)
        loss, nll_loss = label_smoothed_nll_loss(
            lprobs,
            target,
            self.eps,
            ignore_index=self.padding_idx,
            reduce=reduce
        )
        return loss, nll_loss

    def compute_mt_loss(self, model, sample, reduce=True):
        net_output = model(sample['src_txt_tokens'], sample['src_txt_lengths'],
                            sample['net_input']['prev_output_tokens'], 
                            mode="text")
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        return loss, nll_loss

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        asr_loss_sum = sum(log.get("asr_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_ntokens = sum(log.get("st_ntokens", 0) for log in logging_outputs) + 1e-8
        st_nsentences = sum(log.get("st_nsentences", 0) for log in logging_outputs) + 1e-8
        src_ntokens = sum(log.get("src_ntokens", 0) for log in logging_outputs) + 1e-8

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / st_ntokens / math.log(2), st_ntokens, round=3
        )
        metrics.log_scalar(
            "asr_loss", asr_loss_sum / src_ntokens / math.log(2), src_ntokens, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / st_ntokens / math.log(2), st_ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )

        total = utils.item(sum(log.get("total", 0) for log in logging_outputs))
        if total > 0:
            metrics.log_scalar("total", total)
            n_correct = utils.item(
                sum(log.get("n_correct", 0) for log in logging_outputs)
            )
            metrics.log_scalar("n_correct", n_correct)
            metrics.log_derived(
                "accuracy",
                lambda meters: round(
                    meters["n_correct"].sum * 100.0 / meters["total"].sum, 3
                )
                if meters["total"].sum > 0
                else float("nan"),
            )

    @staticmethod
    def logging_outputs_can_be_summed() -> bool:
        """
        Whether the logging outputs returned by `forward` can be summed
        across workers prior to calling `reduce_metrics`. Setting this
        to True will improves distributed training speed.
        """
        return True
