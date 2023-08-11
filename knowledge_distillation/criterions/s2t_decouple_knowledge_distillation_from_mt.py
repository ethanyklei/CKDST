# Standard Library
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from omegaconf import II

# My Stuff
from fairseq import metrics, utils
from fairseq.criterions import register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss
)


def compute_dkd_loss(student_logits, teacher_logits, target, padding_idx, alpha, beta, temperature, reduce=True):
    """
    student_logits: (B x S) x V
    teacher_logits: (B x S) x V
    target: (B x S) 
    """
    assert student_logits.size() == teacher_logits.size()
    if student_logits.dim() == 3:
        student_logits = student_logits.view(-1, student_logits.size(-1))
        teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
        target = target.view(-1)
    
    tgt_mask = _get_tgt_mask(student_logits, target)
    other_mask = _get_other_mask(student_logits, target)

    student_logits = student_logits.float() / temperature
    teacher_logits = teacher_logits.float() / temperature

    student_probs = utils.softmax(student_logits, dim=-1)
    teacher_probs = utils.softmax(teacher_logits, dim=-1)
    
    pad_mask = target.eq(padding_idx)
    teacher_probs = torch.masked_fill(teacher_probs, pad_mask.unsqueeze(-1), 0.0)

    student_probs = cat_mask(student_probs, tgt_mask, other_mask)
    teacher_probs = cat_mask(teacher_probs, tgt_mask, other_mask)

    student_lprobs = torch.log(student_probs)

    tckd_loss = -(teacher_probs*student_lprobs) * (temperature**2)
    if reduce:
        tckd_loss = tckd_loss.sum()


    teacher_logits = torch.masked_fill(teacher_logits, tgt_mask, -1e4 if teacher_logits.dtype == torch.float16 else -1e8)
    teacher_probs_part2 = utils.softmax(teacher_logits, dim=-1)
    teacher_probs_part2 = torch.masked_fill(teacher_probs_part2, pad_mask.unsqueeze(-1), 0.0)

    student_logits = torch.masked_fill(student_logits, tgt_mask, -1e4 if student_logits.dtype == torch.float16 else -1e8)
    student_lprobs_part2 = utils.log_softmax(student_logits, dim=-1)

    nckd_loss = -(teacher_probs_part2*student_lprobs_part2) * (temperature**2)
    if reduce:
        nckd_loss = nckd_loss.sum()

    return alpha * tckd_loss + beta * nckd_loss

def _get_tgt_mask(logits, target):
    mask = torch.zeros_like(logits).scatter_(1, target.unsqueeze(1), 1).bool()
    return mask

def _get_other_mask(logits, target):
    mask = torch.ones_like(logits).scatter_(1, target.unsqueeze(1), 0).bool()
    return mask

def cat_mask(t, mask1, mask2):
    t1 = (t * mask1).sum(dim=1, keepdims=True)
    t2 = (t * mask2).sum(dim=1, keepdims=True)
    rt = torch.cat([t1, t2], dim=1)
    return rt

@dataclass
class S2TDecoupleKnowledgeDistillationFromMTConfig(LabelSmoothedCrossEntropyCriterionConfig):
    dkd_weight: float = field(
        default=0.8
    )
    dkd_warmup: int = field(
        default=4000
    )
    tckd_weight: float = field(
        default=1.0
    )
    nckd_weight: float = field(
        default=1.0
    )
    temperature: float = field(
        default=1.0
    )
    mt_weight: float = field(
        default=1.0
    )


@register_criterion(
    "s2t_decouple_knowledge_distillation_from_mt", dataclass=S2TDecoupleKnowledgeDistillationFromMTConfig
)
class S2TDecoupleKnowledgeDistillationFromMTCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, 
                task, sentence_avg, 
                label_smoothing, 
                ignore_prefix_size=0, 
                report_accuracy=False,
                dkd_weight=0.8,
                dkd_warmup=8000,
                tckd_weight=1.0,
                nckd_weight=4.0,
                temperature=1.0,
                mt_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.dkd_warmup = dkd_warmup
        self.dkd_weight = dkd_weight
        self.tckd_weight = tckd_weight
        self.nckd_weight = nckd_weight
        self.temperature = temperature
        self.mt_weight = mt_weight

    def forward(self, model, sample, reduce=True):
        st_loss = torch.tensor(0.0)
        mt_loss = torch.tensor(0.0)
        dkd_loss = torch.tensor(0.0)

        net_output = model(**sample["net_input"])

        st_loss, nll_loss = self.compute_loss(model, net_output, sample)

        if self.training:
            mt_net_output = model(sample['src_txt_tokens'],
                                    sample['src_txt_lengths'],
                                    sample['net_input']['prev_output_tokens'],
                                    mode="text")
            if self.mt_weight > 0:
                mt_loss, _ = self.compute_loss(model, mt_net_output, sample)
                mt_loss *= self.mt_weight

            if self.dkd_weight > 0:
                dkd_loss = self._compute_dkd_loss(
                                    st_logits=net_output[0],
                                    mt_logits=mt_net_output[0].detach(),
                                    sample=sample,
                                    reduce=reduce
                                )
                dkd_weight = min((model.num_updates / self.dkd_warmup), 1) * self.dkd_weight
                dkd_loss = dkd_loss * dkd_weight
        else:
            dkd_weight = 0.0
        
        loss = (1 - dkd_weight) * st_loss + dkd_loss + mt_loss

        sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "st_loss": utils.item(st_loss.data),
            "mt_loss": utils.item(mt_loss.data),
            "dkd_loss": utils.item(dkd_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def _compute_dkd_loss(self, st_logits, mt_logits, sample, reduce):
        target = sample['target']

        if self.ignore_prefix_size > 0:
            st_logits = st_logits[:, self.ignore_prefix_size:, :].contiguous()
            mt_logits = mt_logits[:, self.ignore_prefix_size:, :].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()
        
        dkd_loss = compute_dkd_loss(
            student_logits=st_logits,
            teacher_logits=mt_logits.detach(),
            target=target,
            padding_idx=self.padding_idx,
            alpha=self.tckd_weight,
            beta=self.nckd_weight,
            temperature=self.temperature,
            reduce=reduce,
        )
        return dkd_loss
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        dkd_loss_sum = sum(log.get("dkd_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)

        metrics.log_scalar(
            "loss", loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "nll_loss", nll_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_derived(
            "ppl", lambda meters: utils.get_perplexity(meters["nll_loss"].avg)
        )
        metrics.log_scalar(
            "st_loss", st_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dkd_loss", dkd_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "mt_loss", mt_loss_sum / sample_size / math.log(2), sample_size, round=3
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