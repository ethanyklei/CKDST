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


def compute_kd_loss(student_logits, teacher_logits, temperature, target, padding_idx, reduce=True):
        """
        kd = \sum p_t*log(p_t) - p_t*log(p_s)
        """
        assert student_logits.size() == teacher_logits.size()
        if student_logits.dim() == 3:
            student_logits = student_logits.view(-1, student_logits.size(-1))
            teacher_logits = teacher_logits.view(-1, teacher_logits.size(-1))
            target = target.view(-1)
        
        teacher_probs = F.softmax((teacher_logits.float() / temperature), dim=-1)
        student_lprobs = utils.log_softmax((student_logits.float() / temperature), dim=-1)

        padding_mask = target.eq(padding_idx)
        teacher_probs = teacher_probs.masked_fill_(padding_mask.unsqueeze(-1), 0.0)
        
        if reduce:
            kd_loss = -(teacher_probs*student_lprobs).sum()
        else:
            kd_loss = -(teacher_probs*student_lprobs).sum(-1, keepdim=True)
            
        return kd_loss

@dataclass
class S2TOnlineKnowledgeDistillationFromMTConfig(LabelSmoothedCrossEntropyCriterionConfig):

    alpha: float = field(
        default=0.8
    )
    temperature: float = field(
        default=1.0
    )
    kd_warmup: int = field(
        default=4000
    )
    mt_weight: float = field(
        default=1.0
    )


@register_criterion(
    "s2t_online_knowledge_distillation_from_mt", dataclass=S2TOnlineKnowledgeDistillationFromMTConfig
)
class S2TOnlineKnowledgeDistillationFromMTCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, 
                task, sentence_avg, 
                label_smoothing, 
                ignore_prefix_size=0, 
                report_accuracy=False,
                alpha=0,
                temperature=1.0,
                kd_warmup=0,
                mt_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.alpha = alpha
        self.temperature = temperature
        self.kd_warmup = kd_warmup
        self.mt_weight = mt_weight

    
    def forward(self, model, sample, reduce=True):
        is_st_dataset = (sample["net_input"]["mode"] == "speech") # st mode dataset

        net_output = model(**sample["net_input"])

        st_loss = torch.tensor(0.0)
        mt_loss = torch.tensor(0.0)
        kd_loss = torch.tensor(0.0)

        if is_st_dataset:
            st_loss, nll_loss = self.compute_loss(model, net_output, sample)

        else:
            # extra mt loss
            mt_loss, nll_loss = self.compute_loss(model, net_output, sample)

        if self.training and is_st_dataset: # st training step
            mt_net_output = model(sample['src_txt_tokens'],
                                    sample['src_txt_lengths'],
                                    sample['net_input']['prev_output_tokens'],
                                    mode="text")                

            if self.mt_weight > 0:
                mt_loss, _ = self.compute_loss(model, mt_net_output, sample)
                mt_loss *= self.mt_weight

            kd_weight = 0.0
            if self.alpha > 0.0:            
                kd_loss = self._compute_kd_loss(
                                   net_output[0] , 
                                   mt_net_output[0].detach(), 
                                   sample,
                                   reduce=reduce)

                if self.kd_warmup > model.num_updates:
                    kd_weight = 0.0
                else:
                    kd_weight = self.alpha
                
        else:
            kd_weight = 0.0
            
        loss = (1.0 - kd_weight) * st_loss + kd_weight * kd_loss + mt_loss
            
        sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "st_loss": utils.item(st_loss.data),
            "mt_loss": utils.item(mt_loss.data),
            "kd_loss": utils.item(kd_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output
    
    def _compute_kd_loss(self, st_logits, mt_logits, sample, reduce=True):
        target = sample['target']

        if self.ignore_prefix_size > 0:
            st_logits = st_logits[:, self.ignore_prefix_size: ,:].contiguous()
            mt_logits = mt_logits[:, self.ignore_prefix_size: ,:].contiguous()
            target = target[:, self.ignore_prefix_size:].contiguous()

        kd_loss = compute_kd_loss(
            student_logits=st_logits,
            teacher_logits=mt_logits,
            temperature=self.temperature,
            target=target,
            padding_idx=self.padding_idx,
            reduce=reduce
        )

        return kd_loss
    
    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        kd_loss_sum = sum(log.get("kd_loss", 0) for log in logging_outputs)
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
            "kd_loss", kd_loss_sum / sample_size / math.log(2), sample_size, round=3
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