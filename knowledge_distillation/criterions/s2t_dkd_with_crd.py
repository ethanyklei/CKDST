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
from knowledge_distillation.criterions.s2t_contrastive_representation_distillation import (
    compute_crd_loss
)
from knowledge_distillation.criterions.s2t_decouple_knowledge_distillation_from_mt import (
    compute_dkd_loss
)
from knowledge_distillation.criterions.s2t_multi_task_with_contrastive_learning import (
    get_sequence_representation
)


@dataclass
class S2TDKDWithCRDConfig(LabelSmoothedCrossEntropyCriterionConfig):
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
    dkd_temperature: float = field(
        default=1.0
    )
    crd_weight: float = field(
        default=1.0
    )
    crd_temperature: float = field(
        default=1.0
    )
    crd_warmup: int = field(
        default=4000
    )
    crd_pos: str = field(
        default="top"
    )
    crd_freeze_tea: bool = field(
        default=True,
    )
    mt_weight: float = field(
        default=1.0
    )


@register_criterion(
    "s2t_dkd_with_crd", dataclass=S2TDKDWithCRDConfig
)
class S2TDKDWithCRDCriterion(LabelSmoothedCrossEntropyCriterion):

    def __init__(self, 
                task, sentence_avg, 
                label_smoothing, 
                ignore_prefix_size=0, 
                report_accuracy=False,
                dkd_weight=0.8,
                dkd_warmup=8000,
                tckd_weight=1.0,
                nckd_weight=4.0,
                dkd_temperature=1.0,
                crd_warmup=4000,
                crd_weight=1.0,
                crd_temperature=1.0,
                crd_pos="top",
                crd_freeze_tea=True,
                mt_weight=1.0):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.dkd_warmup = dkd_warmup
        self.dkd_weight = dkd_weight
        self.tckd_weight = tckd_weight
        self.nckd_weight = nckd_weight
        self.dkd_temperature = dkd_temperature
        self.crd_warmup = crd_warmup
        self.crd_weight = crd_weight
        self.crd_temperature = crd_temperature
        self.crd_pos = crd_pos
        self.crd_freeze_tea = crd_freeze_tea
        self.mt_weight = mt_weight
    
    def forward(self, model, sample, reduce=True):
        st_loss = torch.tensor(0.0)
        mt_loss = torch.tensor(0.0)
        dkd_loss = torch.tensor(0.0)
        crd_loss = torch.tensor(0.0)

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

            dkd_weight = min((model.num_updates / self.dkd_warmup), 1) * self.dkd_weight
            if dkd_weight > 0:
                dkd_loss = self._compute_dkd_loss(
                                    st_logits=net_output[0],
                                    mt_logits=mt_net_output[0].detach(),
                                    sample=sample,
                                    reduce=reduce
                                )
                dkd_loss = dkd_loss * dkd_weight

            crd_weight = min((model.num_updates / self.crd_warmup), 1) * self.crd_weight
            if crd_weight > 0:
                if self.crd_pos == "top":
                    crd_loss = self._compute_crd_loss(
                        st_features=net_output[1]["encoder_out"]["encoder_out"][0].transpose(0,1).contiguous(),
                        st_padding_mask=net_output[1]["encoder_out"]["encoder_padding_mask"][0],
                        mt_features=mt_net_output[1]["encoder_out"]["encoder_out"][0].transpose(0,1).contiguous(),
                        mt_padding_mask=mt_net_output[1]["encoder_out"]["encoder_padding_mask"][0],
                        reduce=reduce
                    )
                elif self.crd_pos == "bottom":
                    crd_loss = self._compute_crd_loss(
                        st_features=net_output[1]["encoder_out"]["encoder_embedding"][0],
                        st_padding_mask=net_output[1]["encoder_out"]["encoder_padding_mask"][0],
                        mt_features=mt_net_output[1]["encoder_out"]["encoder_embedding"][0],
                        mt_padding_mask=mt_net_output[1]["encoder_out"]["encoder_padding_mask"][0],
                        reduce=reduce
                    )
                crd_loss = crd_loss * crd_weight
        else:
            dkd_weight = 0.0
        
        loss = (1 - dkd_weight) * st_loss + dkd_loss + mt_loss  + crd_loss

        sample_size = (
                sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
            )
        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "st_loss": utils.item(st_loss.data),
            "mt_loss": utils.item(mt_loss.data),
            "dkd_loss": utils.item(dkd_loss.data),
            "crd_loss": utils.item(crd_loss.data),
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
            temperature=self.dkd_temperature,
            reduce=reduce,
        )
        return dkd_loss
    
    def _compute_crd_loss(self, st_features, st_padding_mask, mt_features, mt_padding_mask, reduce=True):
        st_seq = get_sequence_representation(st_features, st_padding_mask)
        mt_seq = get_sequence_representation(mt_features if not self.crd_freeze_tea else mt_features.detach(), 
                                            mt_padding_mask)

        crd_loss = compute_crd_loss(
                        st_seq,
                        mt_seq,
                        temperature=self.crd_temperature,
                        reduce=reduce
                        )
        return crd_loss 

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        dkd_loss_sum = sum(log.get("dkd_loss", 0) for log in logging_outputs)
        crd_loss_sum = sum(log.get("crd_loss", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)

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
            "mt_loss", mt_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "dkd_loss", dkd_loss_sum / sample_size / math.log(2), sample_size, round=3
        )
        metrics.log_scalar(
            "crd_loss", crd_loss_sum / nsentences / math.log(2), nsentences, round=3
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