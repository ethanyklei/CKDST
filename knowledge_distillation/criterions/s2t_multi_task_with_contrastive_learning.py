# Standard Library
import math
from dataclasses import dataclass, field

import torch
import torch.nn.functional as F
from omegaconf import II

# My Stuff
from fairseq import metrics, utils
from fairseq.criterions import FairseqCriterion, register_criterion
from fairseq.criterions.label_smoothed_cross_entropy import (
    LabelSmoothedCrossEntropyCriterion,
    LabelSmoothedCrossEntropyCriterionConfig,
    label_smoothed_nll_loss
)


def get_sequence_representation(hidden_states, padding_mask):
    
    # padding_mask = (~padding_mask).float()
    padding_mask = (~padding_mask)
    seq_hidden = (hidden_states * padding_mask.unsqueeze(-1)).sum(dim=1)
    seq_hidden = seq_hidden / padding_mask.sum(dim=1).unsqueeze(-1)
    return seq_hidden


def compute_contrastive_loss(representation_a, padding_mask_a, representation_b, padding_mask_b, temperature, reduce=True):

    seq_rep_a = get_sequence_representation(representation_a, padding_mask_a)
    seq_rep_b = get_sequence_representation(representation_b, padding_mask_b)

    assert seq_rep_a.size() == seq_rep_b.size()

    bsz, hsize = seq_rep_a.size()
    
    logits = F.cosine_similarity(
        seq_rep_a.expand(bsz, bsz, hsize),
        seq_rep_b.expand(bsz, bsz, hsize).transpose(0, 1),
        dim=-1
    )

    logits /= temperature

    loss = -utils.log_softmax(logits.float(), dim=0).diag()

    if reduce:
        loss = loss.sum()
    
    return loss

@dataclass 
class S2TMultiTaskWithContrastiveLearningCriterionConfig(LabelSmoothedCrossEntropyCriterionConfig):
    mt_weight: float = field(
        default=1.0
    )
    asr_weight: float = field(
        default=1.0
    )
    contrastive_weight: float = field(
        default=1.0
    )
    contrastive_temperature: float = field(
        default=1.0
    )
    freeze_text: bool = field(
        default=False
    )
    dual_contrastive: bool = field(
        default=False
    )


@register_criterion(
    "s2t_multi_task_with_contrastive_learning",
    dataclass=S2TMultiTaskWithContrastiveLearningCriterionConfig
)
class S2TMultiTaskWithContrastiveLearningCriterion(LabelSmoothedCrossEntropyCriterion):
    def __init__(self, task, sentence_avg, label_smoothing, 
                ignore_prefix_size=0, report_accuracy=False, 
                mt_weight=1.0,
                asr_weight=1.0,
                contrastive_weight=1.0,
                contrastive_temperature=1.0,
                freeze_text=False,
                dual_contrastive=False):
        super().__init__(task, sentence_avg, label_smoothing, ignore_prefix_size, report_accuracy)
        self.mt_weight = mt_weight
        self.asr_weight = asr_weight
        self.contrastive_weight = contrastive_weight
        self.contrastive_temperature = contrastive_temperature
        self.freeze_text = freeze_text
        self.dual_contrastive = dual_contrastive
    
    def forward(self, model, sample, reduce=True):
        
        st_loss = torch.tensor(0.0)
        asr_loss = torch.tensor(0.0)
        mt_loss = torch.tensor(0.0)
        ctl_loss = torch.tensor(0.0)

        is_st_dataset = sample['net_input']['mode'] == "speech"

        net_output = model(**sample['net_input'])

        if is_st_dataset:
            st_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        else:
            mt_loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)
        
        if self.training and is_st_dataset:
            if self.mt_weight > 0.0:
                mt_net_output = model(
                                sample['src_txt_tokens'], sample['src_txt_lengths'],
                                sample['net_input']['prev_output_tokens'], 
                                mode="text"
                                )
                mt_loss, _  = self.compute_loss(model, mt_net_output, sample, reduce=reduce)
                mt_loss *= self.mt_weight
            
            if self.asr_weight > 0.0:
                asr_loss, _ = self.compute_asr_loss(model, net_output, sample, reduce=reduce)
                asr_loss *= self.asr_weight
    
            if self.contrastive_weight > 0.0:
                st_features = net_output[1]['encoder_out']['encoder_embedding'][0].contiguous()
                st_padding_mask = net_output[1]['encoder_out']['encoder_padding_mask'][0]
                mt_features = mt_net_output[1]['encoder_out']['encoder_embedding'][0].contiguous()
                mt_padding_mask = mt_net_output[1]['encoder_out']['encoder_padding_mask'][0]

                ctl_loss = self._compute_ctl_loss(
                                            st_features=st_features,
                                            st_padding_mask=st_padding_mask, 
                                            mt_features=mt_features.detach() if self.freeze_text else mt_features,
                                            mt_padding_mask=mt_padding_mask,
                                            reduce=reduce)
                ctl_loss *= self.contrastive_weight

        loss = st_loss + mt_loss + ctl_loss + asr_loss
        
        sample_size = (
            sample["target"].size(0) if self.sentence_avg else sample["ntokens"]
        )
        if is_st_dataset:
            st_ntokens = sample['ntokens']
            st_nsentences = sample['target'].size(0)
            src_ntokens = sample["src_ntokens"]
        else:
            st_ntokens = 0
            st_nsentences = 0
            src_ntokens = 0

        logging_output = {
            "loss": utils.item(loss.data),
            "nll_loss": utils.item(nll_loss.data),
            "st_loss": utils.item(st_loss.data),
            "mt_loss": utils.item(mt_loss.data),
            "ctl_loss": utils.item(ctl_loss.data),
            "ntokens": sample["ntokens"],
            "nsentences": sample["target"].size(0),
            "sample_size": sample_size,
            "st_ntokens": st_ntokens,
            "st_nsentences": st_nsentences,
            "src_ntokens": src_ntokens,
        }
        if self.report_accuracy:
            n_correct, total = self.compute_accuracy(model, net_output, sample)
            logging_output["n_correct"] = utils.item(n_correct.data)
            logging_output["total"] = utils.item(total.data)
        return loss, sample_size, logging_output

    def compute_mt_loss(self, model, sample, reduce=True):

        net_output = model(
                    sample['src_txt_tokens'], sample['src_txt_lengths'],
                    sample['net_input']['prev_output_tokens'], 
                    mode="text"
        )
        loss, nll_loss = self.compute_loss(model, net_output, sample, reduce=reduce)

        return loss

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

    def _compute_ctl_loss(self, st_features, st_padding_mask, mt_features, mt_padding_mask, reduce=True):
        
        ctl_loss = compute_contrastive_loss(
                            st_features,
                            st_padding_mask,
                            mt_features,
                            mt_padding_mask,
                            self.contrastive_temperature,
                            reduce=reduce,
        )
        
        return ctl_loss

    def compute_accuracy(self, model, net_output, sample):
        lprobs, target = self.get_lprobs_and_target(model, net_output, sample)
        mask = target.ne(self.padding_idx)
        n_correct = torch.sum(
            lprobs.argmax(1).masked_select(mask).eq(target.masked_select(mask))
        )
        total = torch.sum(mask)
        return n_correct, total
    

    @classmethod
    def reduce_metrics(cls, logging_outputs) -> None:
        """Aggregate logging outputs from data parallel training."""
        loss_sum = sum(log.get("loss", 0) for log in logging_outputs)
        nll_loss_sum = sum(log.get("nll_loss", 0) for log in logging_outputs)
        st_loss_sum = sum(log.get("st_loss", 0) for log in logging_outputs)
        mt_loss_sum = sum(log.get("mt_loss", 0) for log in logging_outputs)
        ntokens = sum(log.get("ntokens", 0) for log in logging_outputs)
        sample_size = sum(log.get("sample_size", 0) for log in logging_outputs)
        nsentences = sum(log.get("nsentences", 0) for log in logging_outputs)
        ctl_loss_sum = sum(log.get("ctl_loss", 0) for log in logging_outputs)
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
            "mt_loss", mt_loss_sum / ntokens / math.log(2), ntokens, round=3
        )
        metrics.log_scalar(
            "ctl_loss", ctl_loss_sum / st_nsentences / math.log(2), st_nsentences, round=3
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