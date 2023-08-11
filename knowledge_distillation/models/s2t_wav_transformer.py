# Standard Library
import contextlib
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from omegaconf import II
from torch import Tensor

# My Stuff
from fairseq import checkpoint_utils, utils
from fairseq.data.data_utils import lengths_to_padding_mask
from fairseq.dataclass import FairseqDataclass
from fairseq.models import (
    FairseqEncoder,
    FairseqEncoderDecoderModel,
    register_model
)
from fairseq.models.speech_to_text.s2t_transformer import Conv1dSubsampler
from fairseq.models.transformer import Embedding, Linear, TransformerDecoder
from fairseq.models.wav2vec import MASKING_DISTRIBUTION_CHOICES
from fairseq.modules import (
    FairseqDropout,
    LayerNorm,
    PositionalEmbedding,
    TransformerEncoderLayer
)

logger = logging.getLogger(__name__)


@dataclass
class Wav2VecConfig(FairseqDataclass):

    w2v_path: Optional[str] = field(
        default=None
    )
    freeze_w2v: bool = field(
        default=False
    )
    override_w2v_args: bool = field(
        default=False,
    )
    #
    dropout_input: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the input (after feat extr)"},
    )
    dropout: float = field(
        default=0.0, metadata={"help": "dropout probability inside wav2vec 2.0 model"}
    )
    attention_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability for attention weights inside wav2vec 2.0 model"
        },
    )
    activation_dropout: float = field(
        default=0.0,
        metadata={
            "help": "dropout probability after activation in FFN inside wav2vec 2.0 model"
        },
    )
    dropout_features: float = field(
        default=0.0,
        metadata={"help": "dropout to apply to the features (after feat extr)"},
    )
    layerdrop: float = field(
        default=0.0, metadata={"help": "probability of dropping a layer in wav2vec 2.0"}
    )
    final_dropout: float = field(
        default=0.0
    )
    # masking
    apply_mask: bool = field(
        default=False, metadata={"help": "apply masking during fine-tuning"}
    )
    mask_length: int = field(
        default=10, metadata={"help": "repeat the mask indices multiple times"}
    )
    mask_prob: float = field(
        default=0.5,
        metadata={
            "help": "probability of replacing a token with mask (normalized by length)"
        },
    )
    mask_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static", metadata={"help": "how to choose masks"}
    )
    mask_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indices"
        },
    )
    no_mask_overlap: bool = field(
        default=False, metadata={"help": "whether to allow masks to overlap"}
    )
    mask_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    require_same_masks: bool = field(
        default=True,
        metadata={
            "help": "whether to number of masked timesteps must be the same across all "
            "examples in a batch"
        },
    )
    mask_dropout: float = field(
        default=0.0,
        metadata={"help": "percent of masks to unmask for each sample"},
    )

    # channel masking
    mask_channel_length: int = field(
        default=10, metadata={"help": "length of the mask for features (channels)"}
    )
    mask_channel_prob: float = field(
        default=0.0, metadata={"help": "probability of replacing a feature with 0"}
    )
    mask_channel_selection: MASKING_DISTRIBUTION_CHOICES = field(
        default="static",
        metadata={"help": "how to choose mask length for channel masking"},
    )
    mask_channel_other: float = field(
        default=0,
        metadata={
            "help": "secondary mask argument (used for more complex distributions), "
            "see help in compute_mask_indicesh"
        },
    )
    no_mask_channel_overlap: bool = field(
        default=False, metadata={"help": "whether to allow channel masks to overlap"}
    )
    freeze_finetune_updates: int = field(
        default=0, metadata={"help": "dont finetune wav2vec for this many updates"}
    )
    feature_grad_mult: float = field(
        default=0.0, metadata={"help": "reset feature grad mult in wav2vec 2.0 to this"}
    )
    mask_channel_min_space: Optional[int] = field(
        default=1,
        metadata={"help": "min space between spans (if no overlap is enabled)"},
    )
    mask_channel_before: bool = False

    data: str = II("task.data")
    # this holds the loaded wav2vec args
    w2v_args: Any = None


@dataclass
class S2TWAVTranformerConfig(FairseqDataclass):
    share_all_embeddings: bool = field(
        default=True,
    )
    load_pretrained_encoder_from: Optional[str] = field(
        default=None
    )
    load_pretrained_decoder_from: Optional[str] = field(
        default=None
    )
    # wav2vec args
    w2v_cfg: Wav2VecConfig = Wav2VecConfig()
    # subsampler args
    conv_kernel_sizes: str = field(
        default="5,5"
    )
    conv_channels: int = field(
        default=1024
    )
    # transformer args
    activation_fn: str = field(
        default="relu"
    )
    dropout: float = field(
        default=0.1
    )
    attention_dropout: float = field(
        default=0.1
    )
    activation_dropout: float = field(
        default=0.0
    )
    encoder_embed_dim: int = field(
        default=512
    )
    encoder_ffn_embed_dim: int = field(
        default=2048
    )
    encoder_layers: int = field(
        default=6
    )
    encoder_attention_heads: int = field(
        default=8
    )
    encoder_normalize_before: bool = field(
        default=False
    )
    encoder_learned_pos: bool = field(
        default=False
    )
    decoder_embed_dim: int = field(
        default=512
    )
    decoder_ffn_embed_dim: int = field(
        default=2048
    )
    decoder_layers: int = field(
        default=6
    )
    decoder_attention_heads: int = field(
        default=8
    )
    decoder_normalize_before: bool = field(
        default=False
    )
    decoder_learned_pos: bool = field(
        default=False
    )
    adaptive_softmax_cutoff: Optional[str] = field(
        default=None
    )
    adaptive_softmax_dropout: float = field(
        default=0.0
    )
    share_decoder_input_output_embed: bool = field(
        default=True
    )
    no_token_positional_embeddings: bool = field(
        default=False
    )
    adaptive_input: bool = field(
        default=False
    )
    encoder_layerdrop: float = field(
        default=0.0
    )
    decoder_layerdrop: float = field(
        default=0.0
    )
    decoder_output_dim: int = field(
        default=512
    )
    decoder_input_dim: int = field(
        default=512
    )
    no_scale_embedding: bool = field(
        default=False
    )
    layernorm_embedding: bool = field(
        default=False
    )

    # Inherited args
    max_text_positions: int = II('task.max_text_positions')
    max_source_positions: int = II("task.max_source_positions")
    max_target_positions: int = II("task.max_target_positions")


@register_model("s2t_wav_transformer", dataclass=S2TWAVTranformerConfig)
class S2TWAVTranformerModel(FairseqEncoderDecoderModel):

    def __init__(self, encoder, decoder):
        super().__init__(encoder, decoder)

    def max_positions(self):
        return None  # it is provided in task

    @classmethod
    def build_encoder(cls, args, task, embed_tokens):
        encoder = WavEncoder(args, task, embed_tokens)
        pretraining_path = getattr(args, "load_pretrained_encoder_from", None)
        if pretraining_path is not None and pretraining_path != "":
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                encoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=encoder, checkpoint=pretraining_path, strict=False
                )
                logger.info(f"loaded pretrained encoder from: {pretraining_path}")
        return encoder

    @classmethod
    def build_decoder(cls, args, dictionary, embed_tokens):
        decoder = WavDecoder(args, dictionary, embed_tokens)
        pretraining_path = getattr(args, "load_pretrained_decoder_from", None)
        if pretraining_path is not None and pretraining_path != "":
            if not Path(pretraining_path).exists():
                logger.warning(
                    f"skipped pretraining because {pretraining_path} does not exist"
                )
            else:
                decoder = checkpoint_utils.load_pretrained_component_from_model(
                    component=decoder, checkpoint=pretraining_path
                )
                logger.info(f"loaded pretrained decoder from: {pretraining_path}")
        return decoder

    @classmethod
    def build_model(cls, args, task):
        "build a new model instance"

        def build_embedding(dictionary, embed_dim):
            num_embeddings = len(dictionary)
            padding_idx = dictionary.pad()
            return Embedding(num_embeddings, embed_dim, padding_idx)

        text_src_dict, tgt_dict = task.text_source_dictionary, task.target_dictionary

        if args.share_all_embeddings:
            if text_src_dict != tgt_dict:
                raise ValueError("--share-all-embeddings requires a joined dictionary")
            encoder_embed_tokens = build_embedding(
                text_src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = encoder_embed_tokens
            args.share_decoder_input_output_embed = True
        else:
            encoder_embed_tokens = build_embedding(
                text_src_dict, args.encoder_embed_dim
            )
            decoder_embed_tokens = build_embedding(
                tgt_dict, args.decoder_embed_dim
            )

        encoder = cls.build_encoder(args, task, encoder_embed_tokens)
        decoder = cls.build_decoder(args, tgt_dict, decoder_embed_tokens)
        return cls(encoder, decoder)

    def get_normalized_probs(
        self,
        net_output: Tuple[Tensor, Optional[Dict[str, List[Optional[Tensor]]]]],
        log_probs: bool,
        sample: Optional[Dict[str, Tensor]] = None
    ):
        lprobs = self.get_normalized_probs_scriptable(net_output, log_probs)
        lprobs.batch_first = True
        return lprobs

    def forward(
            self,
            src_tokens,
            src_lengths,
            prev_output_tokens,
            **kwargs):
        encoder_out = self.encoder(src_tokens=src_tokens, src_lengths=src_lengths, **kwargs)
        decoder_out = self.decoder(
            prev_output_tokens=prev_output_tokens, encoder_out=encoder_out
        )
        return decoder_out

    def set_num_updates(self, num_updates):
        """Set the number of parameters updates."""
        super().set_num_updates(num_updates)
        self.num_updates = num_updates


class WavEncoder(FairseqEncoder):

    def __init__(
        self,
        args,
        task,
        embed_tokens
    ):
        super().__init__(None)

        self.args = args

        self.text_dictionary = task.text_source_dictionary
        self.eos_idx = self.text_dictionary.eos()

        # speech encoder
        self._load_w2v_model(args.w2v_cfg)
        self.subsample = Conv1dSubsampler(
            self.w2v_output_dim,
            args.conv_channels,
            args.encoder_embed_dim,
            [int(k) for k in args.conv_kernel_sizes.split(",")]
        )

        self.embed_tokens = embed_tokens
        self.padding_idx = embed_tokens.padding_idx
        assert embed_tokens.embedding_dim == args.encoder_embed_dim

        self.droput_module = FairseqDropout(
            p=args.dropout, module_name=self.__class__.__name__
        )
        self.embed_scale = math.sqrt(args.encoder_embed_dim)
        if args.no_scale_embedding:
            self.embed_scale = 1.0

        self.embed_positions = PositionalEmbedding(
            args.max_text_positions,
            args.encoder_embed_dim,
            self.padding_idx,
            args.encoder_learned_pos
        )

        if getattr(args, "layernorm_embedding", False):
            self.layernorm_embedding = LayerNorm(args.encoder_embed_dim)
        else:
            self.layernorm_embedding = None

        # shared encoder
        self.layers = nn.ModuleList(
            [TransformerEncoderLayer(args) for _ in range(args.encoder_layers)]
        )

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(args.encoder_embed_dim)
        else:
            self.layer_norm = None

    def _load_w2v_model(self, args):
        # My Stuff
        from fairseq import models, tasks

        assert args.w2v_path is not None
        if args.override_w2v_args:
            arg_overrides = {
                "dropout": args.dropout,
                "activation_dropout": args.activation_dropout,
                "dropout_input": args.dropout_input,
                "attention_dropout": args.attention_dropout,
                "encoder_layerdrop": args.layerdrop,
                "dropout_features": args.dropout_features,
                "mask_length": args.mask_length,
                "mask_prob": args.mask_prob,
                "require_same_masks": getattr(args, "require_same_masks", True),
                "pct_holes": getattr(args, "mask_dropout", 0),
                "mask_selection": args.mask_selection,
                "mask_other": args.mask_other,
                "no_mask_overlap": args.no_mask_overlap,
                "mask_channel_length": args.mask_channel_length,
                "mask_channel_prob": args.mask_channel_prob,
                "mask_channel_before": args.mask_channel_before,
                "mask_channel_selection": args.mask_channel_selection,
                "mask_channel_other": args.mask_channel_other,
                "no_mask_channel_overlap": args.no_mask_channel_overlap,
                "feature_grad_mult": args.feature_grad_mult,
            }
        else:
            arg_overrides = {}

        state = checkpoint_utils.load_checkpoint_to_cpu(args.w2v_path, arg_overrides)
        w2v_args = state.get("cfg", None)
        if w2v_args is None:
            w2v_args = convert_namespace_to_omegaconf(state["args"])
        w2v_args.criterion = None
        w2v_args.lr_scheduler = None
        args.w2v_args = w2v_args

        w2v_args.task.data = args.data
        task = tasks.setup_task(w2v_args.task)
        model = task.build_model(w2v_args.model, from_checkpoint=True)

        model.remove_pretraining_modules()

        model.load_state_dict(state['model'], strict=True)

        self.w2v_model = model
        self.freeze_w2v = args.freeze_w2v
        self.apply_mask = args.apply_mask
        self.w2v_output_dim = w2v_args.model.encoder_embed_dim

    def _get_w2v_features(self, src_tokens, src_lengths):
        padding_mask = lengths_to_padding_mask(src_lengths)
        if src_tokens.size(1) > padding_mask.size(1):
            src_tokens = src_tokens[:, :padding_mask.size(1)]

        if self.freeze_w2v:
            with torch.no_grad():
                w2v_outputs = self.w2v_model.extract_features(
                    src_tokens, padding_mask, mask=(self.apply_mask and self.training))
        else:
            w2v_outputs = self.w2v_model.extract_features(src_tokens, padding_mask, mask=(self.apply_mask and self.training))

        # B x S x H
        w2v_features = w2v_outputs['x']
        padding_mask = w2v_outputs['padding_mask']
        if padding_mask is not None:
            output_lengths = (1 - padding_mask.int()).sum(dim=-1)
        else:
            output_lengths = w2v_features.new_full((w2v_features.size(0),), w2v_features.size(1), dtype=torch.long)

        return w2v_features, padding_mask, output_lengths
    
    def embed_sph(self, src_tokens, src_lengths):

        w2v_features, _, w2v_output_lengths = self._get_w2v_features(src_tokens=src_tokens, src_lengths=src_lengths)

        w2v_features, input_lengths = self.subsample(w2v_features, w2v_output_lengths)  # slen x bsz x hsize
        w2v_features = w2v_features.transpose(0, 1) # bsz x slen x hsize
       
        encoder_padding_mask = lengths_to_padding_mask(input_lengths)

        x = embed = self.embed_scale * w2v_features
        if self.embed_positions is not None:
            x = embed + self.embed_positions(encoder_padding_mask)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.droput_module(x)

        return x, encoder_padding_mask
    
    def embed_text(self, src_tokens, src_lengths):
        encoder_padding_mask = src_tokens.eq(self.padding_idx)

        token_embedding = self.embed_tokens(src_tokens)
        x = embed = self.embed_scale * token_embedding
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding is not None:
            x = self.layernorm_embedding(x)
        x = self.droput_module(x)

        return x, encoder_padding_mask


    def forward(self, src_tokens, src_lengths, mode="speech", **kwargs):
      
        if src_tokens.is_floating_point():

            x, encoder_padding_mask = self.embed_sph(src_tokens, src_lengths)

        else:

            x, encoder_padding_mask = self.embed_text(src_tokens, src_lengths)

        # else:
        #     raise ValueError(f"encoder get an unknow mode: {mode}")

        encoder_embedding = x

        # B x S x H -> S x B x H
        x = x.transpose(0, 1)

        for layer in self.layers:
            x = layer(x, encoder_padding_mask=encoder_padding_mask)

        if self.layer_norm is not None:
            x = self.layer_norm(x)

        return {
            "encoder_out": [x],  # T x B x C
            "encoder_padding_mask": [encoder_padding_mask],  # B x T
            "encoder_embedding": [encoder_embedding],  # B x T x C
            "encoder_states": [],  # List[T x B x C]
            "src_tokens": [],
            "src_lengths": [],
        }
    
    def reorder_encoder_out(self, encoder_out, new_order):
        new_encoder_out = (
            []
            if len(encoder_out["encoder_out"]) == 0
            else [x.index_select(1, new_order) for x in encoder_out["encoder_out"]]
        )

        new_encoder_padding_mask = (
            []
            if len(encoder_out["encoder_padding_mask"]) == 0
            else [
                x.index_select(0, new_order)
                for x in encoder_out["encoder_padding_mask"]
            ]
        )

        new_encoder_embedding = (
            []
            if len(encoder_out["encoder_embedding"]) == 0
            else [
                x.index_select(0, new_order) for x in encoder_out["encoder_embedding"]
            ]
        )

        encoder_states = encoder_out["encoder_states"]
        if len(encoder_states) > 0:
            for idx, state in enumerate(encoder_states):
                encoder_states[idx] = state.index_select(1, new_order)

        return {
            "encoder_out": new_encoder_out,  # T x B x C
            "encoder_padding_mask": new_encoder_padding_mask,  # B x T
            "encoder_embedding": new_encoder_embedding,  # B x T x C
            "encoder_states": encoder_states,  # List[T x B x C]
            "src_tokens": [],  # B x T
            "src_lengths": [],  # B x 1
        }


class WavDecoder(TransformerDecoder):

    def extract_features(
        self,
        prev_output_tokens,
        encoder_out: Optional[Dict[str, List[Tensor]]] = None,
        incremental_state: Optional[Dict[str, Dict[str, Optional[Tensor]]]] = None,
        full_context_alignment: bool = False,
        alignment_layer: Optional[int] = None,
        alignment_heads: Optional[int] = None,
    ):
        # call scriptable method from parent class
        x, extra = self.extract_features_scriptable(
            prev_output_tokens,
            encoder_out,
            incremental_state,
            full_context_alignment,
            alignment_layer,
            alignment_heads,
        )
        if incremental_state is None:
            extra["encoder_out"] = encoder_out
        return x, extra