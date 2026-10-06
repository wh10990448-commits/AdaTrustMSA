"""AdaTrustMSA: adaptive, reliable multimodal sentiment prediction.

This module uses a BERT + three BiGRU backbone so compatible baseline
checkpoints can be warm-started.  Every modality predicts sentiment,
reliability and uncertainty.  A masked transformer, explicit pairwise
interactions and conflict-aware mixture of experts then perform deep fusion.
"""

from __future__ import annotations

from typing import Dict, Optional

import torch
from torch import Tensor, nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from transformers import BertConfig, BertModel


PAD_ID, UNK_ID, CLS_ID, SEP_ID = 0, 100, 101, 102


def _masked_softmax(scores: Tensor, mask: Tensor, dim: int) -> Tensor:
    """Softmax over valid entries; fully masked rows are exactly zero."""
    mask = mask.bool()
    masked = scores.float().masked_fill(~mask, float("-inf"))
    has_values = mask.any(dim=dim, keepdim=True)
    safe_scores = torch.where(has_values, masked, torch.zeros_like(masked))
    return torch.softmax(safe_scores, dim=dim).masked_fill(~mask, 0.0)


def _masked_mean(values: Tensor, mask: Tensor, dim: int) -> Tensor:
    weights = mask.to(values.dtype)
    while weights.ndim < values.ndim:
        weights = weights.unsqueeze(-1)
    total = (values * weights).sum(dim=dim)
    count = weights.sum(dim=dim).clamp_min(1.0)
    return total / count


class _ModalityEncoder(nn.Module):
    """BERT/audio/vision temporal encoder used by the project."""

    def __init__(self, input_dim: int, hidden: int, dropout: float):
        super().__init__()
        output_dim = 2 * hidden
        self.projection = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.gru = nn.GRU(
            input_size=output_dim + 1,
            hidden_size=hidden,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: Tensor, available: Tensor, lengths_cpu: Tensor) -> Tensor:
        projected = self.projection(values)
        projected = projected.masked_fill(~available.unsqueeze(-1), 0.0)
        sequence = torch.cat(
            (projected, available.unsqueeze(-1).to(projected.dtype)), dim=-1
        )
        packed = pack_padded_sequence(
            sequence,
            lengths_cpu.clamp_min(1),
            batch_first=True,
            enforce_sorted=False,
        )
        encoded, _ = self.gru(packed)
        encoded, _ = pad_packed_sequence(
            encoded, batch_first=True, total_length=values.shape[1]
        )
        return self.dropout(encoded).masked_fill(~available.unsqueeze(-1), 0.0)


class _CrossModalInteractionBlock(nn.Module):
    """A small pre-norm transformer block over the T/A/V modality tokens.

    The implementation is deliberately independent of ``MultiheadAttention``
    so its missing-modality mask behaves identically across supported PyTorch
    versions.  Missing keys receive zero probability and missing queries are
    reset to zero after every residual branch.
    """

    def __init__(
        self, dim: int, heads: int, expansion: int, dropout: float
    ) -> None:
        super().__init__()
        if heads < 1 or dim % heads:
            raise ValueError("fusion_heads must divide the modality token dimension")
        if expansion < 1:
            raise ValueError("fusion_expansion must be positive")
        self.heads = heads
        self.head_dim = dim // heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, 3 * dim)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(dim, dim)
        self.residual_dropout = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.feed_forward = nn.Sequential(
            nn.Linear(dim, expansion * dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(expansion * dim, dim),
        )

    def forward(
        self,
        tokens: Tensor,
        present: Tensor,
        quality_bias: Optional[Tensor] = None,
    ) -> Tensor:
        batch, modalities, dim = tokens.shape
        normalized = self.norm1(tokens)
        qkv = self.qkv(normalized).view(
            batch, modalities, 3, self.heads, self.head_dim
        )
        query, key, value = qkv.unbind(dim=2)
        query = query.permute(0, 2, 1, 3)
        key = key.permute(0, 2, 1, 3)
        value = value.permute(0, 2, 1, 3)
        scores = torch.matmul(query, key.transpose(-2, -1)) * self.scale
        if quality_bias is not None:
            if quality_bias.shape != present.shape:
                raise ValueError("quality_bias and present must have matching shapes")
            # A key-side log prior turns independently estimated modality
            # quality into a direct attention preference. Centering makes the
            # result invariant to a sample-wide confidence shift.
            bias = quality_bias.float().masked_fill(~present, 0.0)
            denominator = present.sum(dim=1, keepdim=True).clamp_min(1)
            bias = bias - bias.sum(dim=1, keepdim=True) / denominator
            scores = scores + bias[:, None, None, :].to(scores.dtype)
        key_mask = present[:, None, None, :]
        attention = _masked_softmax(scores, key_mask, dim=-1)
        attention = self.attention_dropout(attention)
        context = torch.matmul(attention, value).permute(0, 2, 1, 3)
        context = context.reshape(batch, modalities, dim)
        query_mask = present.unsqueeze(-1)
        tokens = tokens + self.residual_dropout(self.output_projection(context))
        tokens = tokens.masked_fill(~query_mask, 0.0)
        tokens = tokens + self.residual_dropout(self.feed_forward(self.norm2(tokens)))
        return tokens.masked_fill(~query_mask, 0.0)


class _AttentiveStatisticsPooling(nn.Module):
    """Learn salient timesteps and retain both their mean and dispersion."""

    def __init__(self, dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.score = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1),
        )
        self.projection = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.output_norm = nn.LayerNorm(dim)

    def forward(self, sequence: Tensor, available: Tensor) -> Tensor:
        scores = self.score(sequence).squeeze(-1)
        weights = _masked_softmax(scores, available, dim=1)
        mean = (sequence * weights.unsqueeze(-1)).sum(dim=1)
        variance = (
            (sequence - mean.unsqueeze(1)).square() * weights.unsqueeze(-1)
        ).sum(dim=1)
        standard_deviation = torch.sqrt(variance.clamp_min(1e-6))
        pooled = self.output_norm(
            mean + self.projection(torch.cat((mean, standard_deviation), dim=-1))
        )
        return pooled.masked_fill(~available.any(dim=1, keepdim=True), 0.0)


class AdaTrustMSAModel(nn.Module):
    """BERT/BiGRU backbone with reliability, conflict and uncertainty heads.

    Output tensors use modality order ``text, audio, vision``.  Reliability is
    supervised with known corruption strength during training.  Conflict is the
    total-variation distance between the three unimodal class distributions.
    The final regression head is heteroscedastic and therefore emits both a
    sentiment value and a variance suitable for calibrated selective prediction.
    """

    def __init__(
        self,
        bert_config: dict,
        audio_mean: list,
        audio_std: list,
        vision_mean: list,
        vision_std: list,
        bert_pretrained_dir: Optional[str] = None,
        hidden: int = 48,
        dropout: float = 0.2,
        uncertainty_penalty: float = 0.35,
        conflict_penalty: float = 0.65,
        fusion_layers: int = 0,
        fusion_heads: int = 4,
        fusion_expansion: int = 2,
        prediction_blending: bool = False,
        attentive_pooling: bool = False,
        quality_attention: bool = False,
        quality_attention_strength: float = 1.0,
        use_reliability: bool = True,
        use_conflict: bool = True,
        use_uncertainty: bool = True,
    ):
        super().__init__()
        if hidden < 1:
            raise ValueError("hidden must be positive")
        self.hidden = hidden
        self.output_dim = 2 * hidden
        self.uncertainty_penalty = float(uncertainty_penalty)
        self.conflict_penalty = float(conflict_penalty)
        self.use_reliability = bool(use_reliability)
        self.use_conflict = bool(use_conflict)
        self.use_uncertainty = bool(use_uncertainty)
        self.prediction_blending = bool(prediction_blending)
        self.attentive_pooling = bool(attentive_pooling)
        self.quality_attention = bool(quality_attention)
        self.quality_attention_strength = float(quality_attention_strength)
        if self.quality_attention_strength < 0:
            raise ValueError("quality_attention_strength cannot be negative")
        if fusion_layers < 0:
            raise ValueError("fusion_layers cannot be negative")
        self.deep_fusion = int(fusion_layers) > 0
        if self.prediction_blending and not self.deep_fusion:
            raise ValueError("prediction_blending requires fusion_layers > 0")

        if bert_pretrained_dir is not None:
            self.bert = BertModel.from_pretrained(
                bert_pretrained_dir,
                add_pooling_layer=False,
                local_files_only=True,
            )
        else:
            self.bert = BertModel(BertConfig(**bert_config), add_pooling_layer=False)

        for name, values, width in (
            ("audio_mean", audio_mean, 74),
            ("audio_std", audio_std, 74),
            ("vision_mean", vision_mean, 35),
            ("vision_std", vision_std, 35),
        ):
            buffer = torch.as_tensor(values, dtype=torch.float32).detach().clone()
            if buffer.shape != (width,) or not torch.isfinite(buffer).all():
                raise ValueError(f"{name} must contain {width} finite values")
            if name.endswith("_std"):
                if (buffer < 0).any():
                    raise ValueError(f"{name} cannot contain negative values")
                buffer = buffer.clamp_min(1e-6)
            self.register_buffer(name, buffer)

        dim = self.output_dim
        self.text_encoder = _ModalityEncoder(self.bert.config.hidden_size, hidden, dropout)
        self.audio_encoder = _ModalityEncoder(74, hidden, dropout)
        self.vision_encoder = _ModalityEncoder(35, hidden, dropout)
        if self.attentive_pooling:
            self.temporal_poolers = nn.ModuleList(
                [
                    _AttentiveStatisticsPooling(dim, hidden, dropout)
                    for _ in range(3)
                ]
            )

        def quality_head(output: int = 1) -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(dim + 2, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, output),
            )

        self.reliability_heads = nn.ModuleList([quality_head() for _ in range(3)])
        self.uncertainty_heads = nn.ModuleList([quality_head() for _ in range(3)])
        self.fusion_score_heads = nn.ModuleList([quality_head() for _ in range(3)])
        self.unimodal_classifiers = nn.ModuleList(
            [nn.Linear(dim, 3) for _ in range(3)]
        )
        self.unimodal_regressors = nn.ModuleList(
            [nn.Linear(dim, 1) for _ in range(3)]
        )

        # Deep fusion operates on three cached modality tokens, so the exact
        # same module is used by full inference and the adaptive early-exit
        # policy.  Quality projection injects availability, reliability,
        # uncertainty and conflict into every interaction layer.
        self.modality_input_norm = nn.LayerNorm(dim)
        self.modality_embeddings = nn.Parameter(torch.empty(1, 3, dim))
        nn.init.normal_(self.modality_embeddings, mean=0.0, std=0.02)
        self.quality_projection = nn.Sequential(
            nn.Linear(4, hidden), nn.GELU(), nn.Linear(hidden, dim)
        )
        self.interaction_blocks = nn.ModuleList(
            [
                _CrossModalInteractionBlock(
                    dim=dim,
                    heads=int(fusion_heads),
                    expansion=int(fusion_expansion),
                    dropout=dropout,
                )
                for _ in range(int(fusion_layers))
            ]
        )
        self.modality_experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(dim),
                    nn.Linear(dim, dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                )
                for _ in range(3)
            ]
        )
        self.pair_encoder = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.synergy_expert = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.expert_score_heads = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(dim + 4, hidden),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden, 1),
                )
                for _ in range(4)
            ]
        )
        self.expert_classifier = nn.Linear(dim, 3)
        self.fusion_refiner = nn.Sequential(
            nn.LayerNorm(2 * dim),
            nn.Linear(2 * dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
        )
        self.fusion_output_norm = nn.LayerNorm(dim)

        # fused + availability + reliability + uncertainty + conflict
        final_dim = dim + 12
        self.classifier = nn.Sequential(
            nn.Linear(final_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 3),
        )
        self.regressor = nn.Sequential(
            nn.Linear(final_dim, dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, 1),
        )
        self.log_variance_head = nn.Sequential(
            nn.Linear(final_dim, hidden), nn.GELU(), nn.Linear(hidden, 1)
        )
        if self.prediction_blending:
            self.classification_blender = nn.Sequential(
                nn.Linear(final_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 3),
            )
            self.regression_blender = nn.Sequential(
                nn.Linear(final_dim, hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )

    @staticmethod
    def _standardize(
        values: Tensor, mean: Tensor, std: Tensor, available: Tensor
    ) -> Tensor:
        normalized = ((values.float() - mean) / std).clamp(-10.0, 10.0)
        return normalized.masked_fill(~available.unsqueeze(-1), 0.0)

    def _validate_inputs(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        audio: Tensor,
        vision: Tensor,
    ) -> None:
        if input_ids.ndim != 2 or input_ids.shape[1] == 0:
            raise ValueError("input_ids must have shape (batch, nonempty_sequence)")
        if attention_mask.shape != input_ids.shape or token_type_ids.shape != input_ids.shape:
            raise ValueError("BERT input tensors must have matching shapes")
        if audio.shape != (*input_ids.shape, 74):
            raise ValueError("audio must have shape (batch, sequence, 74)")
        if vision.shape != (*input_ids.shape, 35):
            raise ValueError("vision must have shape (batch, sequence, 35)")

    def _availability_state(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        audio: Tensor,
        vision: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        self._validate_inputs(input_ids, attention_mask, token_type_ids, audio, vision)
        attended = attention_mask.bool()
        content = attended & input_ids.ne(CLS_ID) & input_ids.ne(SEP_ID)
        text_available = content & input_ids.ne(UNK_ID) & input_ids.ne(PAD_ID)
        audio_available = content & audio.ne(0).any(dim=-1)
        vision_available = content & vision.ne(0).any(dim=-1)
        availability = torch.stack(
            (text_available, audio_available, vision_available), dim=-1
        )
        denominator = content.sum(dim=1).clamp_min(1).to(torch.float32).unsqueeze(-1)
        fractions = availability.float().sum(dim=1) / denominator
        present = fractions.gt(0)
        lengths_cpu = attended.sum(dim=1).to(device="cpu", dtype=torch.long)
        return availability, fractions, present, attended, lengths_cpu

    def encode_modality(
        self,
        modality: str,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        audio: Tensor,
        vision: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Encode exactly one modality for use by the cached adaptive policy."""
        modality = modality.upper()
        index_by_name = {"T": 0, "A": 1, "V": 2}
        if modality not in index_by_name:
            raise ValueError("modality must be one of T, A or V")
        availability, fractions, present, attended, lengths_cpu = self._availability_state(
            input_ids, attention_mask, token_type_ids, audio, vision
        )
        index = index_by_name[modality]
        available = availability[:, :, index]
        zero_pooled = audio.new_zeros(
            input_ids.shape[0], self.output_dim, dtype=torch.float32
        )
        if not available.any():
            return zero_pooled, fractions[:, index], present[:, index]

        if modality == "T":
            safe_attention = attended.clone()
            empty = ~attended.any(dim=1)
            safe_attention[empty, 0] = True
            text_vectors = self.bert(
                input_ids=input_ids,
                attention_mask=safe_attention.long(),
                token_type_ids=token_type_ids,
            ).last_hidden_state
            encoded = self.text_encoder(text_vectors, available, lengths_cpu)
        elif modality == "A":
            encoded = self.audio_encoder(
                self._standardize(audio, self.audio_mean, self.audio_std, available),
                available,
                lengths_cpu,
            )
        else:
            encoded = self.vision_encoder(
                self._standardize(vision, self.vision_mean, self.vision_std, available),
                available,
                lengths_cpu,
            )
        if self.attentive_pooling:
            pooled = self.temporal_poolers[index](encoded, available)
        else:
            pooled = _masked_mean(encoded, available, dim=1)
        return pooled, fractions[:, index], present[:, index]

    def predict_from_pooled(
        self, pooled: Tensor, fractions: Tensor, present: Tensor
    ) -> Dict[str, Tensor]:
        """Fuse cached modality states and emit predictions and uncertainty."""
        if pooled.ndim != 3 or pooled.shape[1:] != (3, self.output_dim):
            raise ValueError("pooled must have shape (batch,3,output_dim)")
        if fractions.shape != pooled.shape[:2] or present.shape != pooled.shape[:2]:
            raise ValueError("fractions and present must have shape (batch,3)")

        # Energy is a cheap sample-specific quality cue, not a hand-written
        # reliability rule.  The supervised reliability heads decide how to use it.
        energy = pooled.square().mean(dim=-1).sqrt()
        quality_features = torch.cat(
            (pooled, fractions.unsqueeze(-1), energy.unsqueeze(-1)), dim=-1
        )
        reliability_logits = torch.cat(
            [head(quality_features[:, i]).view(-1, 1) for i, head in enumerate(self.reliability_heads)],
            dim=1,
        )
        reliability = torch.sigmoid(reliability_logits) * present.to(torch.float32)
        modality_uncertainty = torch.cat(
            [
                torch.nn.functional.softplus(head(quality_features[:, i])).view(-1, 1)
                for i, head in enumerate(self.uncertainty_heads)
            ],
            dim=1,
        )
        modality_uncertainty = modality_uncertainty * present.to(torch.float32)

        modality_logits = torch.stack(
            [head(pooled[:, i]) for i, head in enumerate(self.unimodal_classifiers)],
            dim=1,
        )
        modality_regression = 3.0 * torch.tanh(
            torch.cat(
                [head(pooled[:, i]) for i, head in enumerate(self.unimodal_regressors)],
                dim=1,
            )
        )
        modality_probabilities = torch.softmax(modality_logits.float(), dim=-1)
        conflict_matrix = 0.5 * (
            modality_probabilities.unsqueeze(2) - modality_probabilities.unsqueeze(1)
        ).abs().sum(dim=-1)
        pair_available = present.unsqueeze(2) & present.unsqueeze(1)
        conflict_matrix = conflict_matrix.masked_fill(~pair_available, 0.0)
        eye = torch.eye(3, device=conflict_matrix.device, dtype=torch.bool).unsqueeze(0)
        peer_weights = reliability.unsqueeze(1).expand(-1, 3, -1).masked_fill(eye, 0.0)
        peer_weights = peer_weights * pair_available.to(peer_weights.dtype)
        modality_conflict = (conflict_matrix * peer_weights).sum(dim=-1) / peer_weights.sum(
            dim=-1
        ).clamp_min(1e-6)

        base_scores = torch.cat(
            [head(quality_features[:, i]).view(-1, 1) for i, head in enumerate(self.fusion_score_heads)],
            dim=1,
        )
        effective_reliability = reliability if self.use_reliability else present.to(torch.float32)
        effective_uncertainty = modality_uncertainty if self.use_uncertainty else torch.zeros_like(modality_uncertainty)
        effective_conflict = modality_conflict if self.use_conflict else torch.zeros_like(modality_conflict)
        fusion_scores = base_scores
        if self.use_reliability:
            fusion_scores = fusion_scores + torch.log(effective_reliability.clamp_min(1e-6))
        if self.use_uncertainty:
            fusion_scores = fusion_scores - self.uncertainty_penalty * effective_uncertainty
        if self.use_conflict:
            fusion_scores = fusion_scores - self.conflict_penalty * effective_conflict
        modality_weights = _masked_softmax(fusion_scores, present, dim=1)
        base_fused = (pooled * modality_weights.unsqueeze(-1)).sum(dim=1)

        if not self.deep_fusion:
            # Compatibility path for original v1 checkpoints, whose saved
            # model_kwargs do not contain fusion_layers. Extra outputs preserve
            # the v2 contract but do not affect the original prediction path.
            final_features = torch.cat(
                (
                    base_fused,
                    fractions,
                    effective_reliability,
                    effective_uncertainty,
                    effective_conflict,
                ),
                dim=-1,
            )
            logits = self.classifier(final_features)
            regression = 3.0 * torch.tanh(
                self.regressor(final_features).squeeze(-1)
            )
            log_variance = self.log_variance_head(final_features).squeeze(-1).clamp(-6.0, 3.0)
            variance = torch.exp(log_variance)
            confidence = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
            confidence = confidence * torch.exp(-0.5 * variance.clamp(max=6.0))
            synergy_present = present.sum(dim=1).ge(2)
            expert_present = torch.cat((present, synergy_present.unsqueeze(1)), dim=1)
            expert_weights = torch.cat(
                (
                    modality_weights,
                    modality_weights.new_zeros(modality_weights.shape[0], 1),
                ),
                dim=1,
            )
            synergy_logits = (
                modality_logits * modality_weights.unsqueeze(-1)
            ).sum(dim=1, keepdim=True)
            return {
                "logits": logits,
                "regression": regression,
                "log_variance": log_variance,
                "variance": variance,
                "confidence": confidence,
                "fractions": fractions,
                "reliability_logits": reliability_logits,
                "reliability": reliability,
                "modality_uncertainty": modality_uncertainty,
                "modality_logits": modality_logits,
                "modality_regression": modality_regression,
                "conflict_matrix": conflict_matrix,
                "modality_conflict": modality_conflict,
                "modality_weights": modality_weights,
                "interaction_tokens": pooled,
                "expert_logits": torch.cat((modality_logits, synergy_logits), dim=1),
                "expert_present": expert_present,
                "expert_weights": expert_weights,
                "classification_blend": final_features.new_tensor([1.0, 0.0, 0.0])
                .unsqueeze(0)
                .expand(final_features.shape[0], -1),
                "regression_blend": final_features.new_ones(final_features.shape[0]),
            }

        # Reliability-conditioned transformer interaction over T/A/V tokens.
        # Keeping the token count at three makes this inexpensive enough to run
        # at every prefix in the adaptive policy.
        modality_quality = torch.stack(
            (
                fractions,
                effective_reliability,
                torch.log1p(effective_uncertainty),
                effective_conflict,
            ),
            dim=-1,
        )
        interaction_tokens = (
            self.modality_input_norm(pooled)
            + self.modality_embeddings
            + self.quality_projection(modality_quality)
        )
        interaction_tokens = interaction_tokens.masked_fill(
            ~present.unsqueeze(-1), 0.0
        )
        attention_quality = (
            effective_reliability.clamp_min(1e-4)
            * torch.exp(-self.uncertainty_penalty * effective_uncertainty)
            * torch.exp(-self.conflict_penalty * effective_conflict)
        ).masked_fill(~present, 0.0)
        attention_bias = None
        if self.quality_attention:
            attention_bias = self.quality_attention_strength * torch.log(
                attention_quality.clamp_min(1e-4)
            )
        for block in self.interaction_blocks:
            interaction_tokens = block(interaction_tokens, present, attention_bias)

        modality_expert_states = torch.stack(
            [
                expert(interaction_tokens[:, index])
                for index, expert in enumerate(self.modality_experts)
            ],
            dim=1,
        ).masked_fill(~present.unsqueeze(-1), 0.0)

        # Hadamard products capture agreement, while absolute differences
        # capture complementary/conflicting evidence for all three pairs.
        pair_states = []
        pair_strengths = []
        for left, right in ((0, 1), (0, 2), (1, 2)):
            left_state = interaction_tokens[:, left]
            right_state = interaction_tokens[:, right]
            pair_state = self.pair_encoder(
                torch.cat(
                    (left_state * right_state, (left_state - right_state).abs()),
                    dim=-1,
                )
            )
            pair_present = present[:, left] & present[:, right]
            pair_state = pair_state.masked_fill(~pair_present.unsqueeze(-1), 0.0)
            strength = (
                effective_reliability[:, left]
                * effective_reliability[:, right]
                * torch.exp(
                    -0.5
                    * (
                        effective_uncertainty[:, left]
                        + effective_uncertainty[:, right]
                    )
                )
                * (1.0 - conflict_matrix[:, left, right]).clamp_min(0.0)
            )
            pair_states.append(pair_state)
            pair_strengths.append(strength * pair_present.to(strength.dtype))
        pair_states = torch.stack(pair_states, dim=1)
        pair_strengths = torch.stack(pair_strengths, dim=1)
        pair_context = (
            pair_states * pair_strengths.unsqueeze(-1)
        ).sum(dim=1) / pair_strengths.sum(dim=1, keepdim=True).clamp_min(1e-6)
        interaction_pool = (
            interaction_tokens * modality_weights.unsqueeze(-1)
        ).sum(dim=1)
        synergy_state = self.synergy_expert(
            torch.cat((interaction_pool, pair_context), dim=-1)
        )
        synergy_present = present.sum(dim=1).ge(2)
        synergy_state = synergy_state.masked_fill(~synergy_present.unsqueeze(-1), 0.0)
        expert_states = torch.cat(
            (modality_expert_states, synergy_state.unsqueeze(1)), dim=1
        )

        present_float = present.to(pooled.dtype)
        present_count = present_float.sum(dim=1, keepdim=True).clamp_min(1.0)
        synergy_quality = (
            modality_quality * present_float.unsqueeze(-1)
        ).sum(dim=1) / present_count
        expert_quality = torch.cat(
            (modality_quality, synergy_quality.unsqueeze(1)), dim=1
        )
        expert_present = torch.cat((present, synergy_present.unsqueeze(1)), dim=1)
        expert_scores = torch.cat(
            [
                head(torch.cat((expert_states[:, index], expert_quality[:, index]), dim=-1))
                for index, head in enumerate(self.expert_score_heads)
            ],
            dim=1,
        )
        synergy_reliability = synergy_quality[:, 1]
        synergy_uncertainty = torch.expm1(synergy_quality[:, 2]).clamp_min(0.0)
        synergy_conflict = synergy_quality[:, 3]
        expert_reliability = torch.cat(
            (effective_reliability, synergy_reliability.unsqueeze(1)), dim=1
        )
        expert_uncertainty = torch.cat(
            (effective_uncertainty, synergy_uncertainty.unsqueeze(1)), dim=1
        )
        expert_conflict = torch.cat(
            (effective_conflict, synergy_conflict.unsqueeze(1)), dim=1
        )
        if self.use_reliability:
            expert_scores = expert_scores + torch.log(
                expert_reliability.clamp_min(1e-6)
            )
        if self.use_uncertainty:
            expert_scores = expert_scores - self.uncertainty_penalty * expert_uncertainty
        if self.use_conflict:
            expert_scores = expert_scores - self.conflict_penalty * expert_conflict
        expert_weights = _masked_softmax(expert_scores, expert_present, dim=1)
        expert_fused = (expert_states * expert_weights.unsqueeze(-1)).sum(dim=1)

        # The original reliable weighted sum is a residual path.  This gives a
        # stable fallback for T-only routing and lets the deep branch focus on
        # interactions that a scalar weighted sum cannot express.
        fused = self.fusion_output_norm(
            base_fused
            + self.fusion_refiner(torch.cat((base_fused, expert_fused), dim=-1))
        )

        final_features = torch.cat(
            (fused, fractions, effective_reliability, effective_uncertainty, effective_conflict),
            dim=-1,
        )
        expert_logits = self.expert_classifier(expert_states)
        deep_logits = self.classifier(final_features)
        deep_regression = 3.0 * torch.tanh(
            self.regressor(final_features).squeeze(-1)
        )
        if self.prediction_blending:
            # Learn how much to trust the deep joint prediction versus the two
            # supervised consensus paths.  These residual votes make the
            # classifier less brittle when a route contains only T or T+A.
            modality_consensus = (
                modality_logits * modality_weights.unsqueeze(-1)
            ).sum(dim=1)
            expert_consensus = (
                expert_logits * expert_weights.unsqueeze(-1)
            ).sum(dim=1)
            classification_blend = torch.softmax(
                self.classification_blender(final_features).float(), dim=-1
            ).to(deep_logits.dtype)
            prediction_components = torch.stack(
                (deep_logits, modality_consensus, expert_consensus), dim=1
            )
            logits = (
                prediction_components * classification_blend.unsqueeze(-1)
            ).sum(dim=1)
            regression_blend = torch.sigmoid(
                self.regression_blender(final_features).squeeze(-1)
            )
            modality_regression_consensus = (
                modality_regression * modality_weights
            ).sum(dim=1)
            regression = (
                regression_blend * deep_regression
                + (1.0 - regression_blend) * modality_regression_consensus
            )
        else:
            logits = deep_logits
            regression = deep_regression
            classification_blend = final_features.new_zeros(
                final_features.shape[0], 3
            )
            classification_blend[:, 0] = 1.0
            regression_blend = final_features.new_ones(final_features.shape[0])
        log_variance = self.log_variance_head(final_features).squeeze(-1).clamp(-6.0, 3.0)
        variance = torch.exp(log_variance)
        confidence = torch.softmax(logits.float(), dim=-1).max(dim=-1).values
        confidence = confidence * torch.exp(-0.5 * variance.clamp(max=6.0))

        return {
            "logits": logits,
            "regression": regression,
            "log_variance": log_variance,
            "variance": variance,
            "confidence": confidence,
            "fractions": fractions,
            "reliability_logits": reliability_logits,
            "reliability": reliability,
            "modality_uncertainty": modality_uncertainty,
            "modality_logits": modality_logits,
            "modality_regression": modality_regression,
            "conflict_matrix": conflict_matrix,
            "modality_conflict": modality_conflict,
            "modality_weights": modality_weights,
            "attention_quality": attention_quality,
            "interaction_tokens": interaction_tokens,
            "expert_logits": expert_logits,
            "expert_present": expert_present,
            "expert_weights": expert_weights,
            "classification_blend": classification_blend,
            "regression_blend": regression_blend,
        }

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        token_type_ids: Tensor,
        audio: Tensor,
        vision: Tensor,
    ) -> Dict[str, Tensor]:
        """Run all physically present modalities for ordinary training/evaluation."""
        availability, fractions, present, _, _ = self._availability_state(
            input_ids, attention_mask, token_type_ids, audio, vision
        )
        pooled_states = []
        for modality in ("T", "A", "V"):
            pooled, _, _ = self.encode_modality(
                modality,
                input_ids=input_ids,
                attention_mask=attention_mask,
                token_type_ids=token_type_ids,
                audio=audio,
                vision=vision,
            )
            pooled_states.append(pooled)
        output = self.predict_from_pooled(
            torch.stack(pooled_states, dim=1), fractions, present
        )
        output["availability"] = availability
        return output


def load_matching_backbone_weights(model: AdaTrustMSAModel, checkpoint: str) -> Dict[str, object]:
    """Warm-start all shape-compatible backbone parameters.

    Old prediction/fusion heads are intentionally ignored because AdaTrustMSA uses
    different semantics and dimensions.  The function returns an audit record.
    """
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    source = payload["state_dict"]
    target = model.state_dict()
    excluded_prefixes = (
        "classifier.",
        "regressor.",
        "fusion.",
        "interaction_blocks.",
        "pair_encoder.",
        "synergy_expert.",
        "expert_score_heads.",
        "expert_classifier.",
        "fusion_refiner.",
        "fusion_output_norm.",
        "classification_blender.",
        "regression_blender.",
        "temporal_poolers.",
        "modality_scorers.",
        "temporal_scorer.",
    )
    matched = {
        name: value
        for name, value in source.items()
        if name in target
        and target[name].shape == value.shape
        and not name.startswith(excluded_prefixes)
    }
    incompatible = model.load_state_dict(matched, strict=False)
    return {
        "source": str(checkpoint),
        "matched_tensor_count": len(matched),
        "matched_parameter_count": int(sum(value.numel() for value in matched.values())),
        "excluded_prefixes": list(excluded_prefixes),
        "missing_keys": list(incompatible.missing_keys),
        "unexpected_keys": list(incompatible.unexpected_keys),
    }
