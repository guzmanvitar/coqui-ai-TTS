import torch
import torch.nn as nn
import torch.nn.functional as F
import torchaudio

from TTS.vc.layers.freevc.wavlm import get_wavlm


class SLMPerceptualLoss(nn.Module):
    """Dual perceptual loss on raw audio using a self-supervised speech model (WavLM).

    This module computes:
    - P1: reconstruction loss (L1) between SLM embeddings of generated vs ground-truth audio.
    - P2: contrastive (InfoNCE-style) loss to encourage sample-wise matching of prosody.
    """

    def __init__(
        self,
        gt_sample_rate: int,
        gen_sample_rate: int,
        *,
        output_layer: int = 12,
        temperature: float = 0.07,
        target_sample_rate: int = 16000,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.gt_sample_rate = gt_sample_rate
        self.gen_sample_rate = gen_sample_rate
        self.target_sample_rate = target_sample_rate
        self.output_layer = output_layer
        self.temperature = temperature

        # SLM backbone (frozen)
        self.wavlm = get_wavlm(device=device)
        for param in self.wavlm.parameters():
            param.requires_grad = False

    def _resample(self, wav: torch.Tensor, orig_sr: int) -> torch.Tensor:
        """Resample `wav` from `orig_sr` to `self.target_sample_rate`.

        Accepts tensors shaped (B, T) or (B, 1, T).
        """

        if orig_sr == self.target_sample_rate:
            return wav

        if wav.dim() == 2:
            # (B, T) -> (B, 1, T) for torchaudio
            wav = wav.unsqueeze(1)

        wav = torchaudio.functional.resample(
            wav,
            orig_freq=orig_sr,
            new_freq=self.target_sample_rate,
            lowpass_filter_width=64,
            rolloff=0.9475937167399596,
            resampling_method="kaiser_window",
            beta=14.769656459379492,
        )
        # Back to (B, T)
        return wav.squeeze(1)

    def _extract_features(self, wav: torch.Tensor) -> torch.Tensor:
        """Extract pooled SLM features from waveform.

        Args:
            wav: Tensor of shape (B, T) at `self.target_sample_rate`.

        Returns:
            Tensor of shape (B, D) where D is the WavLM hidden size.
        """

        if wav.dim() != 2:
            raise ValueError("Expected waveform of shape (B, T) after resampling.")

        # WavLM expects (B, T). Parameters are frozen, but we allow gradients
        # to flow w.r.t. the input waveform so the generator can be trained.
        features, _ = self.wavlm.extract_features(
            source=wav,
            padding_mask=None,
            mask=False,
            output_layer=self.output_layer,
        )

        # features: (B, T_frames, D) -> mean-pool over time -> (B, D)
        return features.mean(dim=1)

    def forward(self, wav_gen: torch.Tensor, wav_gt: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Compute reconstruction and contrastive losses.

        Args:
            wav_gen: Generated audio, shape (B, T) or (B, 1, T) at `gen_sample_rate`.
            wav_gt: Ground-truth audio, shape (B, T) or (B, 1, T) at `gt_sample_rate`.

        Returns:
            loss_recon, loss_contrast
        """

        # Ensure batch dimension alignment and strip channel dim if present
        if wav_gen.dim() == 3:
            wav_gen = wav_gen.squeeze(1)
        if wav_gt.dim() == 3:
            wav_gt = wav_gt.squeeze(1)

        if wav_gen.shape[0] != wav_gt.shape[0]:
            raise ValueError("Generated and ground-truth batches must have the same size.")

        # Trim to common length to avoid padding artifacts and
        # materialize a fresh contiguous copy to avoid view/in-place issues.
        min_len = min(wav_gen.shape[-1], wav_gt.shape[-1])
        wav_gen = wav_gen[..., :min_len].contiguous()
        wav_gt = wav_gt[..., :min_len].contiguous()

        # Resample both streams to target SLM rate
        wav_gen_16k = self._resample(wav_gen, self.gen_sample_rate)
        wav_gt_16k = self._resample(wav_gt, self.gt_sample_rate)

        # Extract SLM embeddings.
        # We only need gradients flowing through the generated audio branch. The
        # ground-truth branch serves as a fixed target, so we compute it under
        # torch.no_grad() to save memory.
        gen_feats = self._extract_features(wav_gen_16k)
        with torch.no_grad():
            gt_feats = self._extract_features(wav_gt_16k)

        # P1: reconstruction loss (L1 between embeddings)
        loss_recon = F.l1_loss(gen_feats, gt_feats)

        # P2: contrastive loss (InfoNCE) over batch
        gen_norm = F.normalize(gen_feats, dim=-1)
        gt_norm = F.normalize(gt_feats, dim=-1)

        # Similarity matrix: (B, B), positives on the diagonal
        logits = gen_norm @ gt_norm.t() / self.temperature
        targets = torch.arange(logits.size(0), device=logits.device)

        # Contrastive loss requires batch size > 1
        # If batch size is 1, return zero loss (no negative pairs to contrast against)
        if logits.size(0) == 1:
            loss_contrast = torch.tensor(0.0, device=logits.device, dtype=logits.dtype)
        else:
            loss_contrast = F.cross_entropy(logits, targets)

        return loss_recon, loss_contrast

