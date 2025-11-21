import logging
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torchaudio
from coqpit import Coqpit
from torch.utils.data import DataLoader
from trainer.io import load_fsspec
from trainer.torch import DistributedSampler
from trainer.trainer_utils import get_optimizer, get_scheduler

from TTS.tts.configs.xtts_config import XttsConfig
from TTS.tts.datasets.dataset import TTSDataset
from TTS.tts.layers.tortoise.arch_utils import TorchMelSpectrogram
from TTS.tts.layers.xtts.dvae import DiscreteVAE
from TTS.tts.layers.xtts.tokenizer import VoiceBpeTokenizer
from TTS.tts.layers.xtts.trainer.dataset import XTTSDataset
from TTS.tts.layers.xtts.slm_loss import SLMPerceptualLoss

from TTS.tts.models.base_tts import BaseTTS
from TTS.tts.models.xtts import Xtts, XttsArgs
from TTS.utils.generic_utils import is_pytorch_at_least_2_4

logger = logging.getLogger(__name__)


@dataclass
class GPTArgs(XttsArgs):
    min_conditioning_length: int = 66150
    max_conditioning_length: int = 132300
    gpt_loss_text_ce_weight: float = 0.01
    gpt_loss_mel_ce_weight: float = 1.0
    gpt_num_audio_tokens: int = 8194
    debug_loading_failures: bool = False
    max_wav_length: int = 255995  # ~11.6 seconds
    max_text_length: int = 200
    tokenizer_file: str = ""
    mel_norm_file: str = "https://github.com/coqui-ai/TTS/releases/download/v0.14.0_models/mel_norms.pth"
    dvae_checkpoint: str = ""
    xtts_checkpoint: str = ""
    gpt_checkpoint: str = ""  # if defined it will replace the gpt weights on xtts model
    vocoder: str = ""  # override vocoder key on the config to avoid json write issues
    use_self_reference: bool = False  # if True, use each clip as its own reference for better style retention

    # Perceptual loss (SLM) hyperparameters
    slm_loss_recon_weight: float = 1.0
    slm_loss_contrast_weight: float = 0.1
    slm_temperature: float = 0.07
    slm_output_layer: int = 12
    slm_backprop_to_gpt: bool = True  # if True, let SLM loss backpropagate through GPT latents


@dataclass
class GPTTrainerConfig(XttsConfig):
    lr: float = 5e-06
    training_seed: int = 1
    optimizer_wd_only_on_weights: bool = False
    weighted_loss_attrs: dict = field(default_factory=dict)
    weighted_loss_multipliers: dict = field(default_factory=dict)
    test_sentences: list[dict] = field(default_factory=list)
    model_args: GPTArgs = field(default_factory=GPTArgs)


def callback_clearml_load_save(operation_type, model_info):
    # return None means skip the file upload/log, returning model_info will continue with the log/upload
    # you can also change the upload destination file name model_info.upload_filename or check the local file size with Path(model_info.local_model_path).stat().st_size
    assert operation_type in ("load", "save")
    logger.debug("%s %s", operation_type, model_info.__dict__)

    if "similarities.pth" in model_info.__dict__["local_model_path"]:
        return None

    return model_info


class GPTTrainer(BaseTTS):
    def __init__(self, config: Coqpit):
        """
        Tortoise GPT training class
        """
        super().__init__(config, ap=None, tokenizer=None)
        self.config = config
        # init XTTS model
        self.xtts = Xtts(self.config)
        # create the tokenizer with the target vocabulary
        self.xtts.tokenizer = VoiceBpeTokenizer(self.args.tokenizer_file)
        # init gpt encoder and hifigan decoder
        self.xtts.init_models()

        if self.args.xtts_checkpoint:
            self.load_checkpoint(self.config, self.args.xtts_checkpoint, eval=False, strict=False)

        # set mel stats
        if self.args.mel_norm_file:
            self.xtts.mel_stats = load_fsspec(self.args.mel_norm_file)

        # load GPT if available
        if self.args.gpt_checkpoint:
            gpt_checkpoint = torch.load(
                self.args.gpt_checkpoint, map_location=torch.device("cpu"), weights_only=is_pytorch_at_least_2_4()
            )
            # deal with coqui Trainer exported model
            if "model" in gpt_checkpoint.keys() and "config" in gpt_checkpoint.keys():
                logger.info("Coqui Trainer checkpoint detected! Converting it!")
                gpt_checkpoint = gpt_checkpoint["model"]
                states_keys = list(gpt_checkpoint.keys())
                for key in states_keys:
                    if "gpt." in key:
                        new_key = key.replace("gpt.", "")
                        gpt_checkpoint[new_key] = gpt_checkpoint[key]
                        del gpt_checkpoint[key]
                    else:
                        del gpt_checkpoint[key]

            # edit checkpoint if the number of tokens is changed to ensures the better transfer learning possible
            if (
                "text_embedding.weight" in gpt_checkpoint
                and gpt_checkpoint["text_embedding.weight"].shape != self.xtts.gpt.text_embedding.weight.shape
            ):
                num_new_tokens = (
                    self.xtts.gpt.text_embedding.weight.shape[0] - gpt_checkpoint["text_embedding.weight"].shape[0]
                )
                logger.info("Loading checkpoint with %d additional tokens.", num_new_tokens)

                # add new tokens to a linear layer (text_head)
                emb_g = gpt_checkpoint["text_embedding.weight"]
                new_row = torch.randn(num_new_tokens, emb_g.shape[1])
                start_token_row = emb_g[-1, :]
                emb_g = torch.cat([emb_g, new_row], axis=0)
                emb_g[-1, :] = start_token_row
                gpt_checkpoint["text_embedding.weight"] = emb_g

                # add new weights to the linear layer (text_head)
                text_head_weight = gpt_checkpoint["text_head.weight"]
                start_token_row = text_head_weight[-1, :]
                new_entry = torch.randn(num_new_tokens, self.xtts.gpt.text_head.weight.shape[1])
                text_head_weight = torch.cat([text_head_weight, new_entry], axis=0)
                text_head_weight[-1, :] = start_token_row
                gpt_checkpoint["text_head.weight"] = text_head_weight

                # add new biases to the linear layer (text_head)
                text_head_bias = gpt_checkpoint["text_head.bias"]
                start_token_row = text_head_bias[-1]
                new_bias_entry = torch.zeros(num_new_tokens)
                text_head_bias = torch.cat([text_head_bias, new_bias_entry], axis=0)
                text_head_bias[-1] = start_token_row
                gpt_checkpoint["text_head.bias"] = text_head_bias

            self.xtts.gpt.load_state_dict(gpt_checkpoint, strict=True)
            logger.info("GPT weights restored from: %s", self.args.gpt_checkpoint)

        # Mel spectrogram extractor for conditioning
        if self.args.gpt_use_perceiver_resampler:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=2048,
                hop_length=256,
                win_length=1024,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )
        else:
            self.torch_mel_spectrogram_style_encoder = TorchMelSpectrogram(
                filter_length=4096,
                hop_length=1024,
                win_length=4096,
                normalize=False,
                sampling_rate=config.audio.sample_rate,
                mel_fmin=0,
                mel_fmax=8000,
                n_mel_channels=80,
                mel_norm_file=self.args.mel_norm_file,
            )

        # Load DVAE
        self.dvae = DiscreteVAE(
            channels=80,
            normalization=None,
            positional_dims=1,
            num_tokens=self.args.gpt_num_audio_tokens - 2,
            codebook_dim=512,
            hidden_dim=512,
            num_resnet_blocks=3,
            kernel_size=3,
            num_layers=2,
            use_transposed_convs=False,
        )

        self.dvae.eval()
        if self.args.dvae_checkpoint:
            dvae_checkpoint = torch.load(
                self.args.dvae_checkpoint, map_location=torch.device("cpu"), weights_only=is_pytorch_at_least_2_4()
            )
            self.dvae.load_state_dict(dvae_checkpoint, strict=False)
            logger.info("DVAE weights restored from: %s", self.args.dvae_checkpoint)
        else:
            raise RuntimeError(
                "You need to specify config.model_args.dvae_checkpoint path to be able to train the GPT decoder!!"
            )

        # Perceptual loss on waveforms using a self-supervised speech model (WavLM)
        self.slm_perceptual_loss = SLMPerceptualLoss(
            gt_sample_rate=config.audio.sample_rate,
            gen_sample_rate=self.xtts.hifigan_decoder.output_sample_rate,
            output_layer=self.args.slm_output_layer,
            temperature=self.args.slm_temperature,
            device="cpu",
        )

        # Mel spectrogram extractor for DVAE
        self.torch_mel_spectrogram_dvae = TorchMelSpectrogram(
            mel_norm_file=self.args.mel_norm_file, sampling_rate=config.audio.dvae_sample_rate
        )

        # Autograd anomaly detection disabled for production training (saves memory and improves speed)
        # Uncomment the following lines only for debugging gradient issues:
        # torch.autograd.set_detect_anomaly(True)
        # logger.info("Enabled torch.autograd anomaly detection for GPTTrainer.")

    def forward(self, text_inputs, text_lengths, audio_codes, wav_lengths, cond_mels, cond_idxs, cond_lens):
        """
        Forward pass that uses both text and voice in either text conditioning mode or voice conditioning mode
        (actuated by `text_first`).

        text_inputs: long tensor, (b,t)
        text_lengths: long tensor, (b,)
        mel_inputs:  long tensor, (b,m)
        wav_lengths: long tensor, (b,)
        cond_mels: MEL float tensor, (b, num_samples, 80,t_m)
        cond_idxs: cond start and end indexs, (b, 2)
        cond_lens: long tensor, (b,)
        """
        losses = self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
        )
        return losses

    @torch.inference_mode()
    def test_run(self, assets) -> tuple[dict, dict]:  # pylint: disable=W0613
        test_audios = {}
        if self.config.test_sentences:
            # init gpt for inference mode
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.xtts.gpt.eval()
            logger.info("Synthesizing test sentences.")
            for idx, s_info in enumerate(self.config.test_sentences):
                wav = self.xtts.synthesize(
                    s_info["text"],
                    speaker_wav=s_info["speaker_wav"],
                    language=s_info["language"],
                    gpt_cond_len=3,
                )["wav"]
                test_audios[f"{idx}-audio"] = wav

            # delete inference layers
            del self.xtts.gpt.gpt_inference
            del self.xtts.gpt.gpt.wte
        return {"audios": test_audios}

    def test_log(
        self,
        outputs: dict,
        logger: "Logger",
        assets: dict,
        steps: int,  # pylint: disable=unused-argument
    ) -> None:
        logger.test_audios(steps, outputs["audios"], self.args.output_sample_rate)

    def format_batch(self, batch: dict) -> dict:
        return batch

    @torch.no_grad()  # torch no grad to avoid gradients from the pre-processing and DVAE codes extraction
    def format_batch_on_device(self, batch):
        """Compute spectrograms and auxiliary features on the device.

        This prepares:
        - text_inputs / lengths
        - conditioning mels for GPT
        - speaker embeddings for HiFiGAN
        - DVAE audio codes
        and keeps the raw waveform for perceptual loss.
        """

        batch["text_lengths"] = batch["text_lengths"]
        batch["wav_lengths"] = batch["wav_lengths"]
        batch["text_inputs"] = batch["padded_text"]
        batch["cond_idxs"] = batch["cond_idxs"]

        # Compute conditioning mel specs
        # Transform waves from torch.Size([B, num_cond_samples, 1, T]) to
        # torch.Size([B * num_cond_samples, 1, T]) because it is faster than iterating.
        B, num_cond_samples, C, T = batch["conditioning"].size()
        conditioning_reshaped = batch["conditioning"].view(B * num_cond_samples, C, T)
        paired_conditioning_mel = self.torch_mel_spectrogram_style_encoder(conditioning_reshaped)
        # Transform torch.Size([B * num_cond_samples, n_mel, T_mel]) into
        # torch.Size([B, num_cond_samples, n_mel, T_mel])
        n_mel = self.torch_mel_spectrogram_style_encoder.n_mel_channels
        T_mel = paired_conditioning_mel.size(2)
        paired_conditioning_mel = paired_conditioning_mel.view(B, num_cond_samples, n_mel, T_mel)
        batch["cond_mels"] = paired_conditioning_mel

        # Compute speaker embeddings from conditioning waveforms
        conditioning_audio = conditioning_reshaped.squeeze(1)  # (B * num_cond_samples, T)
        speaker_embedding = self.xtts.get_speaker_embedding(
            conditioning_audio,
            sr=self.config.audio.sample_rate,
        )  # (B * num_cond_samples, 512, 1)
        speaker_embedding = speaker_embedding.view(B, num_cond_samples, speaker_embedding.size(1), speaker_embedding.size(2))
        # Average over potentially multiple conditioning samples per item
        speaker_embedding = speaker_embedding.mean(dim=1)
        batch["speaker_embedding"] = speaker_embedding

        # Compute codes using DVAE
        if self.config.audio.sample_rate != self.config.audio.dvae_sample_rate:
            dvae_wav = torchaudio.functional.resample(
                batch["wav"],
                orig_freq=self.config.audio.sample_rate,
                new_freq=self.config.audio.dvae_sample_rate,
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="kaiser_window",
                beta=14.769656459379492,
            )
        else:
            dvae_wav = batch["wav"]
        dvae_mel_spec = self.torch_mel_spectrogram_dvae(dvae_wav)
        codes = self.dvae.get_codebook_indices(dvae_mel_spec)

        batch["audio_codes"] = codes
        # Store conditioning audio for speaker embedding computation in train_step
        # (we need it with gradients for perceptual loss backprop)
        batch["conditioning_audio"] = conditioning_reshaped
        # delete useless batch tensors (keep `wav` for perceptual loss)
        del batch["padded_text"]
        del batch["conditioning"]
        return batch

    def train_step(self, batch, criterion):
        loss_dict = {}
        cond_mels = batch["cond_mels"]
        text_inputs = batch["text_inputs"]
        text_lengths = batch["text_lengths"]
        audio_codes = batch["audio_codes"]
        wav_lengths = batch["wav_lengths"]
        cond_idxs = batch["cond_idxs"]
        cond_lens = batch["cond_lens"]
        wav_gt = batch["wav"]

        # Compute speaker embeddings with gradients for training, without for eval.
        # During training: recompute WITH gradients for perceptual loss backprop.
        # We can't use batch["speaker_embedding"] because it was computed under
        # @torch.no_grad() in format_batch_on_device, which blocks gradient flow
        # through the HiFiGAN decoder's speaker conditioning path.
        # During eval: use pre-computed embeddings (no gradients needed).
        # NOTE: We check if GPT is in training mode because the whole model is set to eval()
        # in on_train_epoch_start, and only GPT is set back to train().
        if self.xtts.gpt.training:
            conditioning_audio = batch["conditioning_audio"]  # (B * num_cond_samples, 1, T)
            B = text_inputs.size(0)
            B_times_num_cond = conditioning_audio.size(0)
            num_cond_samples = B_times_num_cond // B

            # Call speaker encoder directly (bypass @torch.inference_mode() decorator)
            conditioning_audio_16k = torchaudio.functional.resample(
                conditioning_audio.squeeze(1),
                self.config.audio.sample_rate,
                16000
            )
            speaker_embedding = self.xtts.hifigan_decoder.speaker_encoder.forward(
                conditioning_audio_16k,
                l2_norm=True
            ).unsqueeze(-1)  # (B * num_cond_samples, 512, 1)

            # Reshape and average over conditioning samples
            speaker_embedding = speaker_embedding.view(B, num_cond_samples, speaker_embedding.size(1), speaker_embedding.size(2))
            speaker_embedding = speaker_embedding.mean(dim=1)  # (B, 512, 1)

            # CRITICAL: The speaker encoder is frozen (requires_grad=False), so its output
            # also has requires_grad=False. We need to detach it and make it a leaf variable
            # with requires_grad=True so gradients can flow through the HiFiGAN decoder's
            # speaker conditioning path (for perceptual loss), without trying to backprop
            # through the frozen speaker encoder.
            speaker_embedding = speaker_embedding.detach().requires_grad_(True)
        else:
            # During evaluation, use pre-computed embeddings (no gradients)
            speaker_embedding = batch["speaker_embedding"]

        # Standard GPT cross-entropy losses
        loss_text, loss_mel, _ = self.forward(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels,
            cond_idxs,
            cond_lens,
        )
        loss_dict["loss_text_ce"] = loss_text * self.args.gpt_loss_text_ce_weight
        loss_dict["loss_mel_ce"] = loss_mel * self.args.gpt_loss_mel_ce_weight

        # Get GPT latents for HiFiGAN (no CE loss here, just latents)
        gpt_latents = self.xtts.gpt(
            text_inputs,
            text_lengths,
            audio_codes,
            wav_lengths,
            cond_mels=cond_mels,
            cond_idxs=cond_idxs,
            cond_lens=cond_lens,
            return_latent=True,
        )

        # Generate waveforms with HiFiGAN.
        # `slm_backprop_to_gpt` controls whether SLM gradients flow back through GPT
        # latents or only update the decoder. When False we detach latents to keep the
        # SLM path vocoder-only (current stable behavior).
        if self.args.slm_backprop_to_gpt:
            wav_gen = self.xtts.hifigan_decoder(gpt_latents, g=speaker_embedding)
        else:
            wav_gen = self.xtts.hifigan_decoder(gpt_latents.detach(), g=speaker_embedding)

        # Dual perceptual loss on raw audio (SLM)
        loss_recon_slm, loss_contrast_slm = self.slm_perceptual_loss(wav_gen, wav_gt)
        loss_dict["loss_slm_recon"] = loss_recon_slm * self.args.slm_loss_recon_weight
        loss_dict["loss_slm_contrast"] = loss_contrast_slm * self.args.slm_loss_contrast_weight

        loss_dict["loss"] = (
            loss_dict["loss_text_ce"]
            + loss_dict["loss_mel_ce"]
            + loss_dict["loss_slm_recon"]
            + loss_dict["loss_slm_contrast"]
        )
        return {"model_outputs": None}, loss_dict

    def eval_step(self, batch, criterion):
        # ignore masking for more consistent evaluation
        batch["cond_idxs"] = None
        return super().eval_step(batch, criterion)

    def on_train_epoch_start(self, trainer):
        trainer.model.eval()  # the whole model to eval
        # put gpt model in training mode
        if hasattr(trainer.model, "module") and hasattr(trainer.model.module, "xtts"):
            trainer.model.module.xtts.gpt.train()
        else:
            trainer.model.xtts.gpt.train()

    def on_init_end(self, trainer):  # pylint: disable=W0613
        # ignore similarities.pth on clearml save/upload
        if self.config.dashboard_logger.lower() == "clearml":
            from clearml.binding.frameworks import WeightsFileHandler

            WeightsFileHandler.add_pre_callback(callback_clearml_load_save)

    @torch.inference_mode()
    def inference(
        self,
        x,
        aux_input=None,
    ):  # pylint: disable=dangerous-default-value
        return None

    @staticmethod
    def get_criterion():
        return None

    def get_sampler(self, dataset: TTSDataset, num_gpus=1):
        # sampler for DDP
        batch_sampler = DistributedSampler(dataset) if num_gpus > 1 else None
        return batch_sampler

    def get_data_loader(
        self,
        config: Coqpit,
        assets: dict,
        is_eval: bool,
        samples: list[dict] | list[list],
        verbose: bool,
        num_gpus: int,
        rank: int | None = None,
    ) -> "DataLoader":  # pylint: disable=W0613
        if is_eval and not config.run_eval:
            loader = None
        else:
            # init dataloader
            dataset = XTTSDataset(self.config, samples, self.xtts.tokenizer, config.audio.sample_rate, is_eval)

            # wait all the DDP process to be ready
            if num_gpus > 1:
                torch.distributed.barrier()

            # sort input sequences from short to long
            # dataset.preprocess_samples()

            # get samplers
            sampler = self.get_sampler(dataset, num_gpus)

            # ignore sampler when is eval because if we changed the sampler parameter we will not be able to compare previous runs
            if sampler is None or is_eval:
                loader = DataLoader(
                    dataset,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    shuffle=False,
                    drop_last=False,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
            else:
                loader = DataLoader(
                    dataset,
                    sampler=sampler,
                    batch_size=config.eval_batch_size if is_eval else config.batch_size,
                    collate_fn=dataset.collate_fn,
                    num_workers=config.num_eval_loader_workers if is_eval else config.num_loader_workers,
                    pin_memory=False,
                )
        return loader

    def get_optimizer(self) -> list:
        """Initiate and return the optimizer based on the config parameters."""
        # ToDo: deal with multi GPU training
        if self.config.optimizer_wd_only_on_weights:
            # parameters to only GPT model
            net = self.xtts.gpt

            # normalizations
            norm_modules = (
                nn.BatchNorm2d,
                nn.InstanceNorm2d,
                nn.BatchNorm1d,
                nn.InstanceNorm1d,
                nn.BatchNorm3d,
                nn.InstanceNorm3d,
                nn.GroupNorm,
                nn.LayerNorm,
            )
            # nn.Embedding
            emb_modules = (nn.Embedding, nn.EmbeddingBag)

            param_names_notweights = set()
            all_param_names = set()
            param_map = {}
            for mn, m in net.named_modules():
                for k, v in m.named_parameters():
                    v.is_bias = k.endswith(".bias")
                    v.is_weight = k.endswith(".weight")
                    v.is_norm = isinstance(m, norm_modules)
                    v.is_emb = isinstance(m, emb_modules)

                    fpn = f"{mn}.{k}" if mn else k  # full param name
                    all_param_names.add(fpn)
                    param_map[fpn] = v
                    if v.is_bias or v.is_norm or v.is_emb:
                        param_names_notweights.add(fpn)

            params_names_notweights = sorted(list(param_names_notweights))
            params_notweights = [param_map[k] for k in params_names_notweights]
            params_names_weights = sorted(list(all_param_names ^ param_names_notweights))
            params_weights = [param_map[k] for k in params_names_weights]

            groups = [
                {"params": params_weights, "weight_decay": self.config.optimizer_params["weight_decay"]},
                {"params": params_notweights, "weight_decay": 0},
            ]
            # torch.optim.AdamW
            opt = get_optimizer(
                self.config.optimizer,
                self.config.optimizer_params,
                self.config.lr,
                parameters=groups,
            )
            opt._group_names = [params_names_weights, params_names_notweights]
            return opt

        return get_optimizer(
            self.config.optimizer,
            self.config.optimizer_params,
            self.config.lr,
            # optimize only for the GPT model
            parameters=self.xtts.gpt.parameters(),
        )

    def get_scheduler(self, optimizer) -> list:
        """Set the scheduler for the optimizer.

        Args:
            optimizer: `torch.optim.Optimizer`.
        """
        return get_scheduler(self.config.lr_scheduler, self.config.lr_scheduler_params, optimizer)

    def load_checkpoint(
        self,
        config,
        checkpoint_path,
        *,
        eval=False,
        strict=True,
        cache_storage="/tmp/tts_cache",
        target_protocol="s3",
        target_options={"anon": True},
    ):  # pylint: disable=unused-argument, disable=W0201, disable=W0102, redefined-builtin
        """Load the model checkpoint and setup for training or inference"""

        state = self.xtts.get_compatible_checkpoint_state_dict(checkpoint_path)

        # load the model weights
        self.xtts.load_state_dict(state, strict=strict)

        if eval:
            self.xtts.gpt.init_gpt_for_inference(kv_cache=self.args.kv_cache, use_deepspeed=False)
            self.eval()

    @staticmethod
    def init_from_config(config: "GPTTrainerConfig", samples: list[list] | list[dict] = None):
        """Initiate model from config

        Args:
            config (GPTTrainerConfig): Model config.
            samples (Union[List[List], List[Dict]]): Training samples to parse speaker ids for training.
                Defaults to None.
        """
        return GPTTrainer(config)
