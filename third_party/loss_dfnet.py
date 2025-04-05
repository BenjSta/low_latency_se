import warnings
from collections import defaultdict
from typing import Dict, Final, Iterable, List, Literal, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor, nn

from third_party.df.io import resample
#from third_party.df.model import ModelParams
#from third_party.df.modules import Mask, erb_fb
from third_party.df.utils import angle, as_complex


def wg(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS = as_complex(S).abs().square()
    NN = as_complex(N).abs().square()
    return (SS / (SS + NN + eps)).clamp(0, 1)


def irm(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    N = X - S
    SS_mag = as_complex(S).abs()
    NN_mag = as_complex(N).abs()
    return (SS_mag / (SS_mag + NN_mag + eps)).clamp(0, 1)


def iam(S: Tensor, X: Tensor, eps: float = 1e-10) -> Tensor:
    SS_mag = as_complex(S).abs()
    XX_mag = as_complex(X).abs()
    return (SS_mag / (XX_mag + eps)).clamp(0, 1)


class Stft(nn.Module):
    def __init__(self, n_fft: int, hop: Optional[int] = None, window: Optional[Tensor] = None):
        super().__init__()
        self.n_fft = n_fft
        self.hop = hop or n_fft // 4
        if window is not None:
            assert window.shape[0] == n_fft
        else:
            window = torch.hann_window(self.n_fft)
        self.w: torch.Tensor
        self.register_buffer("w", window)

    def forward(self, input: Tensor):
        # Time-domain input shape: [B, T]
        out = torch.stft(
            input,
            n_fft=self.n_fft,
            hop_length=self.hop,
            window=self.w,
            normalized=True,
            return_complex=True,
        )
        return out


class Istft(nn.Module):
    def __init__(self, n_fft_inv: int, hop_inv: int, window_inv: Tensor):
        super().__init__()
        # Synthesis back to time domain
        self.n_fft_inv = n_fft_inv
        self.hop_inv = hop_inv
        self.w_inv: torch.Tensor
        assert window_inv.shape[0] == n_fft_inv
        self.register_buffer("w_inv", window_inv)

    def forward(self, input: Tensor):
        # Input shape: [B, * T, F, (2)]
        input = as_complex(input)
        t, f = input.shape[-2:]
        sh = input.shape[:-2]
        # Even though this is not the DF implementation, it numerical sufficiently close.
        # Pad one extra step at the end to get original signal length
        out = torch.istft(
            F.pad(input.reshape(-1, t, f).transpose(1, 2), (0, 1)),
            n_fft=self.n_fft_inv,
            hop_length=self.hop_inv,
            window=self.w_inv,
            normalized=True,
        )
        if input.ndim > 2:
            out = out.view(*sh, out.shape[-1])
        return out


class MultiResSpecLoss(nn.Module):
    gamma: Final[float]
    f: Final[float]
    f_complex: Final[Optional[List[float]]]

    def __init__(
        self,
        n_ffts: Iterable[int],
        gamma: float = 1,
        factor: float = 1,
        f_complex: Optional[Union[float, Iterable[float]]] = None,
    ):
        super().__init__()
        self.gamma = gamma
        self.f = factor
        self.stfts = nn.ModuleDict({str(n_fft): Stft(n_fft) for n_fft in n_ffts})
        if f_complex is None or f_complex == 0:
            self.f_complex = None
        elif isinstance(f_complex, Iterable):
            self.f_complex = list(f_complex)
        else:
            self.f_complex = [f_complex] * len(self.stfts)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        loss = torch.zeros((), device=input.device, dtype=input.dtype)
        for i, stft in enumerate(self.stfts.values()):
            Y = stft(input)
            S = stft(target)
            Y_abs = Y.abs()
            S_abs = S.abs()
            if self.gamma != 1:
                Y_abs = Y_abs.clamp_min(1e-12).pow(self.gamma)
                S_abs = S_abs.clamp_min(1e-12).pow(self.gamma)
            loss += F.mse_loss(Y_abs, S_abs) * self.f
            if self.f_complex is not None:
                if self.gamma != 1:
                    Y = Y_abs * torch.exp(1j * angle.apply(Y))
                    S = S_abs * torch.exp(1j * angle.apply(S))
                loss += F.mse_loss(torch.view_as_real(Y), torch.view_as_real(S)) * self.f_complex[i]
        return loss


class SpectralLoss(nn.Module):
    gamma: Final[float]
    f_m: Final[float]
    f_c: Final[float]
    f_u: Final[float]

    def __init__(
        self,
        gamma: float = 1,
        factor_magnitude: float = 1,
        factor_complex: float = 1,
        factor_under: float = 1,
    ):
        super().__init__()
        self.gamma = gamma
        self.f_m = factor_magnitude
        self.f_c = factor_complex
        self.f_u = factor_under

    def forward(self, input, target):
        input = as_complex(input)
        target = as_complex(target)
        input_abs = input.abs()
        target_abs = target.abs()
        if self.gamma != 1:
            input_abs = input_abs.clamp_min(1e-12).pow(self.gamma)
            target_abs = target_abs.clamp_min(1e-12).pow(self.gamma)
        tmp = (input_abs - target_abs).pow(2)
        if self.f_u != 1:
            # Weighting if predicted abs is too low
            tmp *= torch.where(input_abs < target_abs, self.f_u, 1.0)
        loss = torch.mean(tmp) * self.f_m
        if self.f_c > 0:
            if self.gamma != 1:
                input = input_abs * torch.exp(1j * angle.apply(input))
                target = target_abs * torch.exp(1j * angle.apply(target))
            loss_c = (
                F.mse_loss(torch.view_as_real(input), target=torch.view_as_real(target)) * self.f_c
            )
            loss = loss + loss_c
        return loss


# class MaskLoss(nn.Module):
#     def __init__(
#         self,
#         df_state: DF,
#         mask: str = "iam",
#         gamma: float = 0.6,
#         powers: List[int] = [2],
#         factors: List[float] = [1],
#         f_under: float = 1,
#         eps=1e-12,
#         factor: float = 1.0,
#         gamma_pred: Optional[float] = None,
#         f_max_idx: Optional[int] = None,  # Maximum frequency bin index
#     ):
#         super().__init__()
#         if mask == "wg":
#             self.mask_fn = wg
#         elif mask == "irm":
#             self.mask_fn = irm
#         elif mask == "iam":
#             self.mask_fn = iam
#         elif mask == "spec":
#             self.mask_fn = None
#         else:
#             raise ValueError(f"Unsupported mask function: {mask}.")
#         self.gamma = gamma
#         self.gamma_pred = gamma if gamma_pred is None else gamma_pred
#         self.powers = powers
#         self.factors = factors
#         self.f_under = f_under
#         self.eps = eps
#         self.factor = factor
#         self.f_max_idx = f_max_idx
#         self.erb_fb: Tensor
#         self.erb_inv_fb: Tensor
#         self.register_buffer("erb_fb", erb_fb(df_state.erb_widths(), ModelParams().sr))
#         self.register_buffer(
#             "erb_inv_fb", erb_fb(df_state.erb_widths(), ModelParams().sr, inverse=True)
#         )

#     def __repr__(self):
#         s = f"MaskLoss {self.mask_fn} (gamma: {self.gamma}"
#         for p, f in zip(self.powers, self.factors):
#             s += f", p: {p}, f: {f}"
#         s += ")"
#         return s

#     @torch.jit.export
#     def erb_mask_compr(self, clean: Tensor, noisy: Tensor, compressed: bool = True) -> Tensor:
#         mask_fn = self.mask_fn or iam
#         mask = mask_fn(clean, noisy)
#         mask = self.erb(mask)
#         if compressed:
#             mask = mask.pow(self.gamma)
#         return mask

#     @torch.jit.export
#     def erb(self, x: Tensor, clamp_min: Optional[float] = None) -> Tensor:
#         x = torch.matmul(x, self.erb_fb)
#         if clamp_min is not None:
#             x = x.clamp_min(clamp_min)
#         return x

#     @torch.jit.export
#     def erb_inv(self, x: Tensor) -> Tensor:
#         return torch.matmul(x, self.erb_inv_fb)

#     def forward(
#         self, input: Tensor, clean: Tensor, noisy: Tensor, max_bin: Optional[Tensor] = None
#     ) -> Tensor:
#         # Input mask shape: [B, C, T, F]
#         b, _, _, f = input.shape
#         if not torch.isfinite(input).all():
#             raise ValueError("Input is NaN")
#         assert input.min() >= 0
#         if self.mask_fn is not None:
#             g_t = self.erb_mask_compr(clean, noisy, compressed=True)
#             g_p = input.clamp_min(self.eps).pow(self.gamma_pred)
#         else:
#             g_t = self.erb(clean.abs()).pow(self.gamma)  # We use directly the clean spectrum
#             g_p = (self.erb(noisy.abs()) * input).pow(self.gamma_pred)
#         loss = torch.zeros((), device=input.device)
#         if self.f_max_idx is not None:
#             g_t = g_t[..., : self.f_max_idx]
#             g_p = g_p[..., : self.f_max_idx]
#         tmp = g_t.sub(g_p).pow(2)
#         if self.f_under != 1:
#             # Weighting if gains are too low
#             tmp = tmp * torch.where(g_p < g_t, self.f_under, 1.0)
#         if max_bin is not None:
#             m = torch.ones((b, 1, 1, f), device=input.device)
#             for i, mb in enumerate(max_bin):
#                 m[i, ..., mb:] = 0
#             tmp = tmp * m
#         for power, factor in zip(self.powers, self.factors):
#             # Reduce the 2 from .pow(2) above
#             loss += tmp.clamp_min(1e-13).pow(power // 2).mean().mul(factor) * self.factor
#         return loss.mean()


# class MaskSpecLoss(nn.Module):
#     def __init__(
#         self, df_state: DF, factor=1.0, gamma: float = 0.6, f_max_idx: Optional[int] = None
#     ):
#         super().__init__()
#         self.f_max_idx = f_max_idx
#         self.apply_mask = Mask(erb_fb(df_state.erb_widths(), ModelParams().sr, inverse=True))
#         self.loss = SpectralLoss(factor_magnitude=factor, gamma=gamma)

#     def forward(self, input: Tensor, clean: Tensor, noisy: Tensor) -> Tensor:
#         enh = self.apply_mask(noisy, input)
#         if self.f_max_idx is not None:
#             enh = enh[..., : self.f_max_idx]
#             clean = clean[..., : self.f_max_idx]
#         return self.loss(enh, clean)



class SiSdr(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, input: Tensor, target: Tensor):
        # Input shape: [B, T]
        eps = torch.finfo(input.dtype).eps
        t = input.shape[-1]
        target = target.reshape(-1, t)
        input = input.reshape(-1, t)
        # Einsum for batch vector dot product
        Rss: Tensor = torch.einsum("bi,bi->b", target, target).unsqueeze(-1)
        a: Tensor = torch.einsum("bi,bi->b", target, input).add(eps).unsqueeze(-1) / Rss.add(eps)
        e_true = a * target
        e_res = input - e_true
        Sss = e_true.square()
        Snn = e_res.square()
        # Only reduce over each sample. Supposed to be used when used as a metric.
        Sss = Sss.sum(-1)
        Snn = Snn.sum(-1)
        return 10 * torch.log10(Sss.add(eps) / Snn.add(eps))


class SdrLoss(nn.Module):
    def __init__(self, factor=0.2):
        super().__init__()
        self.factor = factor
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        # Input shape: [B, T]
        return -self.sdr(input, target).mean() * self.factor


class SegSdrLoss(nn.Module):
    def __init__(self, window_sizes: List[int], factor: float = 0.2, overlap: float = 0):
        # Window size in samples
        super().__init__()
        self.window_sizes = window_sizes
        self.factor = factor
        self.hop = 1 - overlap
        self.sdr = SiSdr()

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # Input shape: [B, T]
        if self.factor == 0:
            return torch.zeros((), device=input.device)
        loss = torch.zeros((), device=input.device)
        for ws in self.window_sizes:
            if ws > input.size(-1):
                warnings.warn(
                    f"Input size {input.size(-1)} smaller than window size. Adjusting window size."
                )
                ws = input.size(1)
            loss += self.sdr(
                input=input.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
                target=target.unfold(-1, ws, int(self.hop * ws)).reshape(-1, ws),
            ).mean()
        return -loss * self.factor



class ASRLoss(nn.Module):
    target_sr = 16000
    n_fft = 400
    hop = 160
    beam_size = 20
    lang = "en"
    task = "transcribe"
    max_ctx = 25

    def __init__(
        self,
        sr: int,
        factor: float = 1,
        factor_lm: float = 1,
        loss_lm: Literal["CTC", "CrossEntropy"] = "CrossEntropy",
        model: str = "base.en",
    ) -> None:
        super().__init__()
        import whisper

        self.sr = sr
        self.factor = factor
        self.factor_lm = factor_lm
        self.model = whisper.load_model(model)
        self.model.requires_grad_(False)
        self.options = whisper.DecodingOptions(
            task=self.task, language=self.lang, without_timestamps=True, sample_len=self.max_ctx
        )
        self.mel_filters: Tensor
        self.register_buffer(
            "mel_filters", torch.from_numpy(self.get_mel_filters(self.target_sr, 400, 80))
        )
        self.tokenizer = whisper.tokenizer.get_tokenizer(
            self.model.is_multilingual, language=self.lang, task=self.options.task
        )
        self.decoder = whisper.decoding.GreedyDecoder(0.0, self.tokenizer.eot)
        # self.decoder = whisper.decoding.BeamSearchDecoder(self.beam_size, self.tokenizer.eot, , 1.)
        self.sot_sequence = self.tokenizer.sot_sequence_including_notimestamps
        self.n_ctx: int = self.model.dims.n_text_ctx
        self.initial_tokens = self._get_initial_tokens()
        self.sot_index: int = self.initial_tokens.index(self.tokenizer.sot)
        self.sample_begin: int = len(self.initial_tokens)
        self.sample_len: int = self.options.sample_len or self.model.dims.n_text_ctx // 2
        self.blank = self.tokenizer.encode(" ")[0]
        self.eot = self.tokenizer.eot
        self.loss_lm = loss_lm

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        features_i = self.model.embed_audio(self.preprocess(input))
        features_t = self.model.embed_audio(self.preprocess(target))
        # Loss based on the audio encoding:
        loss = 0
        if self.factor > 0:
            loss = F.mse_loss(features_i[0], features_t[0]) * self.factor
        if self.factor_lm > 0:
            _, tokens_t = self.decode_tokens(features_t)  # [N, S]
            logits_i, tokens_i = self.decode_tokens(features_i)  # [N, T, C]
            log_probs_i = F.log_softmax(logits_i, dim=-1)

            # Loss based on the logits:
            if self.factor_lm > 0:
                if self.loss_lm == "CTC":
                    input_lengths = torch.as_tensor(
                        [torch.argwhere(t == self.eot)[0] for t in tokens_i],
                        device=input.device,
                        dtype=torch.long,
                    )
                    target_lengths = torch.as_tensor(
                        [torch.argwhere(t == self.eot)[0] for t in tokens_t],
                        device=input.device,
                        dtype=torch.long,
                    )
                    ctc_loss = F.ctc_loss(
                        log_probs=log_probs_i[:, : input_lengths.max()].transpose(0, 1),
                        targets=tokens_t[:, : target_lengths.max()].to(torch.long),
                        input_lengths=input_lengths,
                        target_lengths=target_lengths,
                        blank=self.blank,
                        zero_infinity=True,
                    )
                    loss += ctc_loss * self.factor_lm
                else:
                    delta = log_probs_i.shape[1] - tokens_t.shape[1]
                    if delta > 0:
                        tokens_t = torch.cat(
                            (
                                tokens_t,
                                torch.full(
                                    (tokens_t.shape[0], delta),
                                    self.eot,
                                    device=tokens_t.device,
                                    dtype=tokens_t.dtype,
                                ),
                            ),
                            dim=1,
                        )
                    # if tokens_t.shape[1] != log_probs_i.shape[1]:
                    #     ic(tokens_t.shape, log_probs_i.shape)
                    #     for i in range(tokens_t.shape[0]):
                    #         ic(tokens_t[i])
                    #         ic(log_probs_i[i].argmax(dim=-1))
                    ce_loss = F.nll_loss(
                        log_probs_i.flatten(0, 1),
                        tokens_t[:, : tokens_i.shape[1]].flatten(0, 1),
                    )
                    loss += ce_loss * self.factor_lm
        return loss

    def decode_text(self, tokens: Tensor) -> List[str]:
        tokens = [t[: torch.argwhere(t == self.eot)[0]] for t in tokens]
        return [self.tokenizer.decode(t).strip() for t in tokens]

    def decode_tokens(
        self,
        features: Tensor,
        start_tokens: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        n = features.shape[0]
        sum_logprobs: Tensor = torch.zeros(n, device=features.device)
        tokens: Tensor = start_tokens or torch.tensor(
            [self.initial_tokens], device=features.device
        ).repeat(n, 1)
        logits: List[Tensor] = []
        for i in range(self.sample_len):
            # we don't need no_speech_probs, only use last index (-1)
            logits.append(self.model.logits(tokens, features)[:, -1])
            tokens, completed = self.decoder.update(tokens, logits[-1], sum_logprobs)
            if completed or tokens.shape[-1] > self.n_ctx:
                break
        tokens, _ = self.decoder.finalize(tokens, sum_logprobs)
        return torch.stack(logits, dim=1), tokens[:, self.sample_begin : -1]

    def preprocess(self, audio: Tensor) -> Tensor:
        import whisper

        audio = resample(audio, self.sr, self.target_sr)
        audio = whisper.pad_or_trim(audio.squeeze(1))
        mel = self.log_mel_spectrogram(audio, self.mel_filters.to(audio.device))
        return mel

    def log_mel_spectrogram(self, audio: Tensor, mel_fb: Tensor):
        """From openai/whisper"""
        window = torch.hann_window(self.n_fft).to(audio.device)
        stft = torch.stft(audio, self.n_fft, self.hop, window=window, return_complex=True)
        assert stft.isfinite().all()
        magnitudes = stft[..., :-1].abs() ** 2
        assert magnitudes.isfinite().all()
        assert mel_fb.isfinite().all()

        mel_spec = mel_fb @ magnitudes

        log_spec = torch.clamp(mel_spec, min=1e-10).log10()
        log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
        log_spec = (log_spec + 4.0) / 4.0
        return log_spec

    def get_mel_filters(self, sr, n_fft, n_mels=128, dtype=None):
        """From transformers/models/whisper/feature_extraction"""
        import numpy as np

        dtype = dtype or np.float32
        # Initialize the weights
        n_mels = int(n_mels)
        weights = np.zeros((n_mels, int(1 + n_fft // 2)), dtype=dtype)

        # Center freqs of each FFT bin
        fftfreqs = np.fft.rfftfreq(n=n_fft, d=1.0 / sr)

        # 'Center freqs' of mel bands - uniformly spaced between limits
        min_mel = 0.0
        max_mel = 45.245640471924965

        mels = np.linspace(min_mel, max_mel, n_mels + 2)

        mels = np.asanyarray(mels)

        # Fill in the linear scale
        f_min = 0.0
        f_sp = 200.0 / 3
        freqs = f_min + f_sp * mels

        # And now the nonlinear scale
        min_log_hz = 1000.0  # beginning of log region (Hz)
        min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
        logstep = np.log(6.4) / 27.0  # step size for log region

        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * np.exp(logstep * (mels[log_t] - min_log_mel))

        mel_f = freqs

        fdiff = np.diff(mel_f)
        ramps = np.subtract.outer(mel_f, fftfreqs)

        for i in range(n_mels):
            # lower and upper slopes for all bins
            lower = -ramps[i] / fdiff[i]
            upper = ramps[i + 2] / fdiff[i + 1]

            # .. then intersect them with each other and zero
            weights[i] = np.maximum(0, np.minimum(lower, upper))

        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (mel_f[2 : n_mels + 2] - mel_f[:n_mels])
        weights *= enorm[:, np.newaxis]

        return weights

    def _get_initial_tokens(self) -> Tuple[int]:
        tokens = list(self.sot_sequence)
        prefix = self.options.prefix
        prompt = self.options.prompt

        if prefix:
            prefix_tokens = (
                self.tokenizer.encode(" " + prefix.strip()) if isinstance(prefix, str) else prefix
            )
            if self.sample_len is not None:
                max_prefix_len = self.n_ctx // 2 - self.sample_len
                prefix_tokens = prefix_tokens[-max_prefix_len:]
            tokens = tokens + prefix_tokens

        if prompt:
            prompt_tokens = (
                self.tokenizer.encode(" " + prompt.strip()) if isinstance(prompt, str) else prompt
            )
            tokens = [self.tokenizer.sot_prev] + prompt_tokens[-(self.n_ctx // 2 - 1) :] + tokens

        return tuple(tokens)

