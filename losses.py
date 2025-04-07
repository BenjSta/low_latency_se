import torch
import numpy as np
from third_party.loss_dfnet import MultiResSpecLoss, SpectralLoss, Stft
from torch import nn

class ComplexCompressedSpecMSE(nn.Module):
    def __init__(self, winlen: int, fs: int, f_fade_low: int, f_fade_high: int, lf_complex_loss_weight: float):
        super(ComplexCompressedSpecMSE, self).__init__()
        self.winlen = winlen
        self.fs = fs
        self.f_fade_low = f_fade_low
        self.f_fade_high = f_fade_high
        self.lf_complex_loss_weight = lf_complex_loss_weight
        self.stft = Stft(winlen, winlen // 2, torch.hann_window(winlen))
        self.spectral_loss = SpectralLoss(0.3, 1, 1, 1)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        y_stft = self.stft(
            y_true
        )
        y_pred_stft = self.stft(
            y_pred)

        return self.spectral_loss(y_stft, y_pred_stft)

class combined_loss(nn.Module):
    def __init__(self, winlen, fs, f_fade_low, f_fade_high, lf_complex_loss_weight, n_ffts=[80, 160, 320, 640], gamma=0.3, mrspec_lambda=1e3, ccmse_lambda=5e2):
        super(combined_loss, self).__init__()
        self.mrspec_lambda = mrspec_lambda
        self.ccmse_lambda = ccmse_lambda
        self.mrspec_loss = MultiResSpecLoss(n_ffts, gamma=gamma, factor=mrspec_lambda, f_complex=mrspec_lambda)
        self.ccmse_loss = ComplexCompressedSpecMSE(winlen, fs, f_fade_low, f_fade_high, lf_complex_loss_weight)

    def forward(self, y_true: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        loss1 = self.ccmse_loss(y_true, y_pred)
        loss2 = self.mrspec_loss(y_true, y_pred)
        return self.ccmse_lambda * loss1 + loss2
