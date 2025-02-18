import torch
import numpy as np

def complex_compressed_spec_mse(
        y_true, y_pred, winlen, fs, f_fade_low, f_fade_high, lf_complex_loss_weight
):
    window = torch.hann_window(winlen).to(y_true.device)
    fvec = torch.from_numpy(np.linspace(0, fs / 2, winlen // 2 + 1, endpoint=True)).float().to(y_true.device)

    # create low-pass frequency-dependent piecewise linear fade
    fade = torch.ones_like(fvec)
    fade[fvec < f_fade_low] = 1
    fade[fvec > f_fade_high] = 0
    fade[(fvec >= f_fade_low) & (fvec <= f_fade_high)] = 1 - ((
        f_fade_high - fvec[(fvec >= f_fade_low) & (fvec <= f_fade_high)]
    ) / (f_fade_high - f_fade_low))
    mag_weight = fade * (1- lf_complex_loss_weight)
    complex_weight = 1 - mag_weight


    y_stft = torch.stft(
        y_true, winlen, winlen // 2, winlen, return_complex=True, window=window
    )
    y_pred_stft = torch.stft(
        y_pred, winlen, winlen // 2, winlen, return_complex=True, window=window
    ) # [batch, winlen // 2 + 1, n_frames]

    y_mag = torch.clip(torch.abs(y_stft), 1e-12)
    y_pred_mag = torch.clip(torch.abs(y_pred_stft), 1e-12)

    y_phasor = y_stft / y_mag
    y_pred_phasor = y_pred_stft / y_pred_mag

    y_compressed_mag = y_mag**0.3
    y_pred_compressed_mag = y_pred_mag**0.3

    y_compressed_complex = y_compressed_mag * y_phasor
    y_pred_compressed_complex = y_pred_compressed_mag * y_pred_phasor

    mag_diff = torch.mean((y_compressed_mag - y_pred_compressed_mag)**2, dim=(0, 2))
    complex_diff = torch.mean(torch.abs(y_pred_compressed_complex - y_compressed_complex)**2, dim=(0, 2))

    return torch.mean(mag_weight * mag_diff + complex_weight * complex_diff)
    