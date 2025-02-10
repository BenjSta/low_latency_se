import torch
from torch import nn
from third_party.grouped_gru import GroupedGRU
import numpy as np
from ptflops import get_model_complexity_info

INIT_BIAS_ZERO = False


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride, bias=True):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            padding=(kernel_size[0] - 1, 0),
            bias=bias,
        )
        self.kernel_size = kernel_size

        if INIT_BIAS_ZERO:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        x = self.conv(x)
        x = x[:, :, : x.shape[2] - self.kernel_size[0] + 1, :]
        return x


class CausalTransConvBlock(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        fstride,
        output_padding_f_axis=0,
        bias=True,
    ):
        super().__init__()
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            output_padding=(0, output_padding_f_axis),
            bias=bias,
        )
        self.kernel_size = kernel_size
        if INIT_BIAS_ZERO:
            nn.init.zeros_(self.conv.bias)

    def forward(self, x):
        """
        2D Causal convolution.
        Args:
            x: [B, C, T, F]
        Returns:
            [B, C, T, F]
        """
        x = self.conv(x)
        x = x[:, :, : x.shape[2] - self.kernel_size[0] + 1, :]
        return x


class CRN(nn.Module):
    def __init__(
        self,
        fsize_input,
        num_channels_encoder,
        num_output_channels,
        kernel_sizes,
        fstride,
        n_gru_layers,
        n_gru_groups,
        nonlinearity,
        output_nonlinearity,
        batch_norm,
    ):

        super().__init__()

        # get from nn
        nonlinearity = getattr(nn, nonlinearity)
        output_nonlinearity = getattr(nn, output_nonlinearity)
        num_channels_decoder = num_channels_encoder[:0:-1] + [num_output_channels]

        # Encoder
        fsize = fsize_input
        fsizes = [fsize]
        self.encoder_blocks = nn.ModuleList()
        self.skipcon_convs = nn.ModuleList()
        for i in range(len(num_channels_encoder) - 1):
            conv_layer = CausalConvBlock(
                num_channels_encoder[i],
                num_channels_encoder[i + 1],
                kernel_sizes[i],
                fstride,
            )

            if batch_norm:
                self.encoder_blocks.append(
                    nn.Sequential(
                        conv_layer,
                        nn.BatchNorm2d(num_channels_encoder[i + 1]),
                        nonlinearity(),
                    )
                )
            else:
                self.encoder_blocks.append(nn.Sequential(conv_layer, nonlinearity()))

            self.skipcon_convs.append(
                nn.Conv2d(
                    num_channels_encoder[i + 1],
                    num_channels_encoder[i + 1],
                    (1, 1),
                    (1, 1),
                    (0, 0),
                    bias=True,
                )
            )

            fsize = int((fsize - kernel_sizes[i][1]) / fstride + 1)
            fsizes.append(fsize)

        recurr_input_size = int(fsize * num_channels_encoder[-1])

        # gru
        self.group_gru = GroupedGRU(
            input_size=recurr_input_size,
            hidden_size=recurr_input_size,
            num_layers=n_gru_layers,
            groups=n_gru_groups,
            bidirectional=False,
        )

        # decoder
        self.decoder_blocks = nn.ModuleList()
        for i in range(len(num_channels_decoder) - 1):
            fsize = (fsize - 1) * fstride + kernel_sizes[len(self.encoder_blocks) + i][1]
            output_padding_f_axis = fsizes[-i - 2] - fsize
            conv_layer = CausalTransConvBlock(
                num_channels_decoder[i],
                num_channels_decoder[i + 1],
                kernel_sizes[len(self.encoder_blocks) + i],
                fstride,
                output_padding_f_axis=output_padding_f_axis,
            )
            fsize = fsizes[-i - 2]

            if batch_norm:
                block_wo_nonlin = nn.Sequential(
                    conv_layer, nn.BatchNorm2d(num_channels_decoder[i + 1])
                )
            else:
                block_wo_nonlin = conv_layer

            if i < (len(num_channels_decoder) - 2):
                self.decoder_blocks.append(
                    nn.Sequential(block_wo_nonlin, nonlinearity())
                )
            else:
                self.decoder_blocks.append(
                    nn.Sequential(block_wo_nonlin, output_nonlinearity())
                )

    def forward(self, x):
        conv_skip_outs = []
        for l, s in zip(self.encoder_blocks, self.skipcon_convs):
            x = l(x)
            conv_skip_outs.append(s(x))

        batch_size, n_channels, n_frames, n_bins = x.shape

        x = x.permute(0, 2, 1, 3)
        x = x.reshape(batch_size, n_frames, n_channels * n_bins)
        x, _ = self.group_gru(x)

        x = x.reshape(batch_size, n_frames, n_channels, n_bins)
        x = x.permute(0, 2, 1, 3)

        d = x
        for i in range(len(self.decoder_blocks)):
            skip_connected = d + conv_skip_outs[-i - 1]
            d = self.decoder_blocks[i](skip_connected)

        return d


def asymmetric_hann_window_pair(analysis_winlen, synthesis_winlen):
    analysis_fade_in_len = analysis_winlen - synthesis_winlen // 2

    analysis_window = torch.cat(
        [
            torch.hann_window(analysis_fade_in_len * 2, periodic=True)[
                :analysis_fade_in_len
            ],
            torch.hann_window(synthesis_winlen, periodic=True)[synthesis_winlen // 2 :],
        ]
    )

    synthesis_window = torch.hann_window(synthesis_winlen, periodic=True)

    # ensure the resulting window is a hann window
    synthesis_window = synthesis_window / analysis_window[-synthesis_winlen:]
    return analysis_window, synthesis_window


class SpeechEnhancementModel(nn.Module):
    def __init__(
        self,
        crn_config,
        hopsize,
        winlen,
        method,
        num_filter_frames,
        learnable_transforms,
        algorithmic_delay=0,
    ):
        """
        algorithmic delay: number of samples of algorithmic delay for time-domain filtering method
        """
        super().__init__()

        self.num_filter_frames = num_filter_frames

        crn_config["fsize_input"] = winlen // 2 + 1
        crn_config["num_output_channels"] = self.num_filter_frames * 2

        self.crn = CRN(**crn_config)

        self.winlen = winlen
        self.hopsize = hopsize

        if method == "complex_filter":
            analysis_window, synthesis_window = asymmetric_hann_window_pair(
                winlen, hopsize * 2
            )
        elif method == "time_domain_filtering":
            analysis_window = torch.ones(winlen)
            self.crossfade_window = nn.Parameter(
                torch.hann_window(hopsize * 2, periodic=True), requires_grad=False
            )

        stft_matrix = (
            torch.from_numpy(np.fft.rfft(np.eye(winlen))) * analysis_window[:, None]
        )

        self.forward_transform = nn.Parameter(
            (1 / np.sqrt(winlen))
            * torch.cat([stft_matrix.real.float(), stft_matrix.imag.float()], dim=1).T[
                :, None, :
            ],
            requires_grad=learnable_transforms,
        )  # [1, 2 * (winlen // 2 + 1), winlen]

        istft_matrix_real = torch.from_numpy(
            np.fft.irfft(np.eye(winlen // 2 + 1))
        )  # [winlen // 2 + 1, winlen]
        istft_matrix_imag = torch.from_numpy(np.fft.irfft(1j * np.eye(winlen // 2 + 1)))

        if method == "complex_filter":
            istft_matrix_real = (
                istft_matrix_real[:, -2 * hopsize :] * synthesis_window[None, :]
            )
            istft_matrix_imag = (
                istft_matrix_imag[:, -2 * hopsize :] * synthesis_window[None, :]
            )
            self.inverse_transform = nn.Parameter(
                np.sqrt(winlen)
                * torch.cat(
                    [istft_matrix_real.float(), istft_matrix_imag.float()], dim=0
                )[:, None, :],
                requires_grad=learnable_transforms,
            )  # [2 * (winlen // 2 + 1), 1, 2 * hopsize]
            self.algorithmic_delay = 2 * hopsize
        elif method == "time_domain_filtering":
            self.inverse_transform = nn.Parameter(
                torch.cat(
                    [istft_matrix_real.float(), istft_matrix_imag.float()], dim=0
                ),
                requires_grad=learnable_transforms,
            )  # [2 * (winlen // 2 + 1), winlen]
            self.algorithmic_delay = algorithmic_delay
            self.pad_size = winlen - self.algorithmic_delay
            assert self.pad_size >= 0

        self.method = method

    def forward(self, x_in):
        """
        x_in: audio input signal [B, T]
        """
        xlen = x_in.size(-1)

        # Handle "time_domain_filtering" case
        if self.method == "time_domain_filtering":
            x = torch.nn.functional.pad(
                x_in, (self.pad_size, 0)
            )  # Padding for time-domain

        x = x_in.unsqueeze(1)  # [B, 1, T]
        x = nn.functional.conv1d(
            x, self.forward_transform, stride=self.hopsize
        )  # [B, 2 * winlen // 2 + 1, T // hopsize]
        x = x.view(x.size(0), 2, -1, x.size(-1)).transpose(
            2, 3
        )  # [B, 2, T // hopsize, winlen // 2 + 1]

        x_complex = torch.complex(
            x[:, 0, ...], x[:, 1, ...]
        )  # [B, T // hopsize, winlen // 2 + 1]

        # Nonlinear transformation
        x_mag = torch.clip(torch.abs(x_complex), 1e-12)
        x = x_mag**0.3 * x_complex / x_mag

        x = torch.stack([x.real, x.imag], dim=1)

        x = self.crn(x)  # [B, 2 * num_filter_frames, T // hopsize, winlen // 2 + 1]

        # Reshape for multi-frame filtering
        x = x.view(
            x.size(0), 2, self.num_filter_frames, x.size(-2), x.size(-1)
        )  # [B, 2, num_filter_frames, T // hopsize, winlen // 2 + 1]

        if self.method == "complex_filter":
            x = torch.complex(
                x[:, 0, ...], x[:, 1, ...]
            )  # [B, num_filter_frames, T // hopsize, winlen // 2 + 1]

            # Multi-frame filtering, complex causal convolution
            x_complex = nn.functional.pad(
                x_complex, (0, 0, self.num_filter_frames - 1, 0)
            )
            x_complex = x_complex.unfold(1, self.num_filter_frames, 1).permute(
                0, 3, 1, 2
            )  # [B, num_filter_frames, T // hopsize, winlen // 2 + 1]

            y = torch.sum(x_complex * x, 1)  # [B, T // hopsize, winlen // 2 + 1]

            # Inverse transform
            y = torch.cat([y.real, y.imag], dim=-1).permute(
                0, 2, 1
            )  # [B, 2 * (winlen // 2 + 1), T // hopsize]

            y = nn.functional.pad(
                nn.functional.conv_transpose1d(
                    y,
                    self.inverse_transform,
                    stride=self.hopsize,
                ),
                (self.winlen - 2 * self.hopsize, 0),
            )

            # Pad to input length
            lendiff = y.size(-1) - xlen
            if lendiff > 0:
                y = y[..., :-lendiff]
            elif lendiff < 0:
                y = torch.nn.functional.pad(y, (0, -lendiff))

            return y

        elif self.method == "time_domain_filtering":
            # Reshape for inverse transform
            x = x.permute(0, 2, 3, 1, 4)
            x = x.reshape(
                x.size(0), x.size(1), x.size(2), -1
            )  # [B, num_filter_frames, T // hopsize, 2 * (winlen // 2 + 1)]
            filt = (
                x @ self.inverse_transform
            )  # [B, num_filter_frames, T // hopsize, winlen], time-domain filtering

            # Frame-wise convolution and overlap-add using crossfade window
            x_frame = x_in.unfold(
                1, 2 * self.hopsize, self.hopsize
            )  # [B, T // hopsize, 2 * hopsize]
            x_frame = (
                x_frame * self.crossfade_window[None, None, :]
            )  # [B, T // hopsize, 2 * hopsize]

            n_frames_shorter = min(x_frame.size(1), filt.size(2))

            x_frame = x_frame[:, :n_frames_shorter, :]  # [B, T // hopsize, 2 * hopsize]
            filt = filt[
                :, :, :n_frames_shorter, :
            ]  # [B, num_filter_frames, T // hopsize, winlen]

            # use unfolding both along the frame axis and along the sample axis to do multiframe time-domain filtering
            # y = torch.nn.functional.conv1d(
            x_frame = nn.functional.pad(
                x_frame, (self.winlen - 1, 0, self.num_filter_frames - 1, 0)
            )  # [B, T // hopsize + num_filter_frames - 1, 2 * hopsize + winlen - 1]
            x_frame = x_frame.unfold(
                1, self.num_filter_frames, 1
            )  # [B, T // hopsize, 2 * hopsize + winlen - 1, num_filter_frames]
            x_frame = x_frame.unfold(
                2, self.winlen, 1
            )  # [B, T // hopsize, 2 * hopsize, num_filter_frames, winlen]

            print(x_frame.shape)
            x_filt = (
                (x_frame * filt.transpose(1, 2)[:, :, None, :, :]).sum(4).sum(3)
            )  # [B, T // hopsize, 2 * hopsize]

            # overlap-add
            y = nn.functional.fold(
                x_filt.permute(0, 2, 1),
                output_size=(
                    1,
                    (x_filt.shape[1] - 1) * self.hopsize + 2 * self.hopsize,
                ),
                kernel_size=(1, 2 * self.hopsize),
                stride=(1, self.hopsize),
            ).squeeze(2).squeeze(1)

            return y[:, :xlen]


if __name__ == "__main__":
    model = SpeechEnhancementModel(
        crn_config={
            "num_channels_encoder": [2, 48, 64, 80, 96], #[2, 32, 45, 57, 68]
            "kernel_sizes": [(5, 3)] + 7 * [(1, 3)],
            "fstride": 2,
            "n_gru_layers": 1,
            "n_gru_groups": 4,
            "nonlinearity": "ReLU",
            "output_nonlinearity": "Tanh",
            "batch_norm": False,
        },
        hopsize=160,
        winlen=320,
        method="time_domain_filtering",
        num_filter_frames=5,
        learnable_transforms=True,
    )
    # x = torch.randn(1, 16000)

    macs, params = get_model_complexity_info(
        model, (20 * 16000,), as_strings=False, print_per_layer_stat=True
    )

    print("GMacs/s", macs / 10**9 / 20)
    print("M Params", params / 10**6)

    print("Test passed")
