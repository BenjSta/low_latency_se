import torch
from torch import nn
import numpy as np
from ptflops import get_model_complexity_info
from third_party.grouped_gru import GroupedGRU



class SnakeBeta(nn.Module):
    """
    A modified Snake function which uses separate parameters for the magnitude of the periodic components
    Shape:
        - Input: (B, C, T)
        - Output: (B, C, T), same shape as the input
    Parameters:
        - alpha - trainable parameter that controls frequency
        - beta - trainable parameter that controls magnitude
    References:
        - This activation function is a modified version based on this paper by Liu Ziyin, Tilman Hartwig, Masahito Ueda:
        https://arxiv.org/abs/2006.08195
    Examples:
        >>> a1 = snakebeta(256)
        >>> x = torch.randn(256)
        >>> x = a1(x)
    """

    def __init__(
        self, in_features, alpha=1.0, alpha_trainable=True, alpha_logscale=False
    ):
        """
        Initialization.
        INPUT:
            - in_features: shape of the input
            - alpha - trainable parameter that controls frequency
            - beta - trainable parameter that controls magnitude
            alpha is initialized to 1 by default, higher values = higher-frequency.
            beta is initialized to 1 by default, higher values = higher-magnitude.
            alpha will be trained along with the rest of your model.
        """
        super(SnakeBeta, self).__init__()
        self.in_features = in_features

        # Initialize alpha
        self.alpha_logscale = alpha_logscale
        if self.alpha_logscale:  # Log scale alphas initialized to zeros
            self.alpha = nn.Parameter(torch.zeros(in_features) * alpha)
            self.beta = nn.Parameter(torch.zeros(in_features) * alpha)
        else:  # Linear scale alphas initialized to ones
            self.alpha = nn.Parameter(torch.ones(in_features) * alpha)
            self.beta = nn.Parameter(torch.ones(in_features) * alpha)

        self.alpha.requires_grad = alpha_trainable
        self.beta.requires_grad = alpha_trainable

        self.no_div_by_zero = 0.000000001

    def forward(self, x):
        """
        Forward pass of the function.
        Applies the function to the input elementwise.
        SnakeBeta ∶= x + 1/b * sin^2 (xa)
        """
        alpha = self.alpha.unsqueeze(0).unsqueeze(-1)  # Line up with x to [B, C, T]
        beta = self.beta.unsqueeze(0).unsqueeze(-1)
        if self.alpha_logscale:
            alpha = torch.exp(alpha)
            beta = torch.exp(beta)
        x = x + (1.0 / (beta + self.no_div_by_zero)) * pow(torch.sin(x * alpha), 2)

        return x


class CausalConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, fstride, bias=True):
        super().__init__()
        # Define a causal convolutional layer
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            padding=(kernel_size[0] - 1, 0),
            bias=bias,
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        # Apply convolution and trim the output to maintain causality
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
        # Define a causal transposed convolutional layer
        self.conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=(1, fstride),
            output_padding=(0, output_padding_f_axis),
            bias=bias,
        )
        self.kernel_size = kernel_size

    def forward(self, x):
        # Apply transposed convolution and trim the output to maintain causality
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

        nonlinearity = getattr(nn, nonlinearity)
        output_nonlinearity = getattr(nn, output_nonlinearity)
        num_channels_decoder = num_channels_encoder[:0:-1] + [num_output_channels]

        # Build encoder, GRU, and decoder blocks
        self.encoder_blocks, self.skipcon_convs, fsizes = self._build_encoder(
            num_channels_encoder,
            kernel_sizes[: -len(num_channels_decoder) + 1],
            fstride,
            nonlinearity,
            batch_norm,
            fsize_input,
        )
        self.group_gru, self.recurr_input_size = self._build_gru(
            fsizes[-1], num_channels_encoder[-1], n_gru_layers, n_gru_groups
        )
        self.decoder_blocks = self._build_decoder(
            num_channels_decoder,
            kernel_sizes[-len(num_channels_encoder) + 1:],
            fstride,
            nonlinearity,
            output_nonlinearity,
            batch_norm,
            fsizes,
        )

    def _build_encoder(
        self,
        num_channels_encoder,
        kernel_sizes,
        fstride,
        nonlinearity,
        batch_norm,
        fsize_input,
    ):
        fsize = fsize_input
        fsizes = [fsize]
        encoder_blocks = nn.ModuleList()
        skipcon_convs = nn.ModuleList()
        for i in range(len(num_channels_encoder) - 1):
            # Create encoder blocks with optional batch normalization and nonlinearity
            conv_layer = CausalConvBlock(
                num_channels_encoder[i],
                num_channels_encoder[i + 1],
                kernel_sizes[i],
                fstride,
            )
            if batch_norm:
                encoder_blocks.append(
                    nn.Sequential(
                        conv_layer,
                        nn.BatchNorm2d(num_channels_encoder[i + 1]),
                        nonlinearity(),
                    )
                )
            else:
                encoder_blocks.append(nn.Sequential(conv_layer, nonlinearity()))
            skipcon_convs.append(
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
        return encoder_blocks, skipcon_convs, fsizes

    def _build_gru(self, fsize, num_channels_encoder_last, n_gru_layers, n_gru_groups):
        # Create a grouped GRU layer
        recurr_input_size = int(fsize * num_channels_encoder_last)
        group_gru = GroupedGRU(
            input_size=recurr_input_size,
            hidden_size=recurr_input_size,
            num_layers=n_gru_layers,
            groups=n_gru_groups,
            bidirectional=False,
        )
        return group_gru, recurr_input_size

    def _build_decoder(
        self,
        num_channels_decoder,
        kernel_sizes,
        fstride,
        nonlinearity,
        output_nonlinearity,
        batch_norm,
        fsizes,
    ):
        decoder_blocks = nn.ModuleList()
        for i in range(len(num_channels_decoder) - 1):
            # Create decoder blocks with optional batch normalization and nonlinearity
            fsize = (fsizes[-i - 1] - 1) * fstride + kernel_sizes[i][1]
            output_padding_f_axis = fsizes[-i - 2] - fsize
            conv_layer = CausalTransConvBlock(
                num_channels_decoder[i],
                num_channels_decoder[i + 1],
                kernel_sizes[i],
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
                decoder_blocks.append(nn.Sequential(block_wo_nonlin, nonlinearity()))
            else:
                decoder_blocks.append(
                    nn.Sequential(block_wo_nonlin, output_nonlinearity())
                )
        return decoder_blocks

    def forward(self, x):
        conv_skip_outs = []
        # Pass input through encoder blocks and collect skip connections
        for l, s in zip(self.encoder_blocks, self.skipcon_convs):
            x = l(x)
            conv_skip_outs.append(s(x))

        # Reshape and pass through GRU
        batch_size, n_channels, n_frames, n_bins = x.shape
        x = x.permute(0, 2, 1, 3).reshape(batch_size, n_frames, n_channels * n_bins)
        x = x.contiguous()
        x_hidden, _ = self.group_gru(x)
        x = x_hidden.reshape(batch_size, n_frames, n_channels, n_bins).permute(
            0, 2, 1, 3
        )

        # Pass through decoder blocks with skip connections
        d = x
        for i in range(len(self.decoder_blocks)):
            skip_connected = d + conv_skip_outs[-i - 1]
            d = self.decoder_blocks[i](skip_connected)

        return d, x_hidden


def asymmetric_hann_window_pair(analysis_winlen, synthesis_winlen):
    # Create asymmetric Hann windows for analysis and synthesis
    analysis_fade_in_len = analysis_winlen - synthesis_winlen // 2
    analysis_window = (
        torch.cat(
            [
                torch.hann_window(analysis_fade_in_len * 2, periodic=True)[
                    :analysis_fade_in_len
                ],
                torch.hann_window(synthesis_winlen, periodic=True)[
                    synthesis_winlen // 2 :
                ],
            ]
        )
        ** 0.5
    )
    synthesis_window = torch.hann_window(synthesis_winlen, periodic=True) / torch.clip(
        analysis_window[analysis_window.shape[0] - synthesis_winlen :], 1e-12
    )
    return analysis_window, synthesis_window


class SpeechEnhancementModel(nn.Module):
    def __init__(
        self,
        crn_config,
        hopsize,
        winlen,
        method,
        use_mlp=False,
        num_filter_frames=1,
        learnable_transforms=True,
        downsample_factor=1,
        algorithmic_delay_nn=0,
        algorithmic_delay_filtering=0,
        filtlen = None
    ):
        """
        Initialize the Speech Enhancement Model.

        Args:
            crn_config (dict): Configuration for the CRN model
            hopsize (int): Hop size of the STFT
            winlen (int): Window length of the STFT
            method (str): Method for speech enhancement. One of "complex_filter" or "time_domain_filtering"
            use_mlp (bool): Whether to use an MLP after the CRN (only applies for complex_filter), defaults to False
            num_filter_frames (int): Number of frames to filter in the past and future, defaults to 1
            learnable_transforms (bool): Whether to make the STFT transforms learnable, defaults to True
            downsample_factor (int): Factor to downsample / upsample before/after the neural network, only applies for complex_filter, defaults to 1
            algorithmic_delay_nn (int): Algorithmic delay for the neural network, only applies for time_domain_filter, defaults to 0
            algorithmic_delay_filtering (int): Algorithmic delay for time domain filtering, only applies for time_domain_filter, defaults to 0
            filtlen (int): Length of the filter to be learned, only applies for time_domain_filter, defaults to None, which is equal to winlen
        """
        super().__init__()

        self.num_filter_frames = num_filter_frames

        if method=="time_domain_filtering":
            self.downsample_factor = 1
        else:
            self.downsample_factor = downsample_factor
            
        self.real_valued = False

        crn_config["fsize_input"] = winlen // 2 + 1
        crn_config["num_output_channels"] = self.num_filter_frames * (
            1 if self.real_valued else 3
        )
        crn_config["output_nonlinearity"] = "Sigmoid" if method == "complex_filter" and not(use_mlp) else "ELU"

        # Initialize CRN model
        self.crn = CRN(**crn_config)



        self.winlen = winlen
        self.hopsize = hopsize
        if filtlen is None:
            filtlen = winlen
        self.filtlen = filtlen

        if method == "complex_filter":
            # Create asymmetric Hann windows for complex filtering
            analysis_window, synthesis_window = asymmetric_hann_window_pair(
                winlen, hopsize * 2
            )
        elif method == "time_domain_filtering":
            # use an asymmetric analysis window
            analysis_window, _ = asymmetric_hann_window_pair(
                winlen, algorithmic_delay_nn * 2
            )

            # use an asymmetric crossfade / synthesis window
            fade_in = torch.hann_window(2 * algorithmic_delay_filtering)[
                :algorithmic_delay_filtering
            ]
            crossfade_win_first_half = torch.cat(
                [fade_in, torch.ones(hopsize - algorithmic_delay_filtering)]
            )
            crossfade_win_second_half = 1 - crossfade_win_first_half
            self.crossfade_window = nn.Parameter(
                torch.cat([crossfade_win_first_half, crossfade_win_second_half]),
                requires_grad=True,
            )

        # Initialize forward and inverse STFT transforms
        stft_matrix = (
            torch.from_numpy(np.fft.rfft(np.eye(winlen))) * analysis_window[:, None]
        )
        self.forward_transform = nn.Parameter(
            (1 / np.sqrt(winlen))
            * torch.cat([stft_matrix.real.float(), stft_matrix.imag.float()], dim=1).T[
                :, None, :
            ],
            requires_grad=learnable_transforms,
        )

        istft_matrix_real = torch.from_numpy(np.fft.irfft(np.eye(winlen // 2 + 1)))
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
            )
            
            if use_mlp:
                mlp_out = 3 * (winlen // 2 + 1)
                self.mlp = nn.Sequential(
                    nn.Linear(3 * (winlen // 2 + 1), mlp_out),
                    nn.Sigmoid()
                )

            self.algorithmic_delay_nn = 2 * hopsize
            self.algorithmic_delay_filtering = None
        elif method == "time_domain_filtering":
            assert (
                algorithmic_delay_filtering <= winlen // 2
            ), "Algorithmic delay must be less or equal to winlen // 2"

            # fade_in_len = min(winlen // 4, algorithmic_delay_filtering)
            # fade_in = torch.hann_window(2 * fade_in_len)[:fade_in_len]

            # # use a symmetric fade_in / fade_out if algorithmic_delay is winlen // 2
            # # otherwise use a fade_out longer than_fade_in, but at maximum fade_out_len = winlen // 3
            # # use a zero taper of at maximum winlen // 6

            # fade_out_len = max(
            #     int((1 - algorithmic_delay_filtering / (winlen / 2)) * winlen / 2),
            #     fade_in_len,
            # )
            # zeros_len = 0
            # # zeros_len = int((1 - algorithmic_delay_filtering / (winlen / 2)) * winlen / 6)

            # end_tapering = torch.cat(
            #     [
            #         torch.hann_window(2 * fade_out_len)[fade_out_len:],
            #         torch.zeros(zeros_len),
            #     ]
            # )

            # window = torch.cat(
            #     [
            #         fade_in,
            #         torch.ones(winlen - fade_in_len - len(end_tapering)),
            #         end_tapering,
            #     ]
            # )
            # self.inverse_transform = nn.Parameter(
            #     torch.roll(
            #         torch.cat(
            #             [istft_matrix_real.float(), istft_matrix_imag.float()],
            #             dim=0,
            #         ),
            #         algorithmic_delay_filtering,
            #         dims=-1,
            #     )
            #     * window[None, :],
            #     requires_grad=learnable_transforms,
            # )

            self.inverse_transform = nn.Sequential(
                nn.Linear(3 * (winlen // 2 + 1), winlen),
                nn.ELU(),
                nn.Linear(winlen, filtlen),
                nn.Tanh()
            )

            self.algorithmic_delay_nn = algorithmic_delay_nn
            self.pad_size_nn = winlen - self.algorithmic_delay_nn
            self.algorithmic_delay_filtering = algorithmic_delay_filtering
            assert (
                self.pad_size_nn >= 0
            ), "Algorithmic delay must be less than or equal to winlen"

        self.method = method
        self.std_norm = nn.Parameter(
            torch.ones(winlen // 2 + 1), requires_grad=learnable_transforms
        )
        if not self.real_valued:
            self.v = nn.Parameter(
                torch.tensor(
                    [[1, -1 / 2, -1 / 2], [0, np.sqrt(3) / 2, -np.sqrt(3) / 2]],
                    dtype=torch.float32,
                )
            )

    def set_std_norm(self, std_norm):
        self.std_norm.data = std_norm.to(self.std_norm.device)

    def forward(self, x_in, return_features=False):
        xlen = x_in.size(-1)

        if self.method == "time_domain_filtering":
            x = torch.nn.functional.pad(x_in, (self.pad_size_nn, 0))
        else:
            x = x_in

        x = x.unsqueeze(1)
        x = nn.functional.conv1d(x, self.forward_transform, stride=self.hopsize)
        x = x.view(x.size(0), 2, -1, x.size(-1)).transpose(2, 3)

        x_complex = torch.complex(x[:, 0, ...], x[:, 1, ...])
        x_mag = torch.clip(torch.abs(x_complex), 1e-12)
        x = x_mag**0.3 * x_complex / x_mag
        if return_features:
            return x

        x = x / self.std_norm[None, None, :]
        x = torch.stack([x.real, x.imag], dim=1)

        # downsample
        nfr = x.size(2)
        x = x[:, :, :: self.downsample_factor, :]
        x, bottleneck_out = self.crn(x)
        # upsample again by repeating

        if self.real_valued:
            x = x.view(x.size(0), 1, self.num_filter_frames, x.size(-2), x.size(-1))
            x = nn.functional.pad(x, (0,0,0,0,0,0,0,1), value=0) #pad to add a zero imaginary part
        else:
            x = x.view(x.size(0), 3, self.num_filter_frames, x.size(-2), x.size(-1))
        
        if hasattr(self, "mlp"):
            x = x.permute(0,2,3,4,1)
            x = x.reshape(*x.shape[:3], -1)
            x = self.mlp(x)
            x = x.reshape(*x.shape[:3], x.shape[3] // 3, 3)
            x = x.permute(0,4,1,2,3)
            
        x = torch.repeat_interleave(x, self.downsample_factor, dim=3)[..., :nfr, :]


        if self.method == "complex_filter":
            x = torch.einsum("bijkm,li->bljkm", x, self.v)
            return self._complex_filtering(x, x_complex, xlen)
        elif self.method == "time_domain_filtering":
            return self._time_domain_filtering(x, x_in, bottleneck_out, xlen)

    def _complex_filtering(self, fd_filt, x_complex, xlen):
        fd_filt_complex = torch.complex(fd_filt[:, 0, ...], fd_filt[:, 1, ...])
        x_complex = nn.functional.pad(x_complex, (0, 0, self.num_filter_frames - 1, 0))
        x_complex = x_complex.unfold(1, self.num_filter_frames, 1).permute(0, 3, 1, 2)
        y = torch.einsum("bijk,bijk->bjk", x_complex, fd_filt_complex)
        y = torch.cat([y.real, y.imag], dim=-1).permute(0, 2, 1)
        y = nn.functional.pad(
            nn.functional.conv_transpose1d(
                y, self.inverse_transform, stride=self.hopsize
            ),
            (self.winlen - 2 * self.hopsize, 0),
        ).squeeze(1)
        lendiff = y.size(-1) - xlen
        if lendiff > 0:
            y = y[..., :-lendiff]
        elif lendiff < 0:
            y = torch.nn.functional.pad(y, (0, -lendiff))
        return y

    def _time_domain_filtering(self, fd_filt, x_in, bottleneck_out, xlen):
        fd_filt = fd_filt.permute(0, 2, 3, 1, 4)
        # if self.adaptive_phase_shift:
        #     phase_shift = (
        #         self.linphase_regressor(bottleneck_out).permute(0, 2, 1)[:, :, :, None]
        #         * (self.winlen / 2)
        #         * self.fvec[None, None, None, :]
        #     )  # allow a maximum shift of winlen / 2
        #     phasor = torch.exp(-1j * phase_shift)
        #     fd_filt_complex = fd_filt[:, :, :, 0, :] + 1j * fd_filt[:, :, :, 1, :]
        #     fd_filt_complex = fd_filt_complex * phasor
        #     fd_filt = torch.stack([fd_filt_complex.real, fd_filt_complex.imag], dim=3)

        fd_filt = fd_filt.reshape(fd_filt.size(0), fd_filt.size(1), fd_filt.size(2), -1)
        filt = self.inverse_transform(fd_filt)

        x_frame = nn.functional.pad(
            x_in,
            (
                self.winlen - self.algorithmic_delay_filtering,
                self.algorithmic_delay_filtering,
            ),
        ).unfold(1, 2 * self.hopsize + self.winlen - 1, self.hopsize)

        n_frames_shorter = min(x_frame.size(1), filt.size(2))
        x_frame = x_frame[
            :, :n_frames_shorter, :
        ]  # [batch, n_frames, 2 * hopsize + winlen - 1]
        filt = filt[
            :, :, :n_frames_shorter, :
        ]  # [batch, num_filter_frames, n_frames, winlen]

        x_frame = (
            nn.functional.pad(x_frame, (0, 0, self.num_filter_frames - 1, 0))
            .unfold(1, self.num_filter_frames, 1)
            .permute(0, 3, 1, 2)
        )  # [batch, num_filter_frames, n_frames, 2 * hopsize + winlen - 1]

        # now do the convolution using FFT
        x_fft = torch.fft.rfft(x_frame, n=2 * self.hopsize + self.winlen - 1, dim=-1)
        filt_fft = torch.fft.rfft(filt, n=2 * self.hopsize + self.winlen - 1, dim=-1)
        x_filt = torch.fft.irfft(
            torch.sum(x_fft * filt_fft, 1), n=2 * self.hopsize + self.winlen - 1, dim=-1
        )

        # use only valid part of the convolution
        x_filt = x_filt[
            :, :, -2 * self.hopsize :
        ]  # [batch, n_frames_shorter, 2 * hopsize]

        # apply crossfade window
        x_filt = x_filt * self.crossfade_window[None, None, :]

        y = (
            nn.functional.fold(
                x_filt.permute(0, 2, 1),
                output_size=(
                    1,
                    (n_frames_shorter - 1) * self.hopsize + 2 * self.hopsize,
                ),
                kernel_size=(1, 2 * self.hopsize),
                stride=(1, self.hopsize),
            )
            .squeeze(2)
            .squeeze(1)
        )
        return y[:, :xlen]


if __name__ == "__main__":
    # Example usage of the SpeechEnhancementModel
    model = SpeechEnhancementModel(
        crn_config={
            "num_channels_encoder": [2, 32, 40, 48, 56],
            "kernel_sizes": [(5, 3)] + 7 * [(1, 3)],
            "fstride": 2,
            "n_gru_layers": 1,
            "n_gru_groups": 4,
            "nonlinearity": "ELU",
            "output_nonlinearity": "Tanh",
            "batch_norm": False,
        },
        hopsize=160,
        winlen=320,
        method="time_domain_filtering",
        num_filter_frames=5,
        learnable_transforms=True,
        adaptive_phase_shift=True,
        linphase_regressor_hidden_size=64,
    )

    # Calculate model complexity
    macs, params = get_model_complexity_info(
        model, (20 * 16000,), as_strings=False, print_per_layer_stat=True
    )
    print("GMacs/s", macs / 10**9 / 20)
    print("M Params", params / 10**6)
    print("Test passed")
