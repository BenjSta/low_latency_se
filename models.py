import torch
from torch import nn
import numpy as np
from ptflops import get_model_complexity_info
from third_party.grouped_gru import GroupedGRU


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
            if batch_norm and i < (len(num_channels_decoder) - 2):
                # Apply batch normalization only if not the last layer
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
        algorithmic_delay_nn,
        use_mlp=False,
        num_filter_frames=1,
        learnable_transforms=True,
        downsample_factor=1,
        algorithmic_delay_filtering=0,
        filtlen = None,
        inv_trans_mlp = True,
        output_mapping = "star",
    ):
        """
        Initialize the Speech Enhancement Model.

        Args:
            crn_config (dict): Configuration for the CRN model
            hopsize (int): Hop size of the STFT
            winlen (int): Window length of the STFT
            method (str): Method for speech enhancement. One of "complex_filter", "time_domain_filtering", or "complex_mapping"
            algorithmic_delay_nn (int): Algorithmic delay for the neural network, defaults to 0
            use_mlp (bool): Whether to use an MLP after the CRN (only applies for complex_filter and complex_mapping), defaults to False
            num_filter_frames (int): Number of frames to filter (multiframe filtering), does not apply for complex mapping, defaults to 1
            learnable_transforms (bool): Whether to make the STFT transforms learnable, defaults to True
            downsample_factor (int): Factor to downsample / upsample before/after the neural network, only applies for complex_filter, defaults to 1
            algorithmic_delay_filtering (int): Algorithmic delay for time domain filtering, only applies for time_domain_filter, defaults to 0
            filtlen (int): Length of the filter to be learned, only applies for time_domain_filter, defaults to None, which is equal to winlen
            output_mapping (str): Output mapping method, one of "star" or "re/im"
        """
        super().__init__()

        self.output_mapping = output_mapping
        if method=="complex_mapping":
            self.num_filter_frames = 1
        else:
            self.num_filter_frames = num_filter_frames

        if method=="time_domain_filtering":
            self.downsample_factor = 1
        else:
            self.downsample_factor = downsample_factor
            
        crn_config["fsize_input"] = winlen // 2 + 1
        crn_config["num_output_channels"] =  self.num_filter_frames * 3 if (self.output_mapping in ["star", "gated", "gatednorm"] or method == "time_domain_filtering" or use_mlp) else self.num_filter_frames * 2
        crn_config["output_nonlinearity"] = (#"ELU" if (method == "time_domain_filtering" or use_mlp) else (
            "Sigmoid" if method in ["complex_filter", "time_domain_filtering"] else "Softplus") if self.output_mapping == "star" else "Identity"

        # Initialize CRN model
        self.crn = CRN(**crn_config)


        self.winlen = winlen
        self.hopsize = hopsize
        if filtlen is None:
            filtlen = winlen
        self.filtlen = filtlen
        self.method = method

        if method == "time_domain_filtering":
            self.pad_size_nn = (winlen - algorithmic_delay_nn)
        else:
            self.pad_size_nn = (2*hopsize - algorithmic_delay_nn)

        if method == "complex_filter" or method == "complex_mapping":
            # Create asymmetric Hann windows for complex filtering
            analysis_window, synthesis_window = asymmetric_hann_window_pair(
                winlen, hopsize * 2
            )
        elif method == "time_domain_filtering":
            # use an asymmetric analysis window
            analysis_window, _ = asymmetric_hann_window_pair(
                winlen, np.clip(algorithmic_delay_nn * 2, 0, winlen)
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

        if method == "complex_filter" or method == "complex_mapping":
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
                mlp_out = (3 if self.output_mapping in ["star", 'gated', 'gatednorm'] else 2) * (winlen // 2 + 1)
                self.mlp = nn.Sequential(
                    nn.Linear(3 * (winlen // 2 + 1), mlp_out),
                    (
                        nn.Sigmoid() if method == "complex_filter" else nn.Softplus()) if self.output_mapping == "star" else (
                        nn.Identity()
                    ))

            self.algorithmic_delay_filtering = None
        elif method == "time_domain_filtering":
            assert (
                algorithmic_delay_filtering <= winlen // 2
            ), "Algorithmic delay must be less or equal to winlen // 2"

            if inv_trans_mlp:
                self.inverse_transform = nn.Sequential(
                    nn.Linear(3 * (winlen // 2 + 1), winlen),
                    #nn.BatchNorm1d(winlen),
                    nn.ELU(),
                    nn.Linear(winlen, winlen),
                    #nn.BatchNorm1d(winlen),
                    nn.ELU(),
                    nn.Linear(winlen, filtlen),
                    nn.Tanh(),
                )
            else:
                self.inverse_transform = nn.Sequential(
                    nn.Linear(3 * (winlen // 2 + 1), filtlen,bias=False),
                    
                )
            self.algorithmic_delay_filtering = algorithmic_delay_filtering
            

        self.std_norm = nn.Parameter(
            torch.ones(winlen // 2 + 1), requires_grad=learnable_transforms
        )
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

        
        # filter input path only needed for complex_filter
        if self.method == "complex_filter":
            x = x_in.unsqueeze(1)
            x = nn.functional.conv1d(x, self.forward_transform, stride=self.hopsize)
            x = x.view(x.size(0), 2, -1, x.size(-1)).transpose(2, 3)

            x_complex = torch.complex(x[:, 0, ...], x[:, 1, ...])


        # neural network input path
        if self.pad_size_nn != 0 or self.method != "complex_filter":
            if self.pad_size_nn >= 0:
                x = torch.nn.functional.pad(x_in, (self.pad_size_nn, 0))
            else:
                x = x_in[:, -self.pad_size_nn :]
                x = torch.nn.functional.pad(x, (0, -self.pad_size_nn))

            x = x.unsqueeze(1)
            x = nn.functional.conv1d(x, self.forward_transform, stride=self.hopsize)
            x = x.view(x.size(0), 2, -1, x.size(-1)).transpose(2, 3)
            x = torch.complex(x[:, 0, ...], x[:, 1, ...])
        else:
            x = x_complex
        
        x_mag = torch.clip(torch.abs(x), 1e-12)
        x = x_mag**0.3 * x / x_mag
        if return_features:
            return x

        x = x / self.std_norm[None, None, :]
        x = torch.stack([x.real, x.imag], dim=1)

        # downsample
        nfr = x.size(2)
        x = x[:, :, :: self.downsample_factor, :]
        x, _ = self.crn(x)
        x = x.view(x.size(0), -1, self.num_filter_frames, x.size(-2), x.size(-1))
        
        if hasattr(self, "mlp"):
            x = x.permute(0,2,3,4,1)
            x = x.reshape(*x.shape[:3], -1)
            x = self.mlp(x)
            x = x.reshape(*x.shape[:3], x.shape[3] // 3, 3)
            x = x.permute(0,4,1,2,3)

        # upsample again by repeating   
        x = torch.repeat_interleave(x, self.downsample_factor, dim=3)[..., :nfr, :]

        if self.method == "complex_filter":
            if self.output_mapping == "star":
                x = torch.einsum("bijkm,li->bljkm", x, self.v)
            elif self.output_mapping == "gated":
                # split into gate and gated part
                x_gate = x[:, 0, ...]
                x_gatee = torch.tanh(x[:, [1, 2], ...])
                gate = torch.sigmoid(x_gate)
                x = gate[:, None, :, :] * x_gatee
            elif self.output_mapping == "gatednorm":
                # split into gate and gated part
                x_gate = x[:, 0, ...]
                x_gatee = x[:, [1, 2], ...]
                x_gatee = x_gatee / torch.norm(x_gatee, dim=1, keepdim=True)
                gate = torch.sigmoid(x_gate)
                x = gate[:, None, :, :] * x_gatee
            return self._complex_filtering(x, x_complex, xlen)
        
        elif self.method == "complex_mapping":
            if self.output_mapping == "star":
                x = torch.einsum("bijkm,li->bljkm", x, self.v)
            elif self.output_mapping == "gated":
                # split into gate and gated part
                x_gate = x[:, 0, ...]
                x_gatee = x[:, [1, 2], ...]
                gate = torch.sigmoid(x_gate)
                x = gate[:, None, :, :] * x_gatee
            elif self.output_mapping == "gatednorm":
                raise ValueError(
                    "Gatednorm output mapping is not supported for complex mapping"
                )
            # remove number of filter frames dimension
            x = x.squeeze(2)
            x = torch.cat([x[:, 0, ...], x[:, 1, ...]], dim=-1).permute(0, 2, 1)
            return self._linear_inverse_transform(x, xlen)
        
        elif self.method == "time_domain_filtering":
            return self._time_domain_filtering(x, x_in, xlen)

    def _complex_filtering(self, fd_filt, x_complex, xlen):
        fd_filt_complex = torch.complex(fd_filt[:, 0, ...], fd_filt[:, 1, ...])
        x_complex = nn.functional.pad(x_complex, (0, 0, self.num_filter_frames - 1, 0))
        x_complex = x_complex.unfold(1, self.num_filter_frames, 1).permute(0, 3, 1, 2)
        # truncate to shorter length
        n_frames_shorter = min(x_complex.size(2), fd_filt_complex.size(2))
        x_complex = x_complex[:, :, :n_frames_shorter, :]
        fd_filt_complex = fd_filt_complex[:, :, :n_frames_shorter, :]
        
        
        y = torch.einsum("bijk,bijk->bjk", x_complex, fd_filt_complex)
        y = torch.cat([y.real, y.imag], dim=-1).permute(0, 2, 1)
        
        return self._linear_inverse_transform(y, xlen)
    
    def _linear_inverse_transform(self, x_complex_stacked, xlen):
        y = nn.functional.pad(
            nn.functional.conv_transpose1d(
                x_complex_stacked, self.inverse_transform, stride=self.hopsize
            ),
            (self.winlen - 2 * self.hopsize, 0),
        ).squeeze(1)
        lendiff = y.size(-1) - xlen
        if lendiff > 0:
            y = y[..., :-lendiff]
        elif lendiff < 0:
            y = torch.nn.functional.pad(y, (0, -lendiff))
        return y


    def _time_domain_filtering(self, fd_filt, x_in, xlen):
        fd_filt = fd_filt.permute(0, 2, 3, 1, 4)

        fd_filt = fd_filt.reshape(fd_filt.size(0), fd_filt.size(1), fd_filt.size(2), -1)
        
        # flatten to 2D
        #fd_filt_shape0, fd_filt_shape1, fd_filt_shape2 = fd_filt.size(0), fd_filt.size(1), fd_filt.size(2)
        #fd_filt = fd_filt.reshape(-1, fd_filt.shape[-1])
        filt = self.inverse_transform(fd_filt)
        #filt = filt.reshape(fd_filt_shape0, fd_filt_shape1, fd_filt_shape2, filt.shape[-1])

        x_frame = nn.functional.pad(
            x_in,
            (
                self.filtlen - self.algorithmic_delay_filtering,
                self.algorithmic_delay_filtering,
            ),
        ).unfold(1, 2 * self.hopsize + self.filtlen, self.hopsize)
        n_frames_shorter = min(x_frame.size(1), filt.size(2))
        x_frame = x_frame[
            :, :n_frames_shorter, :
        ]  # [batch, n_frames, 2 * hopsize + winlen - 1]

        x_frame = x_frame * self.analysis_window_filtering[None, None, :]

        filt = filt[
            :, :, :n_frames_shorter, :
        ]  # [batch, num_filter_frames, n_frames, winlen]

        x_frame = (
            nn.functional.pad(x_frame, (0, 0, self.num_filter_frames - 1, 0))
            .unfold(1, self.num_filter_frames, 1)
            .permute(0, 3, 1, 2)
        )  # [batch, num_filter_frames, n_frames, 2 * hopsize + winlen - 1]

        # now do the convolution using FFT
        x_fft = torch.fft.rfft(x_frame, n=2 * self.hopsize + self.filtlen, dim=-1)
        filt_fft = torch.fft.rfft(filt, n=2 * self.hopsize + self.filtlen, dim=-1)
        x_filt = torch.fft.irfft(
            torch.sum(x_fft * filt_fft, 1), n=2 * self.hopsize + self.filtlen, dim=-1
        )


        # use only valid part of the convolution
        x_filt = x_filt[
            :, :, -2 * self.hopsize:
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
        if y.shape[1] < xlen:
            y = torch.nn.functional.pad(y, (0, xlen - y.shape[1]))
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
