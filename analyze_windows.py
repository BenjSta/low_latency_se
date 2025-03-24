import torch
import matplotlib.pyplot as plt
from torch import hann_window
from torch import nn
import numpy as np

# Define parameters
fs_hz = 16000  # Sampling rate in Hz
signal_duration_s = 10  # Signal duration in seconds
n_samples = fs_hz * signal_duration_s  # Total number of samples

# Generate white noise
noise = torch.randn(n_samples)

# Function to compute Welch periodogram
def compute_welch_psd(signal, fs_hz, window_length_samples=3200):
    window = hann_window(window_length_samples, periodic=False)
    noverlap = window_length_samples // 2
    psd = torch.stft(signal, n_fft=window_length_samples, hop_length=noverlap, win_length=window_length_samples, window=window, return_complex=True)
    psd = psd.abs().pow(2).mean(dim=-1)
    psd_db = 10 * torch.log10(psd)  # Convert to dB scale
    frequencies_hz = torch.fft.fftfreq(window_length_samples, 1/fs_hz)[:window_length_samples//2]
    return frequencies_hz, psd_db[:window_length_samples//2]

# Function to create asymmetric Hann window pair
def asymmetric_hann_window_pair(analysis_winlen, synthesis_winlen):
    analysis_fade_in_len = analysis_winlen - synthesis_winlen // 2
    analysis_window = torch.cat(
        [
            torch.hann_window(analysis_fade_in_len * 2, periodic=True)[
                :analysis_fade_in_len
            ],
            torch.hann_window(synthesis_winlen, periodic=True)[synthesis_winlen // 2 :],
        ]
    )**0.5
    synthesis_window = torch.hann_window(synthesis_winlen, periodic=True) / torch.clip(
        analysis_window[len(analysis_window)-synthesis_winlen:], 1e-12
    )
    return analysis_window, synthesis_window

class DummyFilteringModel(torch.nn.Module):
    def __init__(self, hopsize, winlen, method, algorithmic_delay_nn=0, algorithmic_delay_filtering=0):
        super(DummyFilteringModel, self).__init__()

        self.winlen = winlen
        self.hopsize = hopsize
        self.num_filter_frames = 1

        if method == "complex_filter":
            analysis_window, synthesis_window = asymmetric_hann_window_pair(winlen, hopsize * 2)
        elif method == "time_domain_filtering":
            # use an asymmetric analysis window
            analysis_window, _ = asymmetric_hann_window_pair(winlen, algorithmic_delay_nn * 2)
            
            # use an asymmetric crossfade / synthesis window
            fade_in = torch.hann_window(2 * algorithmic_delay_filtering)[:algorithmic_delay_filtering]
            crossfade_win_first_half = torch.cat([
                fade_in, torch.ones(hopsize - algorithmic_delay_filtering)])
            crossfade_win_second_half = 1 - crossfade_win_first_half
            self.crossfade_window = nn.Parameter(
                torch.cat([crossfade_win_first_half, crossfade_win_second_half]),
                requires_grad=True,
            )

        stft_matrix = (
            torch.from_numpy(np.fft.rfft(np.eye(winlen))) * analysis_window[:, None]
        )
        self.forward_transform = nn.Parameter(
            (1 / np.sqrt(winlen))
            * torch.cat([stft_matrix.real.float(), stft_matrix.imag.float()], dim=1).T[
                :, None, :
            ],
            requires_grad=False,
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
                requires_grad=False,
            )
            self.algorithmic_delay_nn = 2 * hopsize
            self.algorithmic_delay_filtering = None
        elif method == "time_domain_filtering":
            # introduce a shift in the inverse transform
            assert algorithmic_delay_filtering < winlen // 2, "Algorithmic delay must be less than winlen // 2"

            fade_in_len = min(winlen // 4, algorithmic_delay_filtering)
            fade_in = torch.hann_window(2 * fade_in_len)[:fade_in_len]
            
            # use a symmetric fade_in / fade_out if algorithmic_delay is winlen // 2
            # otherwise use a fade_out longer than_fade_in, but at maximum fade_out_len = winlen // 3
            # use a zero taper of at maximum winlen // 6

            fade_out_len = max(int((1 - algorithmic_delay_filtering / (winlen / 2)) * winlen / 2), fade_in_len)
            zeros_len = 0
            #zeros_len = int((1 - algorithmic_delay_filtering / (winlen / 2)) * winlen / 6)
            
            end_tapering = torch.cat([
                torch.hann_window(2 * fade_out_len)[fade_out_len:],
                torch.zeros(zeros_len),
            ])
            
            window = torch.cat([
                fade_in,
                torch.ones(winlen - fade_in_len - len(end_tapering)),
                end_tapering
            ])

            self.inverse_transform = nn.Parameter(
                torch.roll(
                    torch.cat(
                    [istft_matrix_real.float(), istft_matrix_imag.float()], dim=0
                ), algorithmic_delay_filtering, dims=-1) * window[None, :],
                requires_grad=False,
            )
            self.algorithmic_delay_nn = algorithmic_delay_nn
            self.pad_size_nn = winlen - self.algorithmic_delay_nn
            self.algorithmic_delay_filtering = algorithmic_delay_filtering
            assert (
                self.pad_size_nn >= 0
            ), "Algorithmic delay must be less than or equal to winlen"

        self.method = method
        self.std_norm = nn.Parameter(
            torch.ones(winlen // 2 + 1), requires_grad=False
        )
        self.v = nn.Parameter(
            torch.tensor(
                [[1, -1 / 2, -1 / 2], [0, np.sqrt(3) / 2, -np.sqrt(3) / 2]],
                dtype=torch.float32,
            )
        )
        self.one_hot_filter = nn.Parameter(
            torch.zeros(2, 1, winlen // 2 + 1), requires_grad=False
        )
        self.one_hot_filter.data[0, 0, winlen // 4] = 1

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
        x = torch.stack([self.one_hot_filter] * x_complex.size(0), 0)
        x = torch.stack([x] * x_complex.size(1), 3)

        if self.method == "complex_filter":
            return self._complex_filtering(x, x_complex, xlen)
        elif self.method == "time_domain_filtering":
            return self._time_domain_filtering(x, x_in, xlen)

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

    def _time_domain_filtering(self, fd_filt, x_in, xlen):
        fd_filt = fd_filt.permute(0, 2, 3, 1, 4)
        fd_filt = fd_filt.reshape(fd_filt.size(0), fd_filt.size(1), fd_filt.size(2), -1)
        filt = fd_filt @ self.inverse_transform
        
        x_frame = nn.functional.pad(
            x_in,
            (self.winlen - self.algorithmic_delay_filtering, self.algorithmic_delay_filtering),
        ).unfold(1, 2 * self.hopsize + self.winlen - 1, self.hopsize)

        
        n_frames_shorter = min(x_frame.size(1), filt.size(2))
        x_frame = x_frame[:, :n_frames_shorter, :]  # [batch, n_frames, 2 * hopsize + winlen - 1]
        filt = filt[
            :, :, :n_frames_shorter, :
        ]  # [batch, num_filter_frames, n_frames, winlen]

        x_frame = (
            nn.functional.pad(x_frame, (0, 0, self.num_filter_frames - 1, 0))
            .unfold(1, self.num_filter_frames, 1)
            .permute(0, 3, 1, 2)
        )  # [batch, num_filter_frames, n_frames, 2 * hopsize + winlen - 1]

        x_fft = torch.fft.rfft(x_frame, n=2 * self.hopsize + self.winlen - 1, dim=-1)
        filt_fft = torch.fft.rfft(filt, n=2 * self.hopsize + self.winlen - 1, dim=-1)
        x_filt = torch.fft.irfft(
            torch.sum(x_fft * filt_fft, 1), n=2 * self.hopsize + self.winlen - 1, dim=-1
        )

        x_filt = x_filt[
            :, :, -2 * self.hopsize:
        ] # [batch, n_frames_shorter, 2 * hopsize]

        x_filt = x_filt * self.crossfade_window[None, None, :]

        y = (
            nn.functional.fold(
                x_filt.permute(0, 2, 1),
                output_size=(
                    1,
                    (n_frames_shorter - 1) * self.hopsize
                    + 2 * self.hopsize
                ),
                kernel_size=(1, 2 * self.hopsize),
                stride=(1, self.hopsize),
            )
            .squeeze(2)
            .squeeze(1)
        )
        return y[:, :xlen]
    
def plot_welch_periodograms(frequencies, psd_unfiltered, psd_filtered_dict):
    plt.figure(figsize=(10, 8))
    plt.plot(frequencies, psd_unfiltered.detach(), label='Unfiltered', alpha=0.7, color='black')
    for label, psd_filtered in psd_filtered_dict.items():
        if 'Time Domain' in label:
            linestyle = '-.'
            linewidth = 1.5
        else:
            linestyle = '--'
            linewidth = 2
        plt.plot(frequencies, psd_filtered.detach(), label=label, linestyle=linestyle, linewidth=linewidth)
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power Spectral Density (dB)')
    plt.title('Welch Periodogram (Filtered vs Unfiltered)')
    plt.ylim([-40, 30])
    plt.xlim([3200, 4800])
    plt.legend()
    plt.grid()
    plt.show()
    plt.savefig('Welch_Periodogram_Filtered_vs_Unfiltered.png')

def plot_individual_periodograms(frequencies, psd_unfiltered, psd_filtered_dict):
        delay_groups = {
            '20ms': [],
            '10ms': [],
            '2.5ms': [],
            '1.25ms': []
        }
        
        for label in psd_filtered_dict.keys():
            if '20ms' in label:
                delay_groups['20ms'].append(label)
            elif '10ms' in label:
                delay_groups['10ms'].append(label)
            elif '2.5ms' in label:
                delay_groups['2.5ms'].append(label)
            elif '1.25ms' in label:
                delay_groups['1.25ms'].append(label)
        
        for delay, labels in delay_groups.items():
            plt.figure(figsize=(10, 8))
            plt.plot(frequencies, psd_unfiltered.detach(), label='Unfiltered', alpha=0.7, color='black')
            for label in labels:
                psd_filtered = psd_filtered_dict[label]
                if 'Time Domain' in label:
                    linestyle = '-.'
                    linewidth = 1.5
                else:
                    linestyle = '--'
                    linewidth = 2
                plt.plot(frequencies, psd_filtered.detach(), label=label, linestyle=linestyle, linewidth=linewidth)
            plt.xlabel('Frequency (Hz)')
            plt.ylabel('Power Spectral Density (dB)')
            plt.title(f'Welch Periodogram ({delay} Delay)')
            plt.ylim([-40, 30])
            plt.xlim([2500, 5500])
            plt.legend()
            plt.grid()
            plt.show()
            plt.savefig(f'Welch_Periodogram_{delay.replace(" ", "_")}_Delay.png')

# Compute Welch periodogram for the original noise
frequencies_welch_hz, psd_welch_db = compute_welch_psd(noise, fs_hz)

# Perform filtering for different window sizes
filter_configs = [
    (160, 320, 'complex_filter', '20ms, 4000Hz'),
    (80, 160, 'complex_filter', '10ms, 4000Hz'),
    (80, 320, 'complex_filter', '10ms Asymmetric, 4000Hz'),
    (160, 320, 'time_domain_filtering', '10ms Time Domain, 4000Hz', 40, 40),
    (20, 320, 'complex_filter', '2.5ms Asymmetric, 4000Hz'),
    (160, 320, 'time_domain_filtering', '2.5ms Time Domain, 4000Hz', 10, 10),
    (10, 320, 'complex_filter', '1.25ms Asymmetric, 4000Hz'),
    (160, 320, 'time_domain_filtering', '1.25ms Time Domain, 4000Hz', 5, 5)
]

filtered_signals = {}
for config in filter_configs:
    hopsize, winlen, method, label = config[:4]
    algorithmic_delay_nn = config[4] if len(config) > 4 else 0
    algorithmic_delay_filtering = config[5] if len(config) > 5 else 0

    model = DummyFilteringModel(hopsize, winlen, method, algorithmic_delay_nn=algorithmic_delay_nn, algorithmic_delay_filtering=algorithmic_delay_filtering)
    filtered_signals[f'Filtered ({label})'] = model(noise.unsqueeze(0)).squeeze(0)

# Compute Welch periodogram for filtered signals
psd_filtered_dict = {}
for label, filtered_signal in filtered_signals.items():
    _, psd_filtered = compute_welch_psd(filtered_signal, fs_hz)
    psd_filtered_dict[label] = psd_filtered

# Plot Welch periodograms of all signals
plot_welch_periodograms(frequencies_welch_hz, psd_welch_db, psd_filtered_dict)
plot_individual_periodograms(frequencies_welch_hz, psd_welch_db, psd_filtered_dict)
