import numpy as np
import matplotlib.pyplot as plt
from models import asymmetric_hann_window_pair

# Parameters
analysis_window_length = 256
synthesis_window_length = 128
hop_size = 64
num_windows = 5

# Generate asymmetric Hann windows
analysis_window, synthesis_window = asymmetric_hann_window_pair(
    analysis_window_length, synthesis_window_length
)

# Also generate standard Hann window for synthesis
hann_synthesis_window = np.hanning(synthesis_window_length)

# Generate dummy signal
signal_length = (num_windows - 1) * hop_size + analysis_window_length
dummy_signal = 0.5 * (np.sin(2 * np.pi * np.linspace(0, 5, signal_length)) + 0.2 * np.random.randn(signal_length))

# Create signal masks
signal_mask1 = np.ones(signal_length) * np.nan
signal_mask2 = np.ones(signal_length) * np.nan
for i in range(num_windows):
    end = i * hop_size + analysis_window_length
    if i == 0:
        signal_mask1[end-analysis_window_length:end-synthesis_window_length//2] = analysis_window[:-synthesis_window_length//2]
        signal_mask1[end-synthesis_window_length//2:end] = 1
        signal_mask2[end-synthesis_window_length:end-synthesis_window_length//2] = synthesis_window[:-synthesis_window_length//2]
        signal_mask2[end-synthesis_window_length//2:end] = 1
    elif i == num_windows - 1:
        signal_mask1[end-analysis_window_length:end-synthesis_window_length//2] = 1
        signal_mask1[end-synthesis_window_length//2:end] = analysis_window[-synthesis_window_length//2:]
        signal_mask2[end-synthesis_window_length:end-synthesis_window_length//2] = 1
        signal_mask2[end-synthesis_window_length//2:end] = synthesis_window[-synthesis_window_length//2:]
    else:
        signal_mask1[end-analysis_window_length:end] = 1
        signal_mask2[end-synthesis_window_length:end] = 1

n_signal = np.arange(signal_length)

# === Synthesis Plot with standard Hann window ===
fig, ax_synth_hann = plt.subplots(figsize=(10, 3))
ax_synth_hann.plot(n_signal, dummy_signal * signal_mask2, color='gray', alpha=0.5)

for i in range(num_windows):
    end = i * hop_size + analysis_window_length
    start = end - synthesis_window_length
    ax_synth_hann.plot(np.arange(start, end), hann_synthesis_window, color='blue', alpha=0.7)

for i in range(num_windows + 4):
    x = i * hop_size
    ax_synth_hann.axvline(x=x, color='black', linestyle='--', alpha=0.5)

ax_synth_hann.axhline(y=0, color='black', linewidth=0.8)

# Cleanup
ax_synth_hann.set_xticks([])
ax_synth_hann.set_yticks([])
ax_synth_hann.set_title("")
ax_synth_hann.set_xlabel("")
ax_synth_hann.set_ylabel("")
ax_synth_hann.legend().set_visible(False)
ax_synth_hann.grid(False)
ax_synth_hann.set_frame_on(False)
for spine in ax_synth_hann.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("synthesis_hann_window.svg", format='svg')
plt.close()

# === Analysis Plot with masked signal only (no windows) ===
fig, ax_analysis_mask = plt.subplots(figsize=(10, 3))
ax_analysis_mask.plot(n_signal, dummy_signal * signal_mask1, color='gray', alpha=0.5)

for i in range(num_windows + 4):
    x = i * hop_size
    ax_analysis_mask.axvline(x=x, color='black', linestyle='--', alpha=0.5)

ax_analysis_mask.axhline(y=0, color='black', linewidth=0.8)

# Cleanup
ax_analysis_mask.set_xticks([])
ax_analysis_mask.set_yticks([])
ax_analysis_mask.set_title("")
ax_analysis_mask.set_xlabel("")
ax_analysis_mask.set_ylabel("")
ax_analysis_mask.legend().set_visible(False)
ax_analysis_mask.grid(False)
ax_analysis_mask.set_frame_on(False)
for spine in ax_analysis_mask.spines.values():
    spine.set_visible(False)

plt.tight_layout()
plt.savefig("analysis_mask_only.svg", format='svg')
plt.close()
