import numpy as np
import matplotlib.pyplot as plt
from models import asymmetric_hann_window_pair

# Define parameters
analysis_window_length = 256  # Analysis window length
synthesis_window_length = 128  # Synthesis window length
hop_size = 64  # Hop size
num_windows = 5  # Number of windows to visualize

# Generate asymmetric Hann windows
analysis_window, synthesis_window = asymmetric_hann_window_pair(
    analysis_window_length, synthesis_window_length
)

# Generate a reduced amplitude waveform
signal_length = (num_windows - 1) * hop_size + analysis_window_length
pseudo_signal = 0.5 * (np.sin(2 * np.pi * np.linspace(0, 5, signal_length)) + 0.2 * np.random.randn(signal_length))

# Mask the signal to appear only where there are windows
signal_mask = np.zeros(signal_length)
for i in range(num_windows):
    start = i * hop_size
    end = start + synthesis_window_length
    signal_mask[start:end] = 1
pseudo_signal *= signal_mask

# Time axis
n_signal = np.arange(signal_length)

# Create subplots
fig, (ax_analysis, ax_synthesis) = plt.subplots(2, 1, figsize=(10, 10), sharex=True)

# Plot pseudo signal on both axes
ax_analysis.plot(n_signal, pseudo_signal, label="Pseudo Signal", color='gray', alpha=0.5)
ax_synthesis.plot(n_signal, pseudo_signal, label="Pseudo Signal", color='gray', alpha=0.5)

# Plot analysis windows (end-aligned with synthesis windows)
for i in range(num_windows):
    end = (i + 1) * hop_size + synthesis_window_length
    start = end - analysis_window_length
    ax_analysis.plot(
        np.arange(start, end),
        analysis_window,
        label=f"Analysis Window {i + 1}" if i == 0 else None,
        color='red',
        alpha=0.7
    )

# Plot synthesis windows
for i in range(num_windows):
    end = (i + 1) * hop_size + synthesis_window_length
    start = end - synthesis_window_length
    ax_synthesis.plot(
        np.arange(start, end),
        synthesis_window,
        label=f"Synthesis Window {i + 1}" if i == 0 else None,
        color='blue',
        alpha=0.7
    )

# Add dashed vertical lines at hop size intervals
for i in range(num_windows + 1):
    x = i * hop_size
    ax_analysis.axvline(x=x, color='black', linestyle='--', alpha=0.5)
    ax_synthesis.axvline(x=x, color='black', linestyle='--', alpha=0.5)

# Customize axes
ax_analysis.set_ylabel("Amplitude")
ax_analysis.set_title("Analysis Windows (End-Aligned)")
ax_analysis.legend()
ax_analysis.grid()

ax_synthesis.set_xlabel("Samples")
ax_synthesis.set_ylabel("Amplitude")
ax_synthesis.set_title("Synthesis Windows")
ax_synthesis.legend()
ax_synthesis.grid()

# Save and show the plot
plt.tight_layout()
plt.savefig("windowing_visualization_end_aligned.png")
plt.show()
plt.close()