import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import group_delay, minimum_phase
from scipy.fftpack import fft, ifft, fftfreq

def design_filters(N, center, bandwidth, fs):
    # Frequency axis
    freqs = np.linspace(0, fs, N+1, endpoint=True)
    #omega = 2 * np.pi * freqs
    freqs = freqs[:N//2]

    # Gaussian magnitude response
    mag = np.exp(-((freqs - center)**2/(2 * (bandwidth / (2 * np.sqrt(np.log(2))))**2)))

    plt.figure()
    plt.plot(freqs, mag, label='Magnitude Response')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Magnitude')
    plt.title('Magnitude Response of Filter')
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("magnitude_response_plot.png")
    h = np.fft.irfft(mag, n=N+1)
    h = np.roll(h, N//2)  # Center the impulse response
    h_fir_min = minimum_phase(h, half=False)
    
    return h_fir_min

def measure_group_delay_at_center(h, fs, center_freq):
    # Compute group delay

    # hfft = np.fft.rfft(h, n=len(h))
    # freq = np.fft.rfftfreq(len(h), d=1/fs)
    # hphase = np.angle(hfft)
    # hphase_unwrapped = np.unwrap(hphase)
    # hphasedelay = hphase_unwrapped / (2 * np.pi * freq)

    w, gd = group_delay((h, 1), fs=fs)
    # w = freq
    # gd = hphasedelay


    # plt.figure()
    # plt.plot(w, gd, label='Group Delay')
    # plt.xlabel('Frequency (Hz)')
    # plt.ylabel('Group Delay (samples)')
    # plt.title('Group Delay of Filter')
    # plt.grid(True)
    # plt.legend()
    # plt.tight_layout()
    # plt.show()
    # plt.savefig("group_delay_plot.png")

    # Find closest frequency to center_freq
    idx = np.argmin(np.abs(w - center_freq))
    return gd[idx]

def sweep_bandwidths(N, center, bandwidths, fs):
    gd_min_list = []
    for bw in bandwidths:
        h_fir_min = design_filters(N, center, bw, fs)
        gd_min = measure_group_delay_at_center(h_fir_min, fs, center)
        gd_min_list.append(gd_min)
    return gd_min_list

if __name__ == "__main__":
    # Parameters
    N = 320
    center = 2000
    fs = 16000

    # Sweep bandwidths
    bandwidths = np.linspace(0.005, 0.1, 50) * fs
    gd_min_list = sweep_bandwidths(N, center, bandwidths, fs)

    # Plot group delay vs bandwidth
    plt.figure(figsize=(6, 4))
    plt.plot(bandwidths, gd_min_list, marker='o')
    plt.title("Minimum-phase Group Delay at Center Frequency")
    plt.xlabel("Bandwidth in Hz")
    plt.xlim(1600, 0)
    plt.ylabel("Group Delay / samples")
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    plt.savefig("group_delay_vs_bandwidth.png")
