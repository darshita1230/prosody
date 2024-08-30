import librosa
import numpy as np
from scipy.signal import lfilter, hilbert
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks

# Load the audio file
audio_path = 'C:/Users/c darshita/Desktop/audios/audio6.wav'
y, sr = librosa.load(audio_path, sr=None)


order = 16

lpc_coeffs = librosa.lpc(y=y, order=order)

lpc_filter = np.concatenate([[1], -lpc_coeffs[1:]])

residual = lfilter(lpc_filter, 1, y)


# Compute the analytic signal
analytic_signal = hilbert(residual)

# Compute the Hilbert envelope
hilbert_envelope = np.abs(analytic_signal)

# Define the Gabor filter
def gabor_filter(t, f, sigma):
    return np.exp(-t**2 / (2 * sigma**2)) * np.cos(2 * np.pi * f * t)

# Parameters for the Gabor filter
t = np.linspace(-0.1, 0.1, int(0.2 * sr))  # time vector
f = 50  # frequency of the sinusoidal wave (in Hz)
sigma = 0.005  # standard deviation of the Gaussian envelope


# Generate the Gabor filter
gabor = gabor_filter(t, f, sigma)

# Apply the Gabor filter
filtered_envelope = np.zeros_like(hilbert_envelope)
filter_length = len(gabor)
half_filter_length = filter_length // 2

# Padding the Hilbert envelope for convolution
padded_envelope = np.pad(hilbert_envelope, (half_filter_length, half_filter_length), 'constant')

for i in range(len(hilbert_envelope)):
    window = padded_envelope[i:i + filter_length]
    filtered_envelope[i] = np.sum(window * gabor)

# Apply Gaussian smoothing 
smoothed_envelope = gaussian_filter1d(filtered_envelope, sigma=110)


# Find peaks
vop_evidence = smoothed_envelope
peaks, _ = find_peaks(vop_evidence)
positive_threshold = 0.30 * np.max(vop_evidence)

# Filter peaks based on the threshold
true_peaks = [peak for peak in peaks if vop_evidence[peak] > positive_threshold]

# Cluster peaks
min_distance = 5000  # Minimum distance between VOPs in samples
clustered_peaks = []
current_cluster = [true_peaks[0]]

for peak in true_peaks[1:]:
    if peak - current_cluster[-1] <= min_distance:
        current_cluster.append(peak)
    else:
        highest_peak = max(current_cluster, key=lambda x: smoothed_envelope[x])
        clustered_peaks.append(highest_peak)
        current_cluster = [peak]

# Add the highest peak of the last cluster
if current_cluster:
    highest_peak = max(current_cluster, key=lambda x: smoothed_envelope[x])
    clustered_peaks.append(highest_peak)

# Convert sample indices to time
time = np.arange(len(y)) / sr
clustered_peaks_time = np.array(clustered_peaks) / sr

# vop_start_times to be plotted with red lines
#6
vop_start_times = np.array([1.239771, 1.470171, 1.711543, 2.144914, 2.435657, 2.627657, 2.929371, 3.302400, 3.423086, 3.724800])
#8vop_start_times = np.array([1.244873, 1.535684, 1.862208, 2.112203, 2.428523, 2.530562, 2.693824, 2.989737, 3.234630])
#4vop_start_times = np.array([1.344113, 1.628444, 1.824892, 1.964473, 2.310840, 2.465930, 2.832976, 3.262059])
#5vop_start_times = np.array([1.131276, 1.687162, 2.145524, 2.389333, 2.545371, 3.032990, 3.228038])
#1vop_start_times = np.array([1.428317, 1.711407, 2.114596, 2.556388, 2.702222])

#PRINT TIMES
print(f"DETECTED VOPS: {clustered_peaks_time}")
print(f"MARKED VOPS: {vop_start_times}")


# Plotting the results
plt.figure(figsize=(15, 10))

# Plot the original signal
plt.subplot(5, 1, 1)
plt.plot(time, y, label='Original Signal')
plt.scatter(clustered_peaks_time, y[clustered_peaks] + 0.8 * max(y), color='red', marker='x', label='Detected VOPs')

for vop_time in vop_start_times:
    plt.axvline(x=vop_time, color='k', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the LP residual
plt.subplot(5, 1, 2)
plt.plot(time, residual, label='LP Residual')
plt.scatter(clustered_peaks_time, residual[clustered_peaks] + 0.8 * max(residual), color='red', marker='x')

for vop_time in vop_start_times:
    plt.axvline(x=vop_time, color='k', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the Hilbert envelope
plt.subplot(5, 1, 3)
hilbert_envelope_time = np.arange(len(hilbert_envelope)) / sr
plt.plot(hilbert_envelope_time, hilbert_envelope, label='Hilbert Envelope')
plt.scatter(clustered_peaks_time, hilbert_envelope[clustered_peaks] +  0.8 * max(hilbert_envelope), color='red', marker='x')

for vop_time in vop_start_times:
    plt.axvline(x=vop_time, color='k', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

# Plot the smoothed envelope
plt.subplot(5, 1, 4)
plt.plot(hilbert_envelope_time, smoothed_envelope, label='Smoothed Envelope')
plt.scatter(clustered_peaks_time, smoothed_envelope[clustered_peaks] + 0.8 * max(smoothed_envelope), color='red', marker='x')

for vop_time in vop_start_times:
    plt.axvline(x=vop_time, color='k', linestyle='-')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()

true_peaks_time = np.array(true_peaks) / sr
# Plot the smoothed envelope
plt.subplot(5, 1, 5)
plt.plot(time, smoothed_envelope, label='Smoothed Envelope', linestyle='--')
plt.plot(true_peaks_time, smoothed_envelope[true_peaks], 'go', color='green', label='True Peaks')
plt.plot(clustered_peaks_time, smoothed_envelope[clustered_peaks], 'go', color='red', label='Clustered Peaks')
plt.xlabel('Time (s)')
plt.ylabel('Amplitude')
plt.legend()
plt.show()


plt.tight_layout()
plt.show()
