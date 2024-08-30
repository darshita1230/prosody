import numpy as np
import matplotlib.pyplot as plt
import librosa
import librosa.display

from scipy.signal import lfilter, hamming
from scipy.linalg import toeplitz

from scipy.signal import hilbert
from scipy.signal import find_peaks

# Load the speech signal
filename = 'C:/Users/c darshita/Downloads/petti.wav'
signal, sr = librosa.load(filename, sr=None)

#generate lp residual

def preemphasis(signal, coeff=0.97):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])

def lpc(signal, order):
    R = np.correlate(signal, signal, mode='full')
    R = R[len(R) // 2:]
    R = R[:order + 1]
    R = toeplitz(R[:-1])
    r = R[:, 0]
    a = np.linalg.solve(R, r)
    return np.concatenate([[1], -a])

def residual_filter(prev_frame, frame, lpc_coeffs, order):
    return lfilter(lpc_coeffs, [1], frame)

def LP_residual(signal, framesize, frameshift, lporder, preempflag, a=0.97, plotflag=False):
    if preempflag == 0:
        prespeech = preemphasis(signal, a)
    else:
        prespeech = signal

    residual = np.zeros(len(signal))
    nframes = (len(signal) - framesize) // frameshift + 1
    LPCoeffs = np.zeros((nframes, lporder + 1))

    j = 0
    for i in range(0, len(signal) - framesize, frameshift):
        SpFrm = signal[i:i+framesize]
        preSpFrm = prespeech[i:i+framesize]

        lpcoef = lpc(hamming(len(preSpFrm)) * preSpFrm, lporder)
        LPCoeffs[j, :] = np.real(lpcoef)

        if i <= lporder:
            PrevFrm = np.zeros(lporder)
        else:
            PrevFrm = signal[i-lporder:i]

        ResFrm = residual_filter(np.real(PrevFrm), np.real(SpFrm), np.real(lpcoef), lporder)
        j += 1

        residual[i:i+frameshift] = ResFrm[:frameshift]

    last_frame_start = i + frameshift
    if last_frame_start < len(signal):
        SpFrm = signal[last_frame_start:]
        preSpFrm = prespeech[last_frame_start:]

        lpcoef = lpc(hamming(len(preSpFrm)) * preSpFrm, lporder)
        
        # Add one more row to LPCoeffs if required to accommodate the last frame
        if j >= LPCoeffs.shape[0]:
            LPCoeffs = np.vstack([LPCoeffs, np.zeros(lporder + 1)])
        
        LPCoeffs[j, :] = np.real(lpcoef)

        PrevFrm = signal[last_frame_start-lporder:last_frame_start]

        ResFrm = residual_filter(np.real(PrevFrm), np.real(SpFrm), np.real(lpcoef), lporder)
        residual[last_frame_start:last_frame_start+len(ResFrm)] = ResFrm[:len(ResFrm)]

    hm = hamming(2 * lporder)
    residual[:len(hm)//2] *= hm[:len(hm)//2]

    # if plotflag:
    #     import matplotlib.pyplot as plt
    #     plt.figure()
    #     plt.subplot(2, 1, 1)
    #     plt.plot(signal)
    #     plt.title('Original Signal')
    #     plt.grid()

    #     plt.subplot(2, 1, 2)
    #     plt.plot(np.real(residual)/max(np.abs(np.real(residual))))
    #     plt.title('LP Residual Signal')
    #     plt.grid()
    #     plt.show()

    return residual, LPCoeffs
# Parameters for LP residual calculation
framesize = int(0.025 * sr)  # 25 ms frame size
frameshift = int(0.010 * sr)  # 10 ms frame shift
lporder = 16  # Order of LPC

# Compute LP residual
residual, LPCoeffs = LP_residual(signal, framesize, frameshift, lporder, preempflag=0, plotflag=True)

#hilbert envelope
# Compute the Hilbert transform to get the analytical signal for the whole signal
analytical_signal = hilbert(residual)

# Calculate the envelope for the entire signal
imag = np.imag(analytical_signal)
envelope = np.abs(analytical_signal)

# Define your Gabor filter
def gabor_filter_1d_odd(sigma, omega, size):
    x = np.linspace(-size // 2, size // 2, size)
    odd_component = np.exp(-(x**2) / (2 * sigma**2)) * np.sin(omega * x)
    return odd_component

# Example Gabor filter parameters
sigma = 200
omega = 0.0057
##0.0114
size = 800

# Apply the Gabor filter
odd_filter = gabor_filter_1d_odd(sigma, omega, size)
filtered_signal = np.convolve(envelope, odd_filter, mode='same')

from scipy.ndimage import gaussian_filter1d
# Apply Gaussian smoothing 
vop_evidence = gaussian_filter1d(filtered_signal, sigma=150)
#vop_evidence = filtered_signal

from scipy.signal import find_peaks
import matplotlib.pyplot as plt
import numpy as np


# Define positive and negative thresholds
positive_threshold = 0.10 * np.max(vop_evidence)
negative_threshold = 0.20 * np.min(vop_evidence)

# Find peaks
peaks, _ = find_peaks(vop_evidence)
true_peaks = [peak for peak in peaks if vop_evidence[peak] > positive_threshold]

# Invert the signal to find valleys as peaks in the inverted signal
inverted_vop_evidence = -vop_evidence
valleys, _ = find_peaks(inverted_vop_evidence)
true_valleys = [valley for valley in valleys if inverted_vop_evidence[valley] > -negative_threshold]

# Find subsequent true peak for each true valley
subsequent_peaks = []
for valley in true_valleys:
    # Find the first true peak that occurs after the valley
    for peak in true_peaks:
        if peak > valley:
            subsequent_peaks.append(peak)
            break  # Exit the loop after finding the first subsequent peak



plt.figure(figsize=(14, 12))


plt.subplot(5, 1, 1)
plt.plot(signal, label='Original Signal')
#plt.plot(subsequent_peaks, signal[subsequent_peaks], 'ro')
plt.title('Original Signal')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


plt.subplot(5, 1, 2)
plt.plot(residual, label='LP Residual')
#plt.plot(subsequent_peaks, residual[subsequent_peaks], 'ro')
plt.title('LP Residual')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


plt.subplot(5, 1, 3)
plt.plot(envelope, label='Envelope')
#plt.plot(subsequent_peaks, envelope[subsequent_peaks], 'ro')
plt.title('Envelope')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


plt.subplot(5, 1, 4)
plt.plot(filtered_signal, label='Filtered Signal')
plt.plot(subsequent_peaks, filtered_signal[subsequent_peaks], 'ro')
plt.title('Filtered Signal ')
plt.xlabel('Sample Index')
plt.ylabel('Amplitude')
plt.legend()
plt.grid(True)


plt.subplot(5, 1, 5)
plt.plot(vop_evidence, label='VOP Evidence')
plt.plot(subsequent_peaks, vop_evidence[subsequent_peaks], 'ro')
plt.title('VOP Evidence')
plt.xlabel('Sample Index')
plt.ylabel('VOP Evidence Value')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# Optional: Visualize the valleys and their subsequent peaks
plt.figure(figsize=(12, 6))
plt.plot(vop_evidence, label='VOP Evidence')
plt.plot(true_peaks, vop_evidence[true_peaks], 'ro', label='True Peaks')
plt.plot(true_valleys, vop_evidence[true_valleys], 'bo', label='True Valleys')
plt.plot(subsequent_peaks, vop_evidence[subsequent_peaks], 'go', label='Subsequent Peaks')
plt.legend()
plt.title("VOP Evidence with True Valleys and Their Subsequent Peaks")
plt.xlabel("Sample Index")
plt.ylabel("VOP Evidence Value")
plt.grid(True)
plt.show()


