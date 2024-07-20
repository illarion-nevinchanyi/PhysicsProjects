# Frequency analysis of a one-dimensional data set
# TuGraz Computational physics - Assignment 2 Exercise 1
# Authors: Christoph Kircher, Gabriele Maschera, Illarion Nevinchanyi
# Date: 2023-11-06

from frequency_analysis_methods import *
import matplotlib.pyplot as plt
from datetime import datetime
from scipy.io.wavfile import write


SAMPLE_RATE = 44100  # Hz

def fourier_deviation(y):
    """
    Method for comparing the self written fourier transform with the internal fft from numpy

    :param y: the data set
    """

    print('Start testing the fourier transform for deviation from internal fft')

    Y_own = fourier_transform(y)
    Y_np = np.fft.fft(y)

    N = Y_own.shape[0]
    variance = np.sum(np.square(np.abs(Y_own - Y_np))) / N
    std_dev = np.sqrt(variance)
    print('The standard deviation between the self written and the internal fourier transform is {:E}'.format(std_dev))


def fourier_time(y):
    """
    Method for calculating the time used to calculate the fourier transform at different data set lengths

    :param y: the data set
    """

    print('Start testing the fourier transform for calculation time')

    num = 21
    M = np.logspace(1, 3, num)
    M = M.astype('int')
    time_values = np.zeros(num)

    for i in range(num):
        time_values[i] = datetime.now().timestamp()
        fourier_transform(y[range(M[i])])
        time_values[i] = datetime.now().timestamp() - time_values[i]
        print("  Finished m={:d}: {:d}/{:d}".format(M[i], i, num))

    # Convert time into ms
    time_values *= 1000

    plt.plot(M, time_values)
    plt.xlabel('Length of data set m')
    plt.ylabel('t / ms')
    plt.title('Calculation time t with different lengths m of the data set')
    plt.show()


def spectral_density(y):
    """
    Method for calculating the power spectral density of the given data set with the numpy fft

    :param y: the data set
    :return: power spectral density
    """

    print('Start of spectral density')

    S = np.square(np.abs(np.fft.fft(y, n=SAMPLE_RATE))) / SAMPLE_RATE**2

    # Crop the last values
    n = np.shape(S)[0]
    S = S[range(n)]
    f = np.linspace(0, n, n, endpoint=False)

    # Calculate peak
    peak_frequency = f[np.argmax(S)]

    # Plot spectral density
    # plt.semilogx(f, S, 'b-', label='spectral density')
    plt.loglog(f, S, 'b-', label='spectral density')
    plt.plot(peak_frequency, np.max(S), "ro", label='Peak at {:.2f} Hz'.format(peak_frequency))
    plt.xlabel('frequency / Hz')
    plt.ylabel('spectral density')
    plt.title('Power spectral density of the audio track')
    plt.legend(loc="upper left")
    plt.show()

    return peak_frequency


def perfect_tone(f=147, num_freq=5, time=5):
    """
    Method for calculating an audio file representing the spectral density of the perfect tone

    :param f: frequeny of the perfect tone
    :param num_freq: number of distinct frequencies
    :param time: seconds of generated wav file
    """

    print('Start of perfect tone generation')

    n = SAMPLE_RATE * time
    S = np.zeros(n)

    for i in range(1, num_freq + 1):
        S[int(f * time * i)] = 1

    y = np.fft.ifft(S)
    scaled = np.int16(np.real(y) / np.max(np.abs(y)) * 1e4)  # 32767
    write('perfect_tone.wav', SAMPLE_RATE, scaled)



data = np.loadtxt('single_tone.txt', 'complex')
y1 = data[range(data.shape[0]), 0]  # perform calculations only with first audio track
fourier_deviation(y1)
fourier_time(y1)
frequency = spectral_density(y1)
perfect_tone(frequency)
