import numpy as np
import doctest


def fft_analysis(t, x, pos='n'):
    """
    FFT_ANALYSIS computes the fft of the signal

    :param t: time array. t.shape = (N,)
    :param x: signal array. x.shape = (N,)
    :param pos: 'y' return spectrum only for positive frequencies (default 'n')
    :return:
      * 'f': frequency array (e.g. for N = 8 ==> f = array([DC, f1, f2, f3, f4, -f3, -f2, -f1])
      * 'X': DFT of x
      * 'fp': frequency array (e.g. for N = 8 ==> fp = array([DC, f1, f2, f3, f4])
      * 'Xp': spectrum in the positive range (doubled X values for all positive frequencies but DC and Nyquist frequency)

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 03.03.2016
    HISTORY:
    04/05/2016 more pythonic way to handle optional input for positive spectrum
    04/05/2016 slice index are forced to be integer when selecting the positive spectrum

    """

    # force A to be a column vector
    dim = x.ndim
    if dim == 1:
        x = np.reshape(x,(x.size, -1))
    if dim > 1:
        (r, c) = x.shape
        if r < c:
            x = x.transpose()

    # determine general parameters
    N = t.size
    Ts = t[1] - t[0]
    T = N * Ts
    df = 1/T

    fnq = N / 2 / T           # Get Nyquist frequency
    f = np.arange(N) * df     # compute frequency (also > fn)
    f[f > fnq] -= fnq * 2     # shift to [positive-negative] range
    # Almost Equivalent to:
    # f = np.fft.fftfreq(t.size, d=Ts)
    # "np.fft.fftfreq" when t.size is even, it is all equal but the Nyquist frequency that is
    # given as a negative value

    X = np.fft.fft(x,axis=0)/N

    if pos == 'y':
        fp = f[f >= 0]
        Xp = X[f >= 0]
        Xp[1:int(np.floor((N-1)/2))] = 2*Xp[1:int(np.floor((N-1)/2))]
        return f, X, fp, Xp
    else:
        return f, X


def inverse_fft(X):
    """
    INVERSE_FFT computes the time domain signal from the spectrum

    :param X: spectrum with positive and negative frequencies
    :return:
        * x: inverse fourier transform of X

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 03.03.2016
    HISTORY:

    """
    import warnings
    toll = 1e-7
    x = np.fft.ifft(X * X.size, axis=0)

    if (x.imag > toll).any():
        warnings.warn("Specturm is NOT complex conjugate.\nPlease verify input")
    return x.real


def sinx_x_interp(t, x, K):
    """

    :param t: time array. t.shape = (N,)
    :param x: signal array. x.shape = (N,)
    :param K: number of points of the signal interpolated
    :return:
        * ti: new time array. ti.shape = (K,)
        * xi: singnal interpolated. xi.shape = (K,)

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 05.03.2016
    HISTORY:

    """

    T = t.size * (t[1] - t[0])

    # Compute spectrum
    f, X = fft_analysis(t, x)
    df = f[1] - f[0]

    # Build new frequency array
    fnq = K / 2 / T  # Get Nyquist frequency
    fi = np.arange(K) * df  # compute frequency (also > fn)
    fi[fi > fnq] -= fnq * 2  # shift to [positive-negative] range

    # Build new time array
    dT = T / K
    ti = np.linspace(0, T - dT, K)

    # Build new spectrum
    Xi = np.zeros(fi.shape, dtype=complex)
    cnt = 0
    toll = df*1e-3
    for freq in f:
        Xi[(fi > (freq - toll)) & (fi < (freq + toll))] = X[cnt]
        cnt += 1

    # IFFT
    xi = inverse_fft(Xi)

    return ti, xi


def adjust_spectrum(X):
    """
    ADJUST_SPECTRUM imposes the symmetries of spectrum related to a real-values signal

    :param X: spectrum of a real-valued signal to be adjusted
    :return:
        Xa: adjusted spectrum

    AUTHOR: Luca Giaccone (luca.giaccone@polito.it)
    DATE: 05.03.2016
    HISTORY:

    EXAMPLE:
    >>> X = np.array([0, 1+1j, 2+1e-10j, 1-0.99j])
    >>> # element 1 and 3 and non complex-conjugate. Element 2 in not perfectly real
    >>> Xa = adjust_spectrum(X)
    >>> Xa
    array([ 0.+0.j   ,  1.+0.995j,  2.+0.j   ,  1.-0.995j])
    >>> # element 1, 3 and 2 fixed
    """

    # local copy of X
    Xa = np.copy(X)
    N = X.size

    # index of all positive frequencies but Nyquist freq.
    ipos = range(1,int(np.ceil(N/2)))
    # index of all negative frequencies
    ineg = sorted(range(int(np.floor(N / 2) + 1), N), reverse=True)

    # impose spectrum symmetry ==> X[fj] = X[-fj].conj()
    Xa[ineg] = .5 * (X[ipos].conj() + X[ineg])
    Xa[ipos] = .5 * (X[ipos] + X[ineg].conj())

    # impose real values for DC and Nyquist frequency (only if N is even)
    if np.mod(N, 2):
        # N is odd
        Xa[0] = X[0].real
    else:
        # N is even
        Xa[0] = X[0].real
        Xa[N/2] = X[N/2].real

    return Xa


if __name__ == '__main__':
    import matplotlib.pylab as plt

    # Parameters
    f = 50      # fundamental frequency
    T = 1/f     # period
    npt = 14     # number of points
    NT = 2      # number of periods
    A1 = 10
    A3 = 2

    # build signal
    dT = NT*T/npt
    t = np.linspace(0,NT*T - dT,npt)
    x = A1 * np.sin(2*np.pi*f*t) + A3 * np.sin(2*np.pi*3*50*t - np.pi / 2)

    # Compute spectrum
    f, X = fft_analysis(t, x)

    # sin(x)/x interpolation
    K = 100
    ti, xi = sinx_x_interp(t, x, K)

    xtrue = A1 * np.sin(2 * np.pi * 50 * ti) + A3 * np.sin(2 * np.pi * 3 * 50 * ti - np.pi/2)


    # Plot signal and interpolation
    fg1 = plt.figure(facecolor='w')
    plt.plot(t, x,'k-o', linewidth=2, markerfacecolor='w',markeredgewidth=2,markersize=8,label='signal')
    plt.plot(ti, xi, 'b', linewidth=2, label=r'$\frac{\sinx}{x}$ interp.')
    plt.plot(ti, xtrue, 'r--', linewidth=2, label='true')
    plt.xlabel('time (s)', fontsize=14)
    plt.ylabel('signal (s)', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.legend(fontsize=18)
    plt.grid()
    plt.tight_layout()
    plt.draw()

    # Plot spectrum
    fg2 = plt.figure(facecolor='w')
    (markerLines, stemLines, baseLines) = plt.stem(f, np.abs(X), '-')
    plt.setp(stemLines, color='b', linewidth=2)
    plt.setp(markerLines, markerfacecolor='w',markersize=10,markeredgecolor='b',markeredgewidth=2)
    plt.xlabel('freq (Hz)', fontsize=14)
    plt.ylabel('magnitude', fontsize=14)
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=14)
    plt.xlim(np.min(f) - f[1] - f[0],np.max(f) + f[1] - f[0])
    plt.grid()
    plt.tight_layout()
    plt.show()

    # enable doctestmod
    doctest.testmod()