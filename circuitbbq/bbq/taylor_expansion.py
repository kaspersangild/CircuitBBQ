import numpy as np


def taylor_coeffs(func, ndim, order=6):
    points1d = ndim * [np.exp(-1j * np.linspace(0, 2 * np.pi, endpoint=False, num=16))]
    points = np.meshgrid(*points1d, indexing="ij")
    f = func(*points)
    slices = tuple(slice(order + 1) for _ in np.arange(self.ndim))
    return np.fft.ifftn(f)[slices]
