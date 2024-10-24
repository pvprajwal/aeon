import scipy
import numpy as np
from aeon.distances._paa_sax_mindist import paa_sax_mindist


def test_mindist():
    X_paa = np.array([0.18305977, 0.01801501, -0.25552613, 0.05445141])

    Y_sax = np.array([6, 1, 11, 13])
    Y_paa = np.array([-0.2794701, -1.2409898, 0.53562105, 0.9848391])

    alphabet_size = 16
    i = 0
    j = 35

    #breakpoints = [-1.53412054, -1.15034938, -0.88714656, -0.67448975
    #               -0.48877641, -0.31863936, -0.15731068,  0.,
    #               0.15731068,  0.31863936,  0.48877641,  0.67448975,
    #               0.88714656,  1.15034938,  1.53412054]

    breakpoints = scipy.stats.norm.ppf(
        np.arange(1, alphabet_size, dtype=np.float64) / alphabet_size,
        scale=1.0,
    )
    length = 256

    min_dist = paa_sax_mindist(
        X_paa, Y_sax, breakpoints, length
    )

    ed = 13.05747

    if min_dist > ed:
        print(f"mindist is:\t {min_dist} but ED is: \t {ed}")
        assert min_dist <= ed

    print("Done")
