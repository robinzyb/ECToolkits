import numpy as np

def birch_murnaghan_equation(V, V0, E0, B0, B0_prime):
    V_ratio = np.power(np.divide(V0, V), np.divide(2, 3))
    E = E0 + np.divide((9 * V0 * B0), 16) * (np.power(V_ratio - 1, 3) * B0_prime
            + np.power((V_ratio -1), 2) * (6 - 4 * V_ratio))
    return E
