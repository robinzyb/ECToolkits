# this module collect acidity calculation functions
#
from cp2kdata.units import *
import numpy as np


# see reference

# quantum correction term Delta A qc
def get_quantum_correction_hydronium(T=298):

    H3O_frequencies_list = np.array([1032, 1668, 1692, 3449, 3523, 3540])
    H2O_frequencies_list = np.array([1610, 3710, 3820])

    quant_corr_hydronium = get_quantum_correction(
        H3O_frequencies_list, T=T) - get_quantum_correction(H2O_frequencies_list, T=298)

    return quant_corr_hydronium


def get_quantum_correction(frequencies_list, T):
    # frequencies in cm-1
    """get quantum correction acidity calculation

    _extended_summary_
    """
    quant_vib_fe = get_quant_vib_fe(frequencies_list=frequencies_list)
    cls_vib_fe = get_cls_vib_fe(frequencies_list=frequencies_list, T=T)
    quant_corr = quant_vib_fe - cls_vib_fe
    return quant_corr


def get_quant_vib_fe(frequencies_list):

    quant_vib_fe = np.sum(frequencies_list) * 0.5 * WaveNumber2eV
    return quant_vib_fe


def get_cls_vib_fe(frequencies_list, T):

    cls_vib_fe = -kB * T * \
        np.log(np.prod((kB*T) / (frequencies_list * WaveNumber2eV)))
    return cls_vib_fe

# dummy insertion free energy


def get_dummy_insert_fe(frequencies_list, T):
    fAd = get_partition_ratio(frequencies_list, T=T)
    Delta_A_Ad = _get_dummy_insert_fe(fAd, T)
    return Delta_A_Ad


def get_dummy_insert_fe_hydronium(T=298):
    fH2Od = get_partition_ratio_hydronium(T=T)
    Delta_A_H2Od = _get_dummy_insert_fe(fH2Od, T=T)
    return Delta_A_H2Od


def _get_dummy_insert_fe(fAd, T):
    """get dummy insertion free energy
     \Delta A_{Ad} = -kBT ln(c0 \Lambda_{\ce{H+}}^{3} f_{Ad})
    _extended_summary_

    Args:
        frequencies_list (_type_): _description_
        T (_type_): _description_
    """
    c0 = NAvo * 1.0e-27  # 1 mol dm-3, or 1 per Ang-3
    Lambda_H = 1.01  # at 298K
    Lambda_Hpow3 = np.power(Lambda_H, 3)

    Delta_A_Ad = -kB * T * np.log(c0 * Lambda_Hpow3 * fAd)
    return Delta_A_Ad


def get_partition_ratio(frequencies_list, T=298):
    prod_theta_vib = np.prod(get_vib_temp(frequencies_list))
    Tpow3 = np.power(T, 3)
    fAd = Tpow3 / prod_theta_vib
    return fAd


def get_partition_ratio_hydronium(T=298):
    # moment of inertia (I) in atomic unit
    # frequency of a vibration mode in cm-1

    I_H2O = np.array([4150, 7740, 11890])
    freq_H2O = np.array([1610, 3650, 3750])
    q_H2O = get_gas_partition(freq_H2O, I_H2O, T=T, sigma=2)

    I_H2Od = np.array([10430, 10710, 15780])
    freq_H2Od = np.array([1450, 1810, 1920, 3340, 3660, 3760])
    q_H2Od = get_gas_partition(freq_H2Od, I_H2Od, T=T, sigma=1)

    fH2Od = q_H2Od/q_H2O

    return fH2Od


def get_gas_partition(frequencies_list, I_list, T, sigma):
    """only used for H2Od and H2O gas partition function.
    In high temperature limit

    _extended_summary_

    Args:
        frequencies_list (_type_): _description_
        I_list (_type_): _description_
        T (_type_): _description_
        sigma (_type_): _description_

    Returns:
        _type_: _description_
    """
    pi_sqrt = np.sqrt(np.pi)

    n = len(frequencies_list)
    Tpown = np.power(T, n)
    prod_theta_vib = np.prod(get_vib_temp(frequencies_list))

    Tpow3 = np.power(T, 3)
    prod_theta_rot = np.prod(get_rot_temp(I_list))

    constant = pi_sqrt / sigma
    partition = Tpown / prod_theta_vib * \
        constant * np.sqrt(Tpow3 / prod_theta_rot)

    return partition


def get_vib_temp(frequency):
    """frequency in cm-1

    _extended_summary_

    Args:
        frequency (_type_): _description_

    Returns:
        _type_: _description_
    """
    vib_temp = frequency * WaveNumber2eV / kB
    return vib_temp


def get_rot_temp(I):
    """moment of inertia (I) in atomic unit

    see ref eq
    B = hbar^2 / 2I
    theta_rot = B/kB
    Args:
        I_list (_type_): _description_
    """
    hbar_au = hbar / au2J / au2s
    B = np.power(hbar_au, 2) * 0.5 / I
    kB_au = kB / au2eV
    theta_rot = B / kB_au
    return theta_rot
