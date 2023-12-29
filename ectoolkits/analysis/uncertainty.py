import numpy as np
from numpy.typing import ArrayLike
from typing import Tuple


def get_f_coarse_grained_data(data_list: ArrayLike, tau: int) -> ArrayLike:
    """
    coarse grain array data in forward direction with block length tau.

    Args:
    -----------
        data_list (ArrayLike):
            Time Serials Data, shape(1,N), like energies.
        tau (int):
            block length

    Returns:
    -----------
        ArrayLike:
            Coarse-grained Time Serials Data, shape(1, int(N/tau)).

    Notes:
    -----------
     see https://doi.org/10.1063/1.1638996

    Examples:
    -----------
    >>> get_f_coarse_grained_data(energies_list, 50)


    """
    new_data_list = []
    n = len(data_list)
    l = int(n/tau)
    for i in range(l):
        Y_i = data_list[i*tau:i*tau+tau].mean()
        new_data_list.append(Y_i)
    new_data_list = np.array(new_data_list)
    return new_data_list


def get_uncertainty(data_list: ArrayLike, tau_range: Tuple[int, int, int] = (1, 50, 1)) -> Tuple[ArrayLike, ArrayLike]:
    """
    calculate uncertainty for a time serial data, like energies.

    Args:
    -----------
        data_list (ArrayLike):
            Time Serials Data, shape(1,N), like energies.
        tau_range (Tuple[int, int, int], optional):
            A list of tau values. Defaults to (1, 50, 1).

    Returns:
    -----------
        Tuple[ArrayLike, ArrayLike]:
            1. a list of mean values corresponding to tau
            2. a list of uncertainties corresponding to tau

    Notes:
    -----------
     see https://doi.org/10.1063/1.1638996

    Examples:
    -----------
    >>>  vgap
    array([17.21097182, 17.21777723, 17.22483708, ..., 17.72987759,
       17.76217987, 17.80458071])
    >>>  vgap.shape
    (13401,)
    >>> get_uncertainty(vgap, tau_range=(1, 50, 1))
    (array([17.72412774, 17.72412174, 17.72412774, 17.72412174, 17.72412174,
        17.72411847, 17.72411847, 17.72412174, 17.72412774, 17.72412174,
        17.72411847, 17.72411469, 17.72409511, 17.72411847, 17.72412228,
        17.72411469, 17.72412114, 17.72411469, 17.72412228, 17.72412174,
        17.72411847, 17.72411847, 17.72401629, 17.72411469, 17.72412174,
        17.72409511, 17.72411469, 17.72395954, 17.72411847, 17.72383193,
        17.72411469, 17.72373849, 17.72411847, 17.72412114, 17.72374005,
        17.72411469, 17.72412206, 17.72373849, 17.72375512, 17.72412174,
        17.72376983, 17.72411847, 17.72372078, 17.72373849, 17.72377165,
        17.72401629, 17.72412228, 17.72411469, 17.72375512]),
    array([0.00521475, 0.00736884, 0.00901134, 0.01038585, 0.01158367,
        0.01265499, 0.01362589, 0.014516  , 0.01533808, 0.01611023,
        0.01683463, 0.01752622, 0.01819687, 0.01873488, 0.01935648,
        0.01993202, 0.02041762, 0.02085271, 0.0213809 , 0.02189274,
        0.02239711, 0.02284273, 0.02327067, 0.02372226, 0.02401315,
        0.0244621 , 0.02488177, 0.02532788, 0.02569048, 0.02586628,
        0.0262883 , 0.02674231, 0.02682957, 0.02712625, 0.02778647,
        0.02759716, 0.02792354, 0.02817088, 0.02851189, 0.02871313,
        0.02902685, 0.0294121 , 0.02958297, 0.02987098, 0.0297415 ,
        0.02995385, 0.03045183, 0.03049436, 0.03114382]))
    """
    err_list = []
    mean_list = []
    for tau in range(*tau_range):
        coarse_grained_data_list = get_f_coarse_grained_data(data_list, tau)
        err = np.sqrt(np.var(coarse_grained_data_list) /
                      len(coarse_grained_data_list))
        mean = coarse_grained_data_list.mean()
        err_list.append(err)
        mean_list.append(mean)
    err_list = np.array(err_list)
    mean_list = np.array(mean_list)
    return mean_list, err_list
