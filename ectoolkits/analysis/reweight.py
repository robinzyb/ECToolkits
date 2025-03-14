from string import Template
from typing import Tuple, Union

import numpy as np
import numpy.typing as npt

from ectoolkits.log import get_logger

# adapted from FES_from_Reweighting.py that written by Michele Invernizzi

logger = get_logger(__name__)

def read_cv(colvar_file: str,
            usecols: int,
            dim:int,
            ):

    if dim == 1 and len(usecols) == 1:
        cv = np.loadtxt(colvar_file, usecols=usecols)
        return cv
    elif dim == 2 and len(usecols) == 2:
        cv = np.loadtxt(colvar_file, usecols=usecols)
        cv_x = cv[:, 0]
        cv_y = cv[:, 1]
        return cv_x, cv_y

def read_bias(colvar_file: str,
              usecols: int=3,
              ) -> npt.NDArray:
    bias = np.loadtxt(colvar_file, usecols=usecols)
    return bias

def compute_bw_silverman(cv_array: npt.NDArray,
                         dim: int,
                         **kwargs,
                         ) -> float:
    """
    Compute bandwidths using Silverman's rule of thumb.

    Parameters:
    cv_array (npt.NDArray): The input array for which the bandwidth is to be computed.
    dim (int): The dimensionality of the data.
    **kwargs: Additional keyword arguments.
        - weights (optional, array-like): Weights for the data points.
        - bias_factor (optional, float): Bias factor to adjust the bandwidth. Default is 1.0.

    Returns:
    float: The computed bandwidth.

    Notes:
    - If weights are provided, the effective sample size (neff) is computed.
    - The bandwidth is computed using the formula:
      bw = sigma_0 / (sqrt(bias_factor)) * ((neff * (dim + 2) / 4.0) ** (-1.0 / (dim + 4.0)))
    - The function prints intermediate values for sigma_0, number of configurations (nconf),
      effective sample size (neff), and the computed bandwidth (bw).
    """
    weights = kwargs.get('weights', None)
    bias_factor = kwargs.get('bias_factor', 1.0)
    if weights is not None:
        neff = np.power(np.sum(weights), 2) / np.sum(np.power(weights, 2))
    else:
        neff = cv_array.shape[0]

    sigma_0 = cv_array.std()
    #sigma_0 = sigma_0/
    bw = sigma_0/(np.sqrt(bias_factor))*np.power((neff *(dim+2)/4.0), -1.0/(dim+4.0))
    logger.info(f"sigma_0 {sigma_0:.3f} nconf: {cv_array.shape[0]} neff: {neff:.3f} bw: {bw:.3f}")
    return bw

def read_fes(file_fes: str,
             dim: int,
             **kwargs,
            ) -> Union[Tuple[npt.NDArray, npt.NDArray],
                      Tuple[npt.NDArray, npt.NDArray, npt.NDArray]
                      ]:
    """
    Read the free energy surface (FES) data from PLUMED fes files.
    """

    if dim == 2:
        if 'num_grid_points_x' not in kwargs or 'num_grid_points_y' not in kwargs:
            raise ValueError("For 2D FES, 'num_grid_points_x' and 'num_grid_points_y' must be provided in kwargs.")
    num_grid_points_x = kwargs.get('num_grid_points_x', None)
    num_grid_points_y = kwargs.get('num_grid_points_y', None)
    if dim == 2:
        print("Reading 2D FES")
        X = np.loadtxt(file_fes, usecols=0)
        Y = np.loadtxt(file_fes, usecols=1)
        Z = np.loadtxt(file_fes, usecols=2)
        X = X.reshape(num_grid_points_x, num_grid_points_y)
        Y = Y.reshape(num_grid_points_x, num_grid_points_y)
        Z = Z.reshape(num_grid_points_x, num_grid_points_y)
        Z = Z*0.0103636  # to eV
        Z_min = np.min(Z)
        Z = Z - Z_min
        return X,Y,Z
    elif dim == 1:
        print("Reading 1D FES")
        x = np.loadtxt(file_fes, usecols=0)
        z = np.loadtxt(file_fes, usecols=1)
        z = z*0.0103636  # to eV
        z_min = np.min(z)
        z = z - z_min
        return x, z
    else:
        raise ValueError("dim should be either 1 or 2")

def make_grid(grid_bin_cv_x: int,
              cv_x_min: float,
              cv_x_max: float,
              dim: int,
              **kwargs
              ) -> Union[npt.NDArray, Tuple[npt.NDArray, npt.NDArray]]:
    """
    Generates a grid of coordinates based on the specified parameters.
    Parameters:
    grid_bin_cv_x (int): Number of bins for the x-axis.
    grid_bin_cv_y (int): Number of bins for the y-axis.
    cv_x_min (float): Minimum value for the x-axis.
    cv_x_max (float): Maximum value for the x-axis.
    cv_y_min (float): Minimum value for the y-axis.
    cv_y_max (float): Maximum value for the y-axis.
    Returns:
    tuple: one 1D array representing the x coordinates of the grid.
    tuple: Two 2D arrays representing the x and y coordinates of the grid.
    """
    grid_bin_cv_x += 1
    gird_cv_x_min = cv_x_min
    gird_cv_x_max = cv_x_max
    grid_cv_x = np.linspace(gird_cv_x_min, gird_cv_x_max, grid_bin_cv_x)
    if dim == 1:
        return grid_cv_x
    elif dim == 2:
        gird_cv_y_min = kwargs.get("cv_y_min", None)
        gird_cv_y_max = kwargs.get("cv_y_max", None)
        grid_bin_cv_y = kwargs.get("grid_bin_cv_y", None)
        grid_bin_cv_y += 1
        grid_cv_y = np.linspace(gird_cv_y_min, gird_cv_y_max, grid_bin_cv_y)
        X,Y=np.meshgrid(grid_cv_x,grid_cv_y)
        return X, Y


def calc_one_FES_point(cv_x: npt.NDArray,
                       bias: npt.NDArray,
                       point_x: float,
                       sigma_x: float,
                       kbT: float,
                       dim: int,
                       **kwargs
                       ) -> float:
    """
    Calculate the free energy surface (FES) value for a given point.

    Parameters:
    - cv_x: float, x-coordinate of the collective variable (CV)
    - cv_y: float, y-coordinate of the collective variable (CV)
    - bias: float, bias potential value
    - point_x: float, x-coordinate of the point
    - point_y: float, y-coordinate of the point
    - sigma_x: float, standard deviation of the x-coordinate
    - sigma_y: float, standard deviation of the y-coordinate
    - kbT: float, product of Boltzmann constant and temperature

    Returns:
    - fes: float, free energy surface value for the given point
    """
    cv_y = kwargs.get("cv_y", None)
    point_y = kwargs.get("point_y", None)
    sigma_y = kwargs.get("sigma_y", None)

    bias = bias / kbT

    dist_x = (point_x - cv_x) / sigma_x
    if dim == 1:
        arg = bias - 0.5*dist_x*dist_x
    elif dim == 2:
        dist_y = (point_y - cv_y) / sigma_y
        arg = bias - 0.5*dist_x*dist_x - 0.5*dist_y*dist_y
    else:
        raise ValueError("dim should be either 1 or 2")

    fes = -kbT*np.logaddexp.reduce(arg)

    return fes

#TODO: add documentation in jupyter book
def calc_FES(cv_x: npt.NDArray,
             bandwidth_x: float,
             nbins_x: int,
             bias: npt.NDArray,
             kbT: float,
             dim: int=2,
             **kwargs,
             ) -> npt.NDArray:
    """
    Calculate the Free Energy Surface (FES) using the provided collective variables (CVs) and bias.

    Parameters:
    cv_x (npt.NDArray): Collective variable values along the x-axis.
    bandwidth_x (float): Bandwidth for the Gaussian kernel along the x-axis.
    nbins_x (int): Number of bins along the x-axis for the grid.
    bias (npt.NDArray): Bias values corresponding to the CVs. must be divied by kbT to make it dimensionless before passing to this function.
    kbT (float): Thermal energy (Boltzmann constant times temperature).
    dim (int, optional): Dimensionality of the FES calculation (1 or 2). Default is 2.
    **kwargs: Additional keyword arguments:
        - cv_y (npt.NDArray, optional): Collective variable values along the y-axis (for 2D FES).
        - bandwidth_y (float, optional): Bandwidth for the Gaussian kernel along the y-axis (for 2D FES).
        - nbins_y (int, optional): Number of bins along the y-axis for the grid (for 2D FES).
        - min_cv_x (float, optional): Minimum value of the CV along the x-axis.
        - max_cv_x (float, optional): Maximum value of the CV along the x-axis.
        - min_cv_y (float, optional): Minimum value of the CV along the y-axis (for 2D FES).
        - max_cv_y (float, optional): Maximum value of the CV along the y-axis (for 2D FES).
        - save_file (bool, optional): Whether to save the FES data to a file. Default is True.
        - name_file (str, optional): Filename to save the FES data. Default is "fes.dat".
        - name_cv_x (str, optional): Name of the CV along the x-axis. Default is "cv_x".
        - name_cv_y (str, optional): Name of the CV along the y-axis. Default is "cv_y".

    Returns:
    np.ndarray: Calculated FES values on the grid.
    """

    # Extract the keyword arguments
    save_file = kwargs.get("save_file", True)
    name_file = kwargs.get("name_file", "fes.dat")

    name_cv_x = kwargs.get("name_cv_x", "cv_x")

    min_cv_x = kwargs.get("min_cv_x", cv_x.min())
    max_cv_x = kwargs.get("max_cv_x", cv_x.max())

    # the syntax is ugly.. any better way to do this?
    template_header_1d = \
"""Free energy surface generated by ECToolkits
FIELDS ${name_cv_x} () free_energy (kJ/mol)
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}"""

    template_header_2d = \
"""Free energy surface generated by ECToolkits
FIELDS ${name_cv_x} () ${name_cv_y} () free_energy (kJ/mol)
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}
SET min_${name_cv_y} ${min_cv_y}
SET max_${name_cv_y} ${max_cv_y}
SET num_grid_points_${name_cv_y} ${num_grid_points_y}"""

    if dim == 1:
        logger.info("Calculating 1D FES")

        fes_x = make_grid(grid_bin_cv_x=nbins_x,
                          cv_x_min=min_cv_x,
                          cv_x_max=max_cv_x,
                          dim=dim
                          )
        # Note the num_grid_points_x are nbins_x+1
        num_grid_points_x = len(fes_x)
        fes = np.zeros(num_grid_points_x)

        for i in range(num_grid_points_x):
            logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
            fes[i] = calc_one_FES_point(cv_x=cv_x,
                                        bias=bias,
                                        point_x=fes_x[i],
                                        sigma_x=bandwidth_x,
                                        kbT=kbT,
                                        dim=dim)

        if save_file:
            header = Template(template_header_1d).substitute(name_cv_x=name_cv_x,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             )

            np.savetxt(name_file,
                       np.array([fes_x.flatten(), fes.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )
        return fes

    elif dim == 2:
        logger.info("Calculating 2D FES")

        name_cv_y = kwargs.get("name_cv_y", "cv_y")
        cv_y = kwargs.get("cv_y", None)
        bandwidth_y = kwargs.get("bandwidth_y", None)
        nbins_y = kwargs.get("nbins_y", None)
        min_cv_y = kwargs.get("min_cv_y", cv_y.min())
        max_cv_y = kwargs.get("max_cv_y", cv_y.max())

        fes_x, fes_y = make_grid(grid_bin_cv_x=nbins_x,
                                grid_bin_cv_y=nbins_y,
                                cv_x_min=min_cv_x,
                                cv_x_max=max_cv_x,
                                cv_y_min=min_cv_y,
                                cv_y_max=max_cv_y,
                                dim=dim,
                                )

        # Note the num_grid_points_x and num_grid_points_y are nbins_x+1 and nbins_y+1
        num_grid_points_x = len(fes_x)
        num_grid_points_y = len(fes_y)
        fes = np.zeros((num_grid_points_x, num_grid_points_y))

        for i in range(num_grid_points_x):
            logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
            for j in range(num_grid_points_y):
                fes[i, j] = calc_one_FES_point(cv_x=cv_x,
                                               cv_y=cv_y,
                                               bias=bias,
                                               point_x=fes_x[i, j],
                                               point_y=fes_y[i, j],
                                               sigma_x=bandwidth_x,
                                               sigma_y=bandwidth_y,
                                               kbT=kbT,
                                               dim=dim,
                                               )


        if save_file:
            header = Template(template_header_2d).substitute(name_cv_x=name_cv_x,
                                                             name_cv_y=name_cv_y,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             min_cv_y=f"{min_cv_y:10.3f}",
                                                             max_cv_y=f"{max_cv_y:10.3f}",
                                                             num_grid_points_y=num_grid_points_y,
                                                             )
            np.savetxt(name_file,
                       np.array([fes_x.flatten(), fes_y.flatten(), fes.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )
        return fes

    else:
        raise ValueError("dim should be either 1 or 2")


#TODO: add pytest for this function
def calc_one_property_point(cv_x: npt.NDArray,
                            bias: npt.NDArray,
                            point_x: float,
                            bandwidth_x: float,
                            kbT: float,
                            dim: int,
                            prop: npt.NDArray,
                            **kwargs,
                            ) -> float:


    cv_y = kwargs.get("cv_y", None)
    point_y = kwargs.get("point_y", None)
    bandwidth_y = kwargs.get("bandwidth_y", None)

    bias = bias/kbT

    dist_x = (point_x - cv_x) / bandwidth_x
    if dim == 1:
        arg = bias - 0.5*dist_x*dist_x
    elif dim == 2:
        dist_y = (point_y - cv_y) / bandwidth_y
        arg = bias - 0.5*dist_x*dist_x - 0.5*dist_y*dist_y

    arg = np.exp(arg)
    #fes = -kbT*np.log(np.sum(np.exp(arg)))
    #TODO: log proper then logaddexp?
    denominator = np.sum(arg)
    numerator = np.sum(prop*arg)
    reweighted_prop = numerator / denominator

    return reweighted_prop



def calc_property_surface(cv_x: npt.NDArray,
                          bias: npt.NDArray,
                          prop: npt.NDArray,
                          bandwidth_x: float,
                          kbT: float,
                          nbins_x: int,
                          dim: int = 2,
                          **kwargs,
                          ):
    """
    Calculates the property surface based on the given input parameters.

    Parameters:
    """
    save_file = kwargs.get("save_file", True)
    name_file = kwargs.get("name_file", "prop_surface.dat")

    name_cv_x = kwargs.get("name_cv_x", "cv_x")

    min_cv_x = kwargs.get("min_cv_x", cv_x.min())
    max_cv_x = kwargs.get("max_cv_x", cv_x.max())

    # the syntax is ugly.. any better way to do this?
    template_header_1d = \
"""Free energy surface generated by ECToolkits
FIELDS ${name_cv_x} () Property ()
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}"""

    template_header_2d = \
"""Free energy surface generated by ECToolkits
FIELDS ${name_cv_x} () ${name_cv_y} () Property ()
SET min_${name_cv_x} ${min_cv_x}
SET max_${name_cv_x} ${max_cv_x}
SET num_grid_points_${name_cv_x} ${num_grid_points_x}
SET min_${name_cv_y} ${min_cv_y}
SET max_${name_cv_y} ${max_cv_y}
SET num_grid_points_${name_cv_y} ${num_grid_points_y}"""

    if dim == 1:
        logger.info("Calculating 1D property surface")

        fes_x = make_grid(grid_bin_cv_x=nbins_x,
                          cv_x_min=min_cv_x,
                          cv_x_max=max_cv_x,
                          dim=dim
                          )
        num_grid_points_x = len(fes_x)
        prop_surface = np.zeros(num_grid_points_x)

        for i in range(num_grid_points_x):
            logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
            prop_surface[i] = calc_one_property_point(cv_x=cv_x,
                                                      bias=bias,
                                                      point_x=fes_x[i],
                                                      bandwidth_x=bandwidth_x,
                                                      kbT=kbT,
                                                      dim=dim,
                                                      prop=prop,
                                                      )

        if save_file:
            header = Template(template_header_1d).substitute(name_cv_x=name_cv_x,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             )

            np.savetxt(name_file,
                       np.array([fes_x.flatten(), prop_surface.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )

        return prop_surface

    elif dim == 2:
        logger.info("Calculating 2D property surface")

        name_cv_y = kwargs.get("name_cv_y", "cv_y")
        cv_y = kwargs.get("cv_y", None)
        bandwidth_y = kwargs.get("bandwidth_y", None)
        nbins_y = kwargs.get("nbins_y", None)
        min_cv_y = kwargs.get("min_cv_y", cv_y.min())
        max_cv_y = kwargs.get("max_cv_y", cv_y.max())

        fes_x, fes_y = make_grid(grid_bin_cv_x=nbins_x,
                                grid_bin_cv_y=nbins_y,
                                cv_x_min=min_cv_x,
                                cv_x_max=max_cv_x,
                                cv_y_min=min_cv_y,
                                cv_y_max=max_cv_y,
                                dim=dim,
                                )

        # Note the num_grid_points_x and num_grid_points_y are nbins_x+1 and nbins_y+1
        num_grid_points_x = len(fes_x)
        num_grid_points_y = len(fes_y)

        prop_surface = np.zeros((num_grid_points_x, num_grid_points_y))


        for i in range(num_grid_points_x):
            logger.info(f'   working...  {(i/num_grid_points_x):.0%}')
            for j in range(num_grid_points_y):
                prop_surface[i, j] = calc_one_property_point(cv_x=cv_x,
                                                             cv_y=cv_y,
                                                             bias=bias,
                                                             point_x=fes_x[i, j],
                                                             point_y=fes_y[i, j],
                                                             bandwidth_x=bandwidth_x,
                                                             bandwidth_y=bandwidth_y,
                                                             kbT=kbT,
                                                             dim=dim,
                                                             prop=prop,
                                                             )

        if save_file:
            header = Template(template_header_2d).substitute(name_cv_x=name_cv_x,
                                                             name_cv_y=name_cv_y,
                                                             min_cv_x=f"{min_cv_x:10.3f}",
                                                             max_cv_x=f"{max_cv_x:10.3f}",
                                                             num_grid_points_x=num_grid_points_x,
                                                             min_cv_y=f"{min_cv_y:10.3f}",
                                                             max_cv_y=f"{max_cv_y:10.3f}",
                                                             num_grid_points_y=num_grid_points_y,
                                                             )
            np.savetxt(name_file,
                       np.array([fes_x.flatten(), fes_y.flatten(), prop_surface.flatten()]).T,
                       fmt="%.6f",
                       header=header,
                       )

        return prop_surface

    else:
        raise ValueError("dim should be either 1 or 2")