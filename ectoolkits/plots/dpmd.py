import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt


def get_mae(data: npt.NDArray[np.float64],
            ref: npt.NDArray[np.float64]
            ) -> float:
    return np.mean(np.abs(data - ref))

def get_rmse(data: npt.NDArray[np.float64],
            ref: npt.NDArray[np.float64]
            ) -> float:
    return np.sqrt(np.mean((data - ref)**2))

def get_dptest_data(e_file, f_file):
    e_data, e_pred = np.loadtxt(e_file, unpack=True)
    e_mean = np.mean(e_data)
    e_data -= e_mean
    e_pred -= e_mean
    nframe = len(e_data)

    fx_data, fy_data, fz_data, fx_pred, fy_pred, fz_pred = np.loadtxt(f_file, unpack=True)
    fx_data = fx_data.reshape((nframe, -1))
    fy_data = fy_data.reshape((nframe, -1))
    fz_data = fz_data.reshape((nframe, -1))
    fx_pred = fx_pred.reshape((nframe, -1))
    fy_pred = fy_pred.reshape((nframe, -1))
    fz_pred = fz_pred.reshape((nframe, -1))

    natom = fx_data.shape[1]
    return e_data, e_pred, fx_data, fy_data, fz_data, fx_pred, fy_pred, fz_pred, natom



def plot_error(ax, data, pred, type='energy', title='Energy', return_err=False):


    ax.scatter(data, pred, s=0.5)

    xlim = ax.get_xlim()

    ax.plot([xlim[0], xlim[1]], [xlim[0], xlim[1]], color='red')

    rmse = get_rmse(data, pred)
    rmse = rmse * 1000

    mae = get_mae(data, pred)
    mae = mae * 1000


    ax.set_title(title)
    if type == 'force':
        ax.set_xlabel(r'Forces (eV/$\mathrm{\AA}$)')
        ax.set_ylabel(r'Forces (eV/$\mathrm{\AA}$)')
        unit=r'meV/$\mathrm{\AA}$'
    elif type == 'energy':
        ax.set_xlabel('Energies (eV/atom)')
        ax.set_ylabel('Energies (eV/atom)')
        unit='meV/atom'


    ax.text(0.1, 0.8, f"RMSE: {rmse:.3f} {unit} ", transform=ax.transAxes)
    ax.text(0.1, 0.9, f"MAE: {mae:.3f} {unit} ", transform=ax.transAxes)
    if return_err:
        return rmse, mae

def plot_dptest(e_file: str, f_file: str, save_name: str="dptest.png", return_err=False):
    plt.style.use("cp2kdata.matplotlibstyle.jcp")
    row = 2
    col = 2
    fig = plt.figure(figsize=(3.37*col, 2.6*row), dpi=200, facecolor='white')
    gs = fig.add_gridspec(row,col, hspace=0.5)
    e_data, e_pred, fx_data, fy_data, fz_data, fx_pred, fy_pred, fz_pred, natom = \
        get_dptest_data(e_file=e_file, f_file=f_file)

    ax = fig.add_subplot(gs[0])
    rmse_e, mae_e = plot_error(ax, e_data/natom, e_pred/natom, type='energy', title='Energy', return_err=True)
    ax  = fig.add_subplot(gs[1])
    rmse_fx, mae_fx = plot_error(ax, fx_data, fx_pred, type='force', title=r'$\mathrm{F_{x}}$', return_err=True)
    ax = fig.add_subplot(gs[2])
    rmse_fy, mae_fy = plot_error(ax, fy_data, fy_pred, type='force', title=r'$\mathrm{F_{y}}$', return_err=True)
    ax = fig.add_subplot(gs[3])
    rmse_fz, mae_fz = plot_error(ax, fz_data, fz_pred, type='force', title=r'$\mathrm{F_{z}}$', return_err=True)

    fig.savefig(save_name, dpi=400)

    if return_err:
        return rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz


