from typing import Tuple, Union, List
import numpy.typing as npt
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import cp2kdata.plots.colormaps


def get_mae(data: npt.NDArray[np.float64],
            ref: npt.NDArray[np.float64]
            ) -> float:
    return np.mean(np.abs(data - ref))

def get_rmse(data: npt.NDArray[np.float64],
            ref: npt.NDArray[np.float64]
            ) -> float:
    return np.sqrt(np.mean((data - ref)**2))

def get_dptest_data(e_file:str,
                    f_file:str
                    ) -> Tuple[npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               npt.NDArray[np.float64],
                               int
                               ]:
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



def plot_error(ax,
               data: npt.NDArray,
               pred: npt.NDArray,
               type: str='energy',
               title: str='Energy',
               return_err: bool=False
               ) -> Union[None, Tuple[float, float]]:


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

def plot_dptest(e_file: str,
                f_file: str,
                save_name: str="dptest.png",
                return_err: bool=False,
                frc_comp: bool=False
                ):

    plt.style.use("cp2kdata.matplotlibstyle.jcp")
    if frc_comp:
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
    else:
        row = 1
        col = 2
        fig = plt.figure(figsize=(3.37*col, 2.6*row), dpi=200, facecolor='white')
        gs = fig.add_gridspec(row,col, hspace=0.5)
        e_data, e_pred, fx_data, fy_data, fz_data, fx_pred, fy_pred, fz_pred, natom = \
            get_dptest_data(e_file=e_file, f_file=f_file)

        f_data = np.array([fx_data, fy_data, fz_data])
        f_pred = np.array([fx_pred, fy_pred, fz_pred])

        ax = fig.add_subplot(gs[0])
        rmse_e, mae_e = plot_error(ax, e_data/natom, e_pred/natom, type='energy', title='Energy', return_err=True)
        ax  = fig.add_subplot(gs[1])
        rmse_f, mae_f = plot_error(ax, f_data, f_pred, type='force', title=r'Force', return_err=True)


    if save_name:
        fig.savefig(save_name, dpi=300)

    if return_err:
        if frc_comp:
            return fig, rmse_e, mae_e, rmse_fx, mae_fx, rmse_fy, mae_fy, rmse_fz, mae_fz
        else:
            return fig, rmse_e, mae_e, rmse_f, mae_f

    else:
        return fig


## model deviation
def _hist_and_stat_model_devi(model_devi_files: List[List[str]],
                              trust_lo: float,
                              trust_hi: float,
                              **kwargs
                              ) -> Tuple[List[Tuple[npt.NDArray, npt.NDArray]],
                                         List[List[float]]
                                         ]:
    hist_model_devi = []
    stat_model_devi = []
    for _iter_files in  model_devi_files:
        _model_devi = None
        for _model_devi_file in _iter_files:
            if _model_devi is None:
                _model_devi = np.loadtxt(_model_devi_file, usecols=4)
            else:
                _model_devi = np.concatenate((_model_devi, np.loadtxt(_model_devi_file, usecols=4)))

        ntotal = len(_model_devi)
        naccurate = np.sum((_model_devi < trust_lo))
        ncandidate = np.sum((_model_devi > trust_lo) &(_model_devi < trust_hi))
        npoor = np.sum((_model_devi > trust_hi))

        ratio_accurate = naccurate / ntotal
        ratio_candidate = ncandidate / ntotal
        ratio_poor = npoor / ntotal
        stat_model_devi.append([ratio_accurate, ratio_candidate, ratio_poor])

        hist, bin_edges = np.histogram(_model_devi, **kwargs)
        hist_model_devi.append([hist, bin_edges])
    return hist_model_devi, stat_model_devi

def plot_model_devi(model_devi_files: List[List[str]],
                    trust_lo: float,
                    trust_hi: float,
                    save_name: str="model_devi.png",
                    **kwargs
                    ):

    hist_model_devi, stat_model_devi = _hist_and_stat_model_devi(model_devi_files, trust_lo, trust_hi, **kwargs)

    # get the maximum value of histograms.
    max_hist = 0
    for _hist, _bin_edges in hist_model_devi:
        if _hist.max() > max_hist:
            max_hist = _hist.max()

    plt.style.use('cp2kdata.matplotlibstyle.jcp')
    niters = len(hist_model_devi)
    cp2kdata_cb_lscmap = mpl.colormaps['cp2kdata_cb_lscmap']
    plt.rcParams["axes.prop_cycle"] = plt.cycler("color", cp2kdata_cb_lscmap(np.linspace(0, 1, niters)))

    row = niters
    col = 1
    fig = plt.figure(figsize=(3.37*col, 1.00*row), dpi=300, facecolor="white")
    gs = fig.add_gridspec(row, col, hspace=0.0)
    shifts = np.linspace(0, 10, niters)

    for i_hist_model_devi, (_hist, _bin_edges) in enumerate(hist_model_devi):
        ax = fig.add_subplot(gs[i_hist_model_devi])
        ax.fill(_bin_edges[:-1], _hist, label=f"iter-{i_hist_model_devi:03d}", alpha=0.5, color=f"C{i_hist_model_devi}")
        ax.axvline(trust_lo, color="black", linestyle="--", label="trust_lo")
        ax.axvline(trust_hi, color="black", linestyle="--", label="trust_hi")
        name = "Accurate:"
        percent = stat_model_devi[i_hist_model_devi][0] * 100
        percent = f"{percent:4.2f}"
        percent = percent.zfill(5)  # I must turn a float into a string to pad zeros.
        ax.text(1.35, 0.3, f"{name:>10} {percent}%", transform=ax.transAxes, ha="right")
        name = "Candidate:"
        percent = stat_model_devi[i_hist_model_devi][1] * 100
        percent = f"{percent:4.2f}"
        percent = percent.zfill(5)
        ax.text(1.35, 0.2, f"{name:>10} {percent}%", transform=ax.transAxes, ha="right")
        name = "Poor:"
        percent = stat_model_devi[i_hist_model_devi][2] * 100
        percent = f"{percent:4.2f}"
        percent = percent.zfill(5)
        ax.text(1.35, 0.1, f"{name:>10} {percent}%", transform=ax.transAxes, ha="right")
        ax.set_ylim(-0.05, max_hist + 0.05)
        ax.legend(bbox_to_anchor=(1.37, 1.00))
    # I prefer fig.supxlabel to ax.set_xlabel, but the y position of fig.supxlabel is scaled with the figure size.
    ax.set_xlabel("Model deviation " + r"(eV/$\mathrm{\AA}$)")
    fig.supylabel("Density")
    if save_name:
        fig.savefig(save_name, bbox_inches="tight")
    return fig
