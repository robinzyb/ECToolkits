import matplotlib.pyplot as plt


def vac_scale(she_scale):
    return -she_scale-4.44


def she_scale(vac_scale):
    return -vac_scale-4.44


def be_data_to_she(ba_dict):
    new_dict = {}
    for name, be_data in ba_dict.items():
        new_dict[name] = {
            "vbm": she_scale(be_data["vbm"]),
            "cbm": she_scale(be_data["cbm"]),
        }
    return new_dict


def plot_one_ba_sc(ax, idx, vbm, cbm, width):
    ax.hlines(vbm, idx-width/2, idx+width/2, color='C0', linewidth=7)
    ax.hlines(cbm, idx-width/2, idx+width/2, color='C3', linewidth=7)


def plot_multiple_ba(ax, ba_dict):

    width = 0.3
    for idx, (name, be_data) in enumerate(ba_dict.items()):
        plot_one_ba_sc(
            ax, idx, vbm=be_data["vbm"], cbm=be_data["cbm"], width=width)


def plot_one_ba_diff(ax, idx, vbm, cbm, ref_vbm, ref_cbm):
    ax.annotate('', xy=(idx, ref_vbm), xytext=(idx, vbm),
                arrowprops=dict(arrowstyle='<->', color='C0'))
    ax.annotate('', xy=(idx, ref_cbm), xytext=(idx, cbm),
                arrowprops=dict(arrowstyle='<->', color='C3'))

    middle_point = (ref_vbm + vbm)/2
    diff = abs(ref_vbm - vbm)
    if abs(middle_point+0.1) < ref_vbm:
        ax.text(idx, middle_point+0.1, f"{diff:1.2f}", fontsize='medium')
    else:
        ax.text(idx, ref_vbm, f"{diff:1.2f}", fontsize='medium')

    middle_point = (ref_cbm + cbm)/2
    diff = abs(ref_cbm - cbm)

    if abs(middle_point+0.1) < ref_cbm:
        ax.text(idx, middle_point+0.1, f"{diff:1.2f}", fontsize='medium')
    else:
        ax.text(idx, ref_cbm, f"{diff:1.2f}", fontsize='medium')


def plot_multiple_ba_diff(ax, ba_dict, key_list, ref_key):
    ref_vbm = ba_dict[ref_key]['vbm']
    ref_cbm = ba_dict[ref_key]['cbm']
    for idx, (name, be_data) in enumerate(ba_dict.items()):
        if name in key_list:
            plot_one_ba_diff(
                ax, idx, be_data["vbm"], be_data["cbm"], ref_vbm=ref_vbm, ref_cbm=ref_cbm)


def plot_band_alignment(ba_dict, show_diff=False, vac_value=False):
    plt.rc('font', size=18)  # controls default text size
    plt.rc('axes', titlesize=23)  # fontsize of the title
    plt.rc('axes', labelsize=20)  # fontsize of the x and y labels
    plt.rc('xtick', labelsize=18)  # fontsize of the x tick labels
    plt.rc('ytick', labelsize=18)  # fontsize of the y tick labels
    plt.rc('legend', fontsize=16)  # fontsize of the legend

    plt.rc('lines', linewidth=2, markersize=10)  # controls default text size

    plt.rc('axes', linewidth=2)
    plt.rc('xtick.major', size=10, width=2)
    plt.rc('ytick.major', size=10, width=2)

    numb_sys = len(ba_dict)
    if vac_value:
        ba_dict = be_data_to_she(ba_dict)

    fig = plt.figure(figsize=(16, 9), dpi=200)
    ax = fig.add_subplot(111)
    plot_multiple_ba(ax, ba_dict=ba_dict)

    name_list = list(ba_dict)
    first_key = name_list[0]
    ax.axhline(ba_dict[first_key]["vbm"], linestyle="--", color='C0')
    ax.axhline(ba_dict[first_key]["cbm"], linestyle="--", color='C3')

    ax.set_ylabel("E(vs. Vac.) [eV]")
    ax.set_xticks(range(numb_sys))
    ax.set_xticklabels(name_list)
    ax.tick_params(direction='in')

    ax.invert_yaxis()
    y2 = ax.secondary_yaxis('right', functions=(vac_scale, she_scale))
    y2.set_ylabel("E(Vac.) [eV]")
    y2.tick_params(direction='in')
    ax.set_ylabel("U(SHE) [V]")

    if show_diff:
        plot_multiple_ba_diff(
            ax, ba_dict, key_list=name_list[1:], ref_key=first_key)

    return fig


def enumerate_subplots(fig):
    from matplotlib.offsetbox import AnchoredText
    import string
    for idx, ax in enumerate(fig.get_axes()):
        at = AnchoredText(f"({string.ascii_lowercase[idx]})",
                          loc='lower left', prop=dict(size=25), frameon=False,
                          bbox_to_anchor=(0., 1.),
                          bbox_transform=ax.transAxes
                          )
        # at.patch.set_boxstyle("round,pad=0.,rounding_size=0.2")
        ax.add_artist(at)
