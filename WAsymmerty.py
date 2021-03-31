import numpy as np
import WPlotting
from WHistograms import lep_asym
import matplotlib.pyplot as plt
import mplhep as hep
from matplotlib.ticker import AutoMinorLocator


def calc_asym(pos_lep_eta, neg_lep_eta):
    asym_hist = []
    for pos_bin, neg_bin in zip(pos_lep_eta, neg_lep_eta):
        asym_bin = (pos_bin - neg_bin) / (pos_bin + neg_bin)
        asym_hist.append(asym_bin)
    asym_hist = np.array(asym_hist)
    return asym_hist


def plot_asym():
    stack_order = ["single top", "diboson", 'DrellYan', "ttbar", "Z", "W"]
    plot_label = "$W \\rightarrow l\\nu$"
    print("==========")
    print("Plotting Asymmetry")

    hist = lep_asym

    h_bin_width = hist["bin_width"]
    h_num_bins = hist["numbins"]
    h_xmin = hist["xmin"]
    h_xmax = hist["xmax"]
    h_xlabel = hist["xlabel"]
    h_title = hist["title"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    switch = int(input('0 for all leptons, 1 for electrons, 2 for muons\n'))
    if switch == 0:
        pos_name = 'pos_eta'
        neg_name = 'neg_eta'
        lep_type = 'all'
        pos_hist = {sample:
                        np.add(WPlotting.read_histogram('pos_ele_eta')[sample],
                               WPlotting.read_histogram('pos_mu_eta')[sample])
                    for sample in ["single top", "diboson", 'DrellYan', "ttbar", "Z", "W", 'data']}
        neg_hist = {sample:
                        np.add(WPlotting.read_histogram('neg_ele_eta')[sample],
                               WPlotting.read_histogram('neg_mu_eta')[sample])
                    for sample in ["single top", "diboson", 'DrellYan', "ttbar", "Z", "W", 'data']}
    elif switch == 1:
        lep_type = 'ele'
    elif switch == 2:
        lep_type = 'mu'
    else:
        raise ValueError('Choice is not in (0, 1, 2)')

    if switch in (1, 2):
        pos_name = f'pos_{lep_type}_eta'
        neg_name = f'neg_{lep_type}_eta'
        pos_hist = WPlotting.read_histogram(pos_name)
        neg_hist = WPlotting.read_histogram(neg_name)

    data_asym_heights = calc_asym(pos_hist['data'], neg_hist['data'])

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    plt.yscale("linear")
    main_axes = plt.gca()
    main_axes.set_title(h_title)

    main_axes.errorbar(x=bin_centers, y=data_asym_heights,
                       xerr=h_bin_width / 2,
                       fmt='ko', label='data')

    mc_tot_heights = {pos_name: np.zeros(len(bin_centers)), neg_name: np.zeros(len(bin_centers))}
    for hist, name in zip((pos_hist, neg_hist), (pos_name, neg_name)):
        for sample in stack_order:
            mc_heights = hist[sample]
            mc_tot_heights[name] = np.add(mc_tot_heights[name], mc_heights)
    mc_asym_heights = calc_asym(mc_tot_heights[pos_name], mc_tot_heights[neg_name])
    main_axes.bar(x=bin_centers, height=mc_asym_heights, width=h_bin_width,
                  color='lightblue', label='MC Simulation')


    handles, labels = main_axes.get_legend_handles_labels()
    main_axes.legend(reversed(handles), reversed(labels), title=plot_label, loc="upper right")
    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_xticklabels([])

    plt.axes([0.1, 0.1, 0.85, 0.2])
    plt.yscale("linear")
    ratio_axes = plt.gca()
    ratio_axes.errorbar(bin_centers, data_asym_heights / mc_asym_heights, xerr=h_bin_width / 2.,
                        fmt='.', color="black")
    ratio_axes.set_ylim(0.65, 1.35)
    ratio_axes.set_yticks([0.75, 1., 1.25])
    ratio_axes.set_xlim(h_xmin, h_xmax)
    ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel(f"Charge Asymmetry")
    ratio_axes.set_ylabel("Ratio\nData/MC")
    ratio_axes.set_xlabel(h_xlabel)
    plt.grid("True", axis="y", color="black", linestyle="--")

    plt.savefig(f'../Results_8TeV/asym_{lep_type}.jpeg')
    return None


plot_asym()