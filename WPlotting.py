import matplotlib.pyplot as plt
import numpy as np
import pandas
import uproot3
import uproot

import infofile
import WSamples
import mplhep as hep
from WHistograms import hist_dicts

from matplotlib.ticker import AutoMinorLocator


def read_histogram_from_file(hist_name, path):
    file = uproot3.open(path)
    histogram = file[hist_name]
    return histogram.values


def read_histogram_from_sample(hist_name, sample):
    tot_heights = None
    for val in WSamples.samples[sample]["list"]:
        path = f'../Output_8TeV/{val}.root'
        if tot_heights is None:
            tot_heights = read_histogram_from_file(hist_name, path)
        else:
            tot_heights = np.add(tot_heights, read_histogram_from_file(hist_name, path))
    return tot_heights


def read_histogram(hist_name):
    histogram = {}
    for sample in ["data", "diboson", "ttbar", "Z", "single top", "W", 'DrellYan']:
        histogram[sample] = read_histogram_from_sample(hist_name, sample)
    return histogram


def plot_histogram(hist_name):
    stack_order = ["single top", "diboson", 'DrellYan', "ttbar", "Z", "W"]
    plot_label = "$W \\rightarrow l\\nu$"
    print("==========")
    print("Plotting {0} histogram".format(hist_name))

    hist = hist_dicts[hist_name]

    h_bin_width = hist["bin_width"]
    h_num_bins = hist["numbins"]
    h_xmin = hist["xmin"]
    h_xmax = hist["xmax"]
    h_xlabel = hist["xlabel"]
    h_title = hist["title"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    hist_heights = read_histogram(hist_name)

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    plt.yscale("linear")
    main_axes = plt.gca()
    main_axes.set_title(h_title)

    main_axes.errorbar(x=bin_centers, y=hist_heights['data'],
                       yerr=np.sqrt(hist_heights['data']), xerr=h_bin_width/2,
                       fmt='ko')
    mc_labels = []
    mc_tot_heights = np.zeros(len(bin_centers))
    mc_x = []
    for sample in stack_order:
        mc_labels.append(sample)
        mc_color = WSamples.samples[sample]["color"]
        mc_heights = hist_heights[sample]
        main_axes.bar(x=bin_centers, height=mc_heights, width=h_bin_width, bottom=mc_tot_heights,
                      color=mc_color, label=sample)
        mc_tot_heights = np.add(mc_tot_heights, mc_heights)
    handles, labels = main_axes.get_legend_handles_labels()
    main_axes.legend(reversed(handles), reversed(labels), title=plot_label, loc="upper right")
    main_axes.set_xlim(h_xmin, h_xmax)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_xticklabels([])

    plt.axes([0.1, 0.1, 0.85, 0.2])
    plt.yscale("linear")
    ratio_axes = plt.gca()
    ratio_axes.errorbar(bin_centers, hist_heights['data'] / mc_tot_heights, xerr=h_bin_width / 2.,
                        fmt='.', color="black")
    ratio_axes.set_ylim(0, 2.5)
    ratio_axes.set_yticks([0, 1, 2])
    ratio_axes.set_xlim(h_xmin, h_xmax)
    ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel(f"Events/{h_bin_width} GeV")
    ratio_axes.set_ylabel("Ratio\nData/MC")
    ratio_axes.set_xlabel(h_xlabel)
    plt.grid("True", axis="y", color="black", linestyle="--")
    plt.savefig(f'../Results_8TeV/{hist_name}.jpeg')
    return None


def plotting_main():
    print('Started plotting')
    for hist_name in hist_dicts.keys():
        plot_histogram(hist_name)
    print('Plotting finished')


# plotting_main()