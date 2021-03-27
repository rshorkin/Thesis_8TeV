import pandas
import uproot
import time
import matplotlib.pyplot as plt
import os
import psutil
import numpy as np

import WSamples
from WHistograms import hist_dicts
import mplhep as hep
from matplotlib.ticker import AutoMinorLocator


branches = ['mtw', 'totalWeight']

pandas.options.mode.chained_assignment = None


lumi = 1  # 10 fb-1
common_path = "../DataForFit_8TeV/"


def read_file(path, sample, branches=branches):
    with uproot.open(path) as file:
        tree = file["FitTree"]
        df = pandas.DataFrame.from_dict(tree.arrays(branches, library='np'))
    return df


def read_sample(sample):
    print("###==========###")
    print("Processing: {0} SAMPLES".format(sample))

    start = time.time()
    frames = []
    for val in WSamples.samples[sample]["list"]:
        path = common_path + f'{val}/'
        partial_dfs = []
        if not path == "":
            for filename in os.listdir(path):
                if filename.endswith('.root'):
                    filepath = os.path.join(path, filename)
                    partial_df = read_file(filepath, val)
                    partial_dfs.append(partial_df)
            temp_df = pandas.concat(partial_dfs)
            frames.append(temp_df)
        else:
            raise ValueError("Error! {0} not found!".format(val))
    df_sample = pandas.concat(frames)
    print("###==========###")
    print("Finished processing {0} samples".format(sample))
    print("Time elapsed: {0} seconds".format(time.time() - start))
    return df_sample


def get_data_from_files():
    data = {}

    mem = psutil.virtual_memory()
    mem_at_start = mem.available / (1024 ** 2)
    print(f'Available Memory: {mem_at_start:.0f} MB')

    # switch = int(input("What do you want to analyze? 0 for all, 1 for data, 2 for MC\n")) todo
    switch = 0
    if switch == 0:
        samples = ["data", "diboson", "ttbar", "Z", "single top", "W", 'DrellYan']
    elif switch == 1:
        samples = ["data"]
    elif switch == 2:
        samples = ["diboson", "ttbar", "Z", "single top", "W", 'DrellYan']
    else:
        raise ValueError("Option {0} cannot be processed".format(switch))
    for s in samples:
        data[s] = read_sample(s)
        mem = psutil.virtual_memory()
        actual_mem = mem.available / (1024 ** 2)
        print(f'Current available memory {actual_mem:.0f} MB '
              f'({100 * actual_mem / mem_at_start:.0f}% of what we started with)')
        if actual_mem < 150:
            raise Warning('Out of RAM')
    return data


def plot_data(data):
    print("###==========####")
    print("Started plotting")

    plot_label = "$W \\rightarrow l\\nu$"
    signal_label = "Signal $W$"

    signal = None
    stack_order = ["DrellYan", "diboson", "Z", "ttbar", "single top", "W"]

    hist = hist_dicts['mtw']
    h_bin_width = hist["bin_width"]
    h_num_bins = hist["numbins"]
    h_xmin = hist["xmin"]
    h_xmax = hist["xmax"]
    h_xlabel = hist["xlabel"]
    x_var = hist["xvariable"]
    h_title = hist["title"]

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(data["data"][x_var].values, bins=bins)

    mc_x = []
    mc_weights = []
    mc_colors = []
    mc_labels = []
    mc_tot_heights = np.zeros(len(bin_centers))

    for sample in stack_order:
        mc_labels.append(sample)
        mc_x.append(data[sample][x_var].values)
        mc_colors.append(WSamples.samples[sample]["color"])
        mc_weights.append(data[sample].totalWeight.values)
        mc_heights, _ = np.histogram(data[sample][x_var].values, bins=bins, weights=data[sample].totalWeight.values)
        mc_tot_heights = np.add(mc_tot_heights, mc_heights)

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    plt.yscale("linear")
    main_axes = plt.gca()
    main_axes.set_title(h_title)
    hep.histplot(main_axes.hist(data["data"][x_var], bins=bins, log=False, facecolor="none"),
                 color="black", yerr=True, histtype="errorbar", label='data')
    ns, n_bins, patches = main_axes.hist(mc_x, bins=bins, weights=mc_weights, stacked=True, color=mc_colors,
                                         label=mc_labels)
    handles, labels = main_axes.get_legend_handles_labels()
    main_axes.legend(reversed(handles), reversed(labels), title=plot_label, loc="upper right")
    main_axes.set_xlim(h_xmin * 0.9, h_xmax * 1.1)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_xticklabels([])

    plt.axes([0.1, 0.1, 0.85, 0.2])
    plt.yscale("linear")
    ratio_axes = plt.gca()
    ratio_axes.errorbar(bin_centers, data_x / mc_tot_heights, xerr=h_bin_width / 2., fmt='.', color="black")
    ratio_axes.set_ylim(0, 2.5)
    ratio_axes.set_yticks([0, 1, 2])
    ratio_axes.set_xlim(h_xmin * 0.9, h_xmax * 1.1)
    ratio_axes.xaxis.set_minor_locator(AutoMinorLocator())

    main_axes.set_ylabel(f"Events/{h_bin_width}")
    ratio_axes.set_ylabel("Ratio\nData/MC")
    ratio_axes.set_xlabel(h_xlabel)
    plt.grid("True", axis="y", color="black", linestyle="--")
    # plt.show()
    plt.savefig(f"../Results_8TeV/fit_test.jpg")


data = get_data_from_files()
plot_data(data)
