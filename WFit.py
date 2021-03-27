import pandas
import uproot
import time
import matplotlib.pyplot as plt
import os
import psutil
import numpy as np

import tensorflow as tf
import zfit
import probfit
import matplotlib.pyplot as plt

from WHistograms import hist_dicts
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


def format_data(data, obs, sample=None):
    if sample != 'data':
        return zfit.Data.from_numpy(obs, data.mtw.to_numpy(), weights=data.totalWeight.to_numpy())
    else:
        return zfit.Data.from_numpy(obs, data.mtw.to_numpy())


def create_initial_model(obs, sample):
    # Crystal Ball

    mu = zfit.Parameter(f"mu_{sample}", 80., 60., 120.)
    sigma = zfit.Parameter(f'sigma_{sample}', 8., 1., 100.)
    alpha = zfit.Parameter(f'alpha_{sample}', -.5, -10., 0.)
    n = zfit.Parameter(f'n_{sample}', 120., 0.01, 500.)
    model = zfit.pdf.CrystalBall(obs=obs, mu=mu, sigma=sigma, alpha=alpha, n=n)

    return model


def sum_func(*args):
    return sum(i for i in args)


def initial_fitter(data, sample, initial_parameters, obs):
    print('==========')
    print(f'Fitting {sample} sample')
    df = data[sample]
    bgr_yield = len(df.index)

    mu = zfit.Parameter(f"mu_{sample}", initial_parameters[sample]['mu'], 40., 100.)
    sigma = zfit.Parameter(f'sigma_{sample}', initial_parameters[sample]['sigma'], 1., 100.)
    alphal = zfit.Parameter(f'alphal_{sample}', initial_parameters[sample]['alphal'], 0., 10.)
    alphar = zfit.Parameter(f'alphar_{sample}', initial_parameters[sample]['alphar'], 0., 10.)
    nl = zfit.Parameter(f'nl_{sample}', initial_parameters[sample]['nl'], 0.01, 500.)
    nr = zfit.Parameter(f'nr_{sample}', initial_parameters[sample]['nr'], 0.01, 500.)
    n_bgr = zfit.Parameter(f'yield_DCB_{sample}', bgr_yield, 0., int(1.3 * bgr_yield), step_size=1)

    DCB = zfit.pdf.DoubleCB(obs=obs, mu=mu, sigma=sigma, alphal=alphal, nl=nl, alphar=alphar, nr=nr)
    DCB = DCB.create_extended(n_bgr)

    mu_g = zfit.Parameter(f"mu_gauss_{sample}", 41., 40., 100.)
    sigma_g = zfit.Parameter(f'sigma_gauss_{sample}', 8., 1., 100.)
    low = zfit.Parameter(f'low_{sample}', 50., 30., 200.)
    high = zfit.Parameter(f'high_{sample}', 100., 30., 200.)
    ad_yield = zfit.Parameter(f'yield_gauss_{sample}', int(0.2 * bgr_yield), 0., int(1.3 * bgr_yield), step_size=1)

   # gauss = zfit.pdf.TruncatedGauss(mu=mu_g, sigma=sigma_g, low=low, high=high, obs=obs)
   # gauss = gauss.create_extended(ad_yield)

   # model = zfit.pdf.SumPDF([DCB, gauss])
    model = DCB

    bgr_data = format_data(df, obs)
    # Create NLL
    nll = zfit.loss.ExtendedUnbinnedNLL(model=model, data=bgr_data)
    # Create minimizer
    minimizer = zfit.minimize.Minuit(verbosity=0, use_minuit_grad=True, tolerance=0.001)
    result = minimizer.minimize(nll)
    if result.valid:
        print("Result is valid")
        print("Converged:", result.converged)
        # param_errors = result.hesse()
        print(result.params)
        if not model.is_extended:
            raise Warning('MODEL NOT EXTENDED')
        return model
    else:
        print('Minimization failed')
        print(result.params)
        return model


# Plotting

def plot_fit_result(models, data, obs, sample='data'):
    plt_name = "mtw"
    print(f'Plotting {sample}')

    lower, upper = obs.limits

    h_bin_width = hist_dicts[plt_name]["bin_width"]
    h_num_bins = hist_dicts[plt_name]["numbins"]
    h_xmin = hist_dicts[plt_name]["xmin"]
    h_xmax = hist_dicts[plt_name]["xmax"]
    h_xlabel = hist_dicts[plt_name]["xlabel"]
    plt_label = "$W \\rightarrow l\\nu$"

    bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
    bin_centers = [h_xmin + h_bin_width / 2 + x * h_bin_width for x in range(h_num_bins)]

    data_x, _ = np.histogram(data.mtw.values, bins=bins, weights=data.totalWeight.values)
    data_sum = data_x.sum()
    plot_scale = data_sum * obs.area() / h_num_bins

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    main_axes = plt.gca()
    hep.histplot(main_axes.hist(data.mtw, bins=bins, log=False, facecolor="none", weights=data.totalWeight.values),
                 color="black", yerr=True, histtype="errorbar", label=sample)

    main_axes.set_xlim(h_xmin, h_xmax)
    main_axes.set_ylim(0., 1.4 * max(data_x))
    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_xlabel(h_xlabel)
    main_axes.set_title("W Transverse Mass Fit")
    main_axes.set_ylabel("Events/4 GeV")
    # main_axes.ticklabel_format(axis='y', style='sci', scilimits=[-2, 2]) todo

    x_plot = np.linspace(lower[-1][0], upper[0][0], num=1000)
    for model_name, model in models.items():
        if model.is_extended:
            print('Model is extended')
            main_axes.plot(x_plot, model.ext_pdf(x_plot) * obs.area() / h_num_bins, label=model_name)
        else:
            main_axes.plot(x_plot, model.pdf(x_plot) * plot_scale, label=model_name)
            print('Model is not extended')
    main_axes.legend(title=plt_label, loc="best")
    plt.savefig(f"../Results_8TeV/{sample}_fit_{plt_name}_Complex.jpg")
    plt.close()


def plot_component(dfs, component):
    print("###==========####")
    print("Started plotting")

    plot_label = "$W \\rightarrow l\\nu$"
    signal_label = "Signal $W$"

    signal = None
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

    data_x, _ = np.histogram(dfs[component][x_var].values, bins=bins)

    plt.clf()
    plt.style.use(hep.style.ATLAS)
    _ = plt.figure(figsize=(9.5, 9))
    plt.axes([0.1, 0.30, 0.85, 0.65])
    plt.yscale("linear")
    main_axes = plt.gca()
    main_axes.set_title(h_title)
    hep.histplot(main_axes.hist(data[component][x_var], bins=bins, log=False, facecolor="none"),
                 color="black", yerr=True, histtype="errorbar", label='data')

    main_axes.set_xlim(h_xmin * 0.9, h_xmax * 1.1)

    main_axes.xaxis.set_minor_locator(AutoMinorLocator())
    main_axes.set_ylabel(f"Events/{h_bin_width}")
    plt.savefig(f"../Results_8TeV/{component}_mtw.jpg")


data = get_data_from_files()
# for sample in ["diboson", "Z", "ttbar", "single top", "W", 'DrellYan']:
# plot_component(data, sample)

obs = zfit.Space('mtw', limits=(30, 200))
initial_parameters = {'diboson': {'mu': 80., 'sigma': 18., 'nl': 1., 'alphal': 1.3, 'nr': 90., 'alphar': 0.9},
                      'ttbar': {'mu': 77., 'sigma': 21., 'nl': 100., 'alphal': 0.1, 'nr': 10., 'alphar': 1.4},
                      'single top': {'mu': 80., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 110., 'alphar': 1.},
                      'Z': {'mu': 73., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 110., 'alphar': 1.},
                      'DrellYan': {'mu': 62.5, 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 110., 'alphar': 1.},
                      'W': {'mu': 80., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 110., 'alphar': 1.},
                      'data': {'mu': 80., 'sigma': 16., 'nl': 2., 'alphal': 0.6, 'nr': 110., 'alphar': 1.}}

model = initial_fitter(data, 'W', initial_parameters, obs)
plot_fit_result({'W': model}, data['W'], obs, sample='W')
