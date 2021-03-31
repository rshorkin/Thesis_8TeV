import ROOT
import numpy as np
import pandas
import math
import uproot
import time

import concurrent.futures

import os
import psutil
import gc
import uproot3

import WCuts
import infofile
import WSamples
from WHistograms import hist_dicts

import types
import uproot3_methods.classes.TH1

branches = ['eventNumber', 'runNumber', 'mcWeight',   # check for not needed
            'passGRL', 'hasGoodVertex', 'trigE', 'trigM',
            'scaleFactor_PILEUP', 'scaleFactor_ELE', 'scaleFactor_MUON', 'scaleFactor_TRIGGER', 'scaleFactor_ZVERTEX',
            'vxp_z', 'pvxp_n',

            'lep_n', 'lep_pt', 'lep_eta', 'lep_phi', 'lep_E', 'lep_type', 'lep_charge',
            'lep_ptcone30', 'lep_etcone20', 'lep_trackd0pvunbiased', 'lep_tracksigd0pvunbiased',
            'lep_trigMatched', 'lep_z0', 'lep_flag',     # mb can delete

            'alljet_n', 'jet_pt', 'jet_eta', 'jet_E', 'jet_phi', 'jet_m', 'jet_jvf', 'jet_MV1',  # check for un-needed

            'met_et', 'met_phi'
            ]

pandas.options.mode.chained_assignment = None


lumi = 1  # 10 fb-1
fraction = .05
common_path = "/media/sf_Shared/data_8TeV/"
# save_choice = int(input("Save dataframes? 0 for no, 1 for yes\n")) todo
save_choice = 0
if save_choice != 1:
    save_file = None
elif save_choice == 1:
    save_file = "csv"


def top_weight(x):
    return x / abs(x)


def calc_mtw(lep_pt, met_et, lep_phi, met_phi):
    return math.sqrt(2 * lep_pt * met_et * (1 - math.cos(lep_phi - met_phi)))


def get_xsec_weight(totalWeight, sample):
    info = infofile.infos[sample]
    weight = (lumi * 1000 * info["xsec"]) / (info["sumw"] * info["red_eff"])  # *1000 to go from fb-1 to pb-1
    weight = totalWeight * weight
    return weight


def to_GeV(x):
    return x / 1000.


def extract_good_lepton(x, id):
    return x[id]


def abs_value(x):
    return abs(x)


def calc_weight(mcWeight, scaleFactor_ELE, scaleFactor_MUON,
                scaleFactor_PILEUP, scaleFactor_TRIGGER, scaleFactor_ZVERTEX):
    return mcWeight * scaleFactor_ELE * scaleFactor_MUON * \
           scaleFactor_PILEUP * scaleFactor_TRIGGER * scaleFactor_ZVERTEX


def read_file(path, sample, branches=branches):
    print("=====")
    print("Processing {0} file".format(sample))
    mem = psutil.virtual_memory()
    mem_at_start = mem.available / (1024 ** 2)
    print(f'Available Memory: {mem_at_start:.0f} MB')
    count = 0
    hists = {}
    executor = concurrent.futures.ThreadPoolExecutor(4)
    start = time.time()
    batch_num = 0
    with uproot.open(path) as file:
        tree = file['mini']
        numevents = tree.num_entries
        print(f'Total number of events in file: {numevents}')

        for batch in tree.iterate(branches, step_size='30 MB', library='np',
                                  decompression_executor=executor,
                                  interpretation_executor=executor):
            print('==============')
            df = pandas.DataFrame.from_dict(batch)
            del batch
            num_before_cuts = len(df.index)
            print("Events before cuts: {0}".format(num_before_cuts))
            count += num_before_cuts
            if 'Data' not in sample:
                df['totalWeight'] = np.vectorize(calc_weight)(df.mcWeight, df.scaleFactor_ELE, df.scaleFactor_MUON,
                                                              df.scaleFactor_PILEUP, df.scaleFactor_TRIGGER,
                                                              df.scaleFactor_ZVERTEX)
                df["totalWeight"] = np.vectorize(get_xsec_weight)(df.totalWeight, sample)
            else:
                df['totalWeight'] = [1 for item in range(len(df.index))]

            df.drop(["mcWeight", "scaleFactor_ELE", "scaleFactor_MUON",
                     "scaleFactor_PILEUP", "scaleFactor_TRIGGER", 'scaleFactor_ZVERTEX'],
                    axis=1,
                    inplace=True)

            # Standard selection cuts
            df = df.query("trigE or trigM")
            df = df.query('passGRL')
            df = df.query('hasGoodVertex')

            df.drop(["trigE", "trigM", "passGRL", "hasGoodVertex"],
                    axis=1,
                    inplace=True)

            # Lepton requirements
            df['good_lepton'] = np.vectorize(WCuts.cut_GoodLepton)(df.lep_flag, df.lep_pt,
                                                                   df.lep_ptcone30, df.lep_etcone20,
                                                                   df.lep_n, df.lep_type)
            df = df.query('good_lepton > -1')
            for column in df.columns:
                if 'lep' in column and column not in ['lep_n', 'good_lepton']:
                    df[column] = np.vectorize(extract_good_lepton)(df[column], df['good_lepton'])

            # W transverse mass
            df['mtw'] = np.vectorize(calc_mtw)(df.lep_pt, df.met_et, df.lep_phi, df.met_phi)
            df = df.query('mtw > 30000.')
            df = df.query('met_et > 30000.')

            # Convert MeV to GeV
            df['lep_pt'] = df['lep_pt'] / 1000
            df['met_et'] = df['met_et'] / 1000
            df['mtw'] = df['mtw'] / 1000

            df['jet_n'] = df['alljet_n']
            df.drop(['alljet_n'], axis=1, inplace=True)

            # Asymmetry related histograms
            df['pos_ele_eta'] = df.query('lep_type == 11 and lep_charge == 1')['lep_eta']
            df['pos_ele_eta'] = np.vectorize(abs_value)(df.pos_ele_eta)

            df['neg_ele_eta'] = df.query('lep_type == 11 and lep_charge == -1')['lep_eta']
            df['neg_ele_eta'] = np.vectorize(abs_value)(df.neg_ele_eta)

            df['pos_mu_eta'] = df.query('lep_type == 13 and lep_charge == 1')['lep_eta']
            df['pos_mu_eta'] = np.vectorize(abs_value)(df.pos_mu_eta)

            df['neg_mu_eta'] = df.query('lep_type == 13 and lep_charge == -1')['lep_eta']
            df['neg_mu_eta'] = np.vectorize(abs_value)(df.neg_mu_eta)

            num_after_cuts = len(df.index)
            print("Number of events after cuts: {0}".format(num_after_cuts))
            print(f'Currently at {(count * 100 / numevents):.0f}% of events ({count}/{numevents})')

            for key, hist in hist_dicts.items():
                h_bin_width = hist["bin_width"]
                h_num_bins = hist["numbins"]
                h_xmin = hist["xmin"]

                x_var = hist["xvariable"]

                bins = [h_xmin + x * h_bin_width for x in range(h_num_bins + 1)]
                data_x, binning = np.histogram(df[x_var].values, bins=bins, weights=df.totalWeight.values)
                data_x = data_x.astype('float64')
                histo = uproot3_methods.classes.TH1.from_numpy((data_x, binning))
                if key not in hists.keys():
                    hists[key] = histo
                else:
                    for i in range(len(hists[key])):
                        hists[key][i] += histo[i]
            if not os.path.exists(f'../DataForFit_8TeV/{sample}/'):
                os.mkdir(f'../DataForFit_8TeV/{sample}')
            f = uproot3.recreate(f'../DataForFit_8TeV/{sample}/{sample}_{batch_num}.root')

            f['FitTree'] = uproot3.newtree({'mtw': uproot3.newbranch(np.float64, 'mtw'),
                                            'jet_n': uproot3.newbranch(np.int32, 'jet_n'),
                                            'totalWeight': uproot3.newbranch(np.float64, 'totalWeight')})

            f['FitTree'].extend({'mtw': df['mtw'].to_numpy(dtype=np.float64),
                                 'jet_n': df['jet_n'].to_numpy(dtype=np.int32),
                                 'totalWeight': df['totalWeight'].to_numpy(dtype=np.float64)})
            f.close()
            batch_num += 1
            del df
            gc.collect()
            # diagnostics
            mem = psutil.virtual_memory()
            actual_mem = mem.available / (1024 ** 2)
            print(f'Current available memory {actual_mem:.0f} MB '
                  f'({100 * actual_mem / mem_at_start:.0f}% of what we started with)')

    file = uproot3.recreate(f'../Output_8TeV/{sample}.root', uproot3.ZLIB(4))

    for key, hist in hists.items():
        file[key] = hist
        print(f'{key} histogram')
        file[key].show()

    file.close()

    mem = psutil.virtual_memory()
    actual_mem = mem.available / (1024 ** 2)
    print(f'Current available memory {actual_mem:.0f} MB '
          f'({100 * actual_mem / mem_at_start:.0f}% of what we started with)')
    print('Finished!')
    print(f'Time elapsed: {time.time() - start} seconds')
    return None


def read_sample(sample):
    print("###==========###")
    print("Processing: {0} SAMPLES".format(sample))
    start = time.time()

    for val in WSamples.samples[sample]["list"]:
        if sample == "data":
            prefix = "Data/"
        else:
            prefix = "MC/mc_{0}.".format(infofile.infos[val]["DSID"])
        path = common_path + prefix + val + ".root"
        if not path == "":
            read_file(path, val)
        else:
            raise ValueError("Error! {0} not found!".format(val))
    print("###==========###")
    print("Finished processing {0} samples".format(sample))
    print("Time elapsed: {0} seconds".format(time.time() - start))
    return None


def get_data_from_files():
    data = {}
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
        read_sample(s)
    return None


get_data_from_files()

