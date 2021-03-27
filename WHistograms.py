# Common stuff for 8 TeV

etmiss = {"bin_width": 10,
          "numbins": 20,
          "xmin": 0,
          "xmax": 200,
          "xlabel": "$E^{miss}_T$, [GeV]",
          "xvariable": "met_et",
          "title": "Missing Transverse Momentum",
          }

pvxp_n = {"bin_width": 1,
          "numbins": 30,
          "xmin": -0.5,
          "xmax": 29.5,
          "xlabel": "N_{vertex}",
          "xvariable": "pvxp_n",
          "title": "Number of Vertices",
          }

mtw = {"bin_width": 5,
       "numbins": 40,
       "xmin": 0,
       "xmax": 200,
       "xlabel": "$M^W_T$, [GeV]",
       "xvariable": "mtw",
       "title": "Transverse Mass"}

lep_pt = {"bin_width": 10,
          "numbins": 20,
          "xmin": 0,
          "xmax": 200,
          "xlabel": "$p_T^{lep}$, [GeV]",
          "xvariable": "lep_pt",
          "title": "Lepton Transverse Momentum"}

lep_eta = {"bin_width": 0.2,
           "numbins": 24,
           "xmin": -2.4,
           "xmax": 2.4,
           "xlabel": "$\eta^{lep}$",
           "xvariable": "lep_eta",
           "title": "Lepton Pseudorapidity"}

jet_n = {"bin_width": 1,
         "numbins": 10,
         "xmin": -0.5,
         "xmax": 9.5,
         "xlabel": "$N_{jets}$",
         "xvariable": "jet_n",
         "title": "Number of Jets"}

lep_ch = {"bin_width": 0.5,
          "numbins": 5,
          "xmin": -1.25,
          "xmax": 1.25,
          "xlabel": "$Q^{lep}$",
          "xvariable": "lep_charge",
          "title": "Lepton Charge"}

lep_type = {"bin_width": 1,
            "numbins": 3,
            "xmin": 10.5,
            "xmax": 13.5,
            "xlabel": "$|PDG ID|^{lep}$",
            "xvariable": "lep_type",
            "title": "Lepton Absolute PDG ID"}

# Asymmetry-related stuff 8 TeV

lep_abs_eta = {"bin_width": 0.2,
               "numbins": 12,
               "xmin": 0.,
               "xmax": 2.4,
               "xlabel": "$|\eta|^{lep}$",
               "xvariable": "lep_eta",
               "title": "Lepton Absolute Pseudorapidity"}

lep_asym = {"bin_width": 0.2,
            "numbins": 12,
            "xmin": 0.,
            "xmax": 2.4,
            "xlabel": "$|\eta|^{lep}$",
            "xvariable": "lep_asym",
            "title": "Lepton Charge Asymmetry"}

pos_ele_eta = lep_abs_eta.copy()
pos_ele_eta['xvariable'] = 'pos_ele_eta'
neg_ele_eta = lep_abs_eta.copy()
neg_ele_eta['xvariable'] = 'neg_ele_eta'
pos_mu_eta = lep_abs_eta.copy()
pos_mu_eta['xvariable'] = 'pos_mu_eta'
neg_mu_eta = lep_abs_eta.copy()
neg_mu_eta['xvariable'] = 'neg_mu_eta'

hist_dicts = {"met_et": etmiss, "mtw": mtw, "jet_n": jet_n, "lep_pt": lep_pt, "lep_eta": lep_eta, 'pvxp_n': pvxp_n,
              'pos_ele_eta': pos_ele_eta, 'neg_ele_eta': neg_ele_eta,
              'pos_mu_eta': pos_mu_eta, 'neg_mu_eta': neg_mu_eta}
