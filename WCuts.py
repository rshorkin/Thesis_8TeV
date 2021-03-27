import math
import numpy as np


# Triggered by muon or electron
def cut_trig(trigE, trigM):
    return not (trigE or trigM)


# Passes good run list
def cut_passGRL(pass_GRL):
    return not pass_GRL


# has good vertex
def cut_hasGoodVertex(hasGoodVertex):
    return not hasGoodVertex


def cut_GoodLepton(lep_flag, lep_pt, lep_ptcone30, lep_etcone20, lep_n, lep_type):
    good_leptons = []
    for lepton in range(lep_n):
        e_or_mu = lep_type[lepton] == 11 or lep_type[lepton] == 13
        tight = bool(lep_flag[lepton] & 512)          # wtf
        fast = lep_pt[lepton] > 25000.
        iso_etcone_rel_20 = lep_etcone20[lepton]/lep_pt[lepton] < 0.15
        iso_ptcone_rel_30 = lep_ptcone30[lepton]/lep_pt[lepton] < 0.15
        if e_or_mu and tight and fast and iso_ptcone_rel_30 and iso_etcone_rel_20:
            good_leptons.append(lepton)
    if len(good_leptons) != 1:
        return -1
    else:
        return good_leptons[0]

