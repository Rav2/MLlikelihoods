import pyhf
import json
import jsonpatch
import spey
from spey_pyhf.helper_functions import WorkspaceInterpreter
import onnxruntime as rt
import onnx
from onnx2keras import onnx_to_keras
import numpy as np
import pandas as pd

def parse_patch_name(name):
    tokens = name.split('_')
    m1 = int(tokens[-2])
    m2 = int(tokens[-1])
    return m1, m2

pyhf.set_backend("tensorflow")


model_path = '../models/1911.06660-no-leakage/model.onnx'
bkgfile = "../data/1911.06660/Region-combined/BkgOnly.json"
patchset_path = "../data/1911.06660/Region-combined/patchset.json"
Nchan = 5

bkgspec = json.load(open(bkgfile))
bkgonly = pyhf.Workspace(bkgspec)


patchset = pyhf.PatchSet(json.load(open(patchset_path)))
DATA = []

for ip, patch in enumerate(patchset.patches[:5]):
    
    m1, m2 = parse_patch_name(patch.name)

    ws = jsonpatch.apply_patch ( bkgonly, patch )
    model = ws.model()
    if ip == 0:
        # Get the list of channels
        bins_names = []
        for k,v in model.config.channel_nbins.items():
            for ii in range(v):
                bins_names.append(k+'-{}'.format(ii))
        print(bins_names)
    
    print(f"[{ip+1}/{len(patchset.patches)}] m1={m1} GeV m2={m2} GeV")
    data = ws.data ( model )
    fit0=pyhf.infer.mle.fixed_poi_fit(0., data, model, return_fitted_val=True)
    lSM0 = float ( fit0[-1] ) / 2. ## this should be the nll!
    fit1=pyhf.infer.mle.fixed_poi_fit(1., data, model, return_fitted_val=True)
    lSM1 = float ( fit1[-1] ) / 2. ## this should be the nll!

    # Set the parameter of interest (POI) to 1
    # pyhf typically uses the first parameter as the POI (e.g., signal strength)
    poi_index = model.config.poi_index
    initial_parameters = model.config.suggested_init()

    initial_parameters[poi_index] = 1.0  # Set POI to 1
    # Set the fixed parameters (if any)
    fixed_parameters = model.config.suggested_fixed()
    # Calculate the expected data
    expected_data = model.expected_data(initial_parameters, fixed_parameters)
    # Create the observed data (for demonstration, use the expected data)
    observed_data = data

    # Evaluate the likelihood
    data = pyhf.tensorlib.astensor(observed_data)
    parameters = pyhf.tensorlib.astensor(initial_parameters)
    log_likelihood = model.logpdf(parameters, data)
    print(log_likelihood)

    # Calculate the expected data
    # expected_data = model.expected_data(initial_parameters, fixed_parameters)[:Nchan]
    # DATA.append([m1, m2]+list(expected_data)+[])
    # print("Expected data for POI=1:", expected_data)
    stat_wrapper = spey.get_backend("pyhf")

    statistical_model = stat_wrapper(
                                        background_only_model=bkgspec,
                                        signal_patch=patch,
                                    )

    nLL_obs_mu0 = statistical_model.likelihood(poi_test=0.0, expected='observed')
    nLL_obs_mu1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
    print(nLL_obs_mu0, nLL_obs_mu1)

    print ( "pyHF: nLL_obs_mu0={:.2f} nLL_obs_mu1={:.2f}".format(lSM0, lSM1) ) 

# dset = pd.DataFrame(DATA, columns=['m1', 'm2']+bins_names)
# dset.to_csv('signal_points.csv')
