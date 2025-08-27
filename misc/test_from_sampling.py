import pyhf
import spey
from spey_pyhf.helper_functions import WorkspaceInterpreter
from os.path import join
import json 
import numpy as np
spey.set_log_level(0)
input_folder = "../data/1908.08215/"


with open(join(input_folder, 'bkgonly.json'), 'r') as fbkg:
    bkg_json = json.load(fbkg)

# with open(join(input_folder, 'patchset.json'), 'r') as fpatch:
#     patch_json = json.load(fpatch)

interpreter = WorkspaceInterpreter(bkg_json)
Nchannels = len(bkg_json['channels'])
for c in bkg_json['channels']:
    # print(c['name'], c['samples'][0]['data'])
    b = len(c['samples'][0]['data'])
    #bin_vals = np.array(np.array(c['samples'][0]['data'])*np.random.uniform(0.5,1.5)) 
    bin_vals = np.array([0.0] * b)
    interpreter.inject_signal(c['name'], bin_vals)

stat_wrapper = spey.get_backend("pyhf")
statistical_model = stat_wrapper(
                                    background_only_model=bkg_json,
                                    signal_patch=interpreter.make_patch(),
                                )
statistical_model.backend.manager.backend = "tensorflow"
nLL_exp_mu1 = statistical_model.likelihood(poi_test=1.0, expected='apriori')       
nLL_obs_mu1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
nLL_apost_mu1 = statistical_model.likelihood(poi_test=1.0, expected='aposteriori')

nLL_exp_mu0 = statistical_model.likelihood(poi_test=0.0, expected='apriori')       
nLL_obs_mu0 = statistical_model.likelihood(poi_test=0.0, expected='observed')
nLL_apost_mu0 = statistical_model.likelihood(poi_test=0.0, expected='aposteriori')

print(f'nLL_exp_mu1: {nLL_exp_mu1}')
print(f'nLL_obs_mu1: {nLL_obs_mu1}')
print(f'nLL_apost_mu1: {nLL_apost_mu1}')

print(f'nLL_exp_mu0: {nLL_exp_mu0}')
print(f'nLL_obs_mu0: {nLL_obs_mu0}')
print(f'nLL_apost_mu0: {nLL_apost_mu0}')

_, model_bkg, data_bkg = statistical_model.backend.model(expected=spey.ExpectationType.apriori)
print('data_bkg', data_bkg[:Nchannels])

_, model_apost, data_apost = statistical_model.backend.model(expected=spey.ExpectationType.aposteriori)
print('data_apost', data_apost[:Nchannels])

_, model_obs, data_obs = statistical_model.backend.model(expected=spey.ExpectationType.observed)
print('data_obs', data_obs[:Nchannels])

print("Checking nLL when there is observed signal in only one bin")
Sobs = np.array(data_obs[:Nchannels]) - np.array(data_bkg[:Nchannels]) + 1e-3

ll = 0
for c in bkg_json['channels']:
    print(f'Channel {c["name"]}. Signal: {Sobs[ll:ll+b]}')
    print(f'Expected {data_bkg[ll:ll+b]}, Observed {data_obs[ll:ll+b]}')
    interpreter = WorkspaceInterpreter(bkg_json)
    # print(c['name'], c['samples'][0]['data'])
    b = len(c['samples'][0]['data'])
    bin_vals = np.array(Sobs[ll:ll+b])
    interpreter.inject_signal(c['name'], bin_vals)

    stat_wrapper = spey.get_backend("pyhf")
    statistical_model = stat_wrapper(
                                        background_only_model=bkg_json,
                                        signal_patch=interpreter.make_patch(),
                                    )
    statistical_model.backend.manager.backend = "tensorflow"
    nLL_exp_mu1 = statistical_model.likelihood(poi_test=1.0, expected='apriori')       
    nLL_obs_mu1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
    nLL_apost_mu1 = statistical_model.likelihood(poi_test=1.0, expected='aposteriori')

    nLL_exp_mu0 = statistical_model.likelihood(poi_test=0.0, expected='apriori')       
    nLL_obs_mu0 = statistical_model.likelihood(poi_test=0.0, expected='observed')
    nLL_apost_mu0 = statistical_model.likelihood(poi_test=0.0, expected='aposteriori')

    print(f'nLL_exp_mu1: {nLL_exp_mu1}')
    print(f'nLL_obs_mu1: {nLL_obs_mu1}')
    print(f'nLL_apost_mu1: {nLL_apost_mu1}')

    print(f'nLL_exp_mu0: {nLL_exp_mu0}')
    print(f'nLL_obs_mu0: {nLL_obs_mu0}')
    print(f'nLL_apost_mu0: {nLL_apost_mu0}')

    ll += b

