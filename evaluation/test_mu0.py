import numpy as np 
import pandas as pd
import onnxruntime as rt
import onnx
import json 
import os, sys

model_path = sys.argv[1]#'../models/1909.09226-100k-fluct10%/NNAsimov_v1.onnx'

onnx_model = onnx.load(model_path)
std_arr, mean_arr, nLL_exp_mu0, nLL_obs_mu0, nLLA_exp_mu0, nLLA_obs_mu0 = None, None, None, None, None, None
for em in onnx_model.metadata_props:
    # print(f'{em.key}\t:\t{em.value}')
    if em.key == 'standardization_std':
        std_arr = np.array(json.loads(em.value))
    elif em.key == 'standardization_mean':
        mean_arr = np.array(json.loads(em.value))
    elif em.key == 'nLL_exp_mu0':
        nLL_exp_mu0 = json.loads(em.value)
    elif em.key == 'nLL_obs_mu0':
        nLL_obs_mu0 = json.loads(em.value)
    elif em.key == 'nLLA_exp_mu0':
        nLLA_exp_mu0 = json.loads(em.value)
    elif em.key == 'nLLA_obs_mu0':
        nLLA_obs_mu0 = json.loads(em.value)
    elif em.key == 'bkg_yields':
        bkg = json.loads(em.value)
    elif em.key == 'obs_yields':
        obs = json.loads(em.value)
    elif em.key == 'channels':
        channel_types = json.loads(em.value)

print('means', mean_arr)
print('stds', std_arr)
print()

channels_bkg = [em[0] for em in bkg]
channels_obs = [em[0] for em in obs]
assert channels_bkg == channels_obs
print('channels', channels_bkg)

nll_mu0 = np.array([nLL_exp_mu0, nLL_obs_mu0, nLLA_exp_mu0, nLLA_obs_mu0,])

bkg_yields = [em[1] for em in bkg]
obs_yields = [em[1] for em in obs]

Nchannels = len(bkg_yields)
assert Nchannels+4 == len(mean_arr)
assert Nchannels+4 == len(std_arr)
print('Nchannels', Nchannels)
print()
bkg_yields_norm = (bkg_yields - mean_arr[:Nchannels])/std_arr[:Nchannels]
print('bkg_yields', bkg_yields)
print('bkg_yields_norm', bkg_yields_norm)
print()
obs_yields_norm = (obs_yields - mean_arr[:Nchannels])/std_arr[:Nchannels]
print('obs_yields', obs_yields)
print('obs_yields_norm', obs_yields_norm)
print()

print('nll_mu0', nll_mu0)
print()

sess = rt.InferenceSession(onnx_model.SerializeToString())
print('BKG')
y_pred_norm = sess.run(None, {'input_1': np.array([bkg_yields_norm], dtype=np.float32)})
y_pred_norm = np.ndarray.flatten(y_pred_norm[0]) 
print('y_pred_norm', y_pred_norm)
y_pred =  nll_mu0 + y_pred_norm * std_arr[-4:] + mean_arr[-4:]
print('y_pred', y_pred)
print('2nd+3rd term:', y_pred_norm * std_arr[-4:] + mean_arr[-4:])
print()

print('OBS')
y_pred_norm = sess.run(None, {'input_1': np.array([obs_yields_norm], dtype=np.float32)})
y_pred_norm = np.ndarray.flatten(y_pred_norm[0])
print('y_pred_norm', y_pred_norm)
y_pred =  nll_mu0 + y_pred_norm * std_arr[-4:] + mean_arr[-4:]
print('y_pred', y_pred)
print('2nd+3rd term:', y_pred_norm * std_arr[-4:] + mean_arr[-4:])
print()

print('EXPECTED')
exp_inputs = [ bkg[i][1] if channel_types[0][bkg[i][0].split('-')[0] ] == 'SR' else obs[i][1] for i in range(len(bkg)) ] 
exp_inputs_norm = (exp_inputs - mean_arr[:Nchannels])/std_arr[:Nchannels]
y_pred_norm = sess.run(None, {'input_1': np.array([exp_inputs_norm], dtype=np.float32)})
y_pred_norm = np.ndarray.flatten(y_pred_norm[0])
print('y_pred_norm', y_pred_norm)
y_pred =  nll_mu0 + y_pred_norm * std_arr[-4:] + mean_arr[-4:]
print('y_pred', y_pred)
print('2nd+3rd term:', y_pred_norm * std_arr[-4:] + mean_arr[-4:])
print()

# print('FOUND')
# found_yields = [ bkg[ii][1] if channel_types[bkg[ii][0]] == 'SR' else obs[ii][1] ]

# for ii in [-8, -6, -4, -2]:
#     found_yields[ii+1] = found_yields[ii+1] - found_yields[ii]
#     found_yields[ii] = None
# found_yields = [x for x in found_yields if x is not None] 

# found_yields_norm = (found_yields - mean_arr)/std_arr
# print('found_yields', found_yields[:Nchannels])
# print('found__yields_norm', found_yields_norm[:Nchannels])
# print('found_yields_targets', found_yields[Nchannels:])
# print('found__yields_targets_norm', found_yields_norm[Nchannels:])

# y_pred_norm = sess.run(None, {'input_1': np.array( [found_yields_norm[:Nchannels]], dtype=np.float32)})
# y_pred_norm = np.ndarray.flatten(y_pred_norm[0])
# print('y_pred_norm', y_pred_norm)
# y_pred =  nll_mu0 + y_pred_norm * std_arr[-4:] + mean_arr[-4:]
# print('y_pred', y_pred)
# print('2nd+3rd term:', y_pred_norm * std_arr[-4:] + mean_arr[-4:])


 # 4.18988813   2.90037017   1.00784736  10.9629761   10.32217295
 #    6.93054017   5.37724509   2.81015898   1.38682771 178.36254048
 #  680.52547771 717.51200764 474.40051217 130.14000746
