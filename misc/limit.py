import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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
import matplotlib.pyplot as plt
import logging

# Configure logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

# Clear existing handlers
if logger.hasHandlers():
    logger.handlers.clear()

# Create handlers
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_handler = logging.FileHandler('process.log')
file_handler.setLevel(logging.INFO)

# Create formatter and add it to the handlers
formatter = logging.Formatter('[%(asctime)s] - [%(levelname)s] - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Add handlers to the logger
logger.addHandler(console_handler)
logger.addHandler(file_handler)

def parse_patch_name(name):
    tokens = name.split('_')
    m1 = int(tokens[-2])
    m2 = int(tokens[-1])
    return m1, m2

pyhf.set_backend("tensorflow")

model_path = '../models/1909.09226-mu1-1/model.onnx'
bkgfile = "../data/1909.09226/BkgOnly.json"
patchset_path = "../data/1909.09226/patchset.json"
Nchan = 14

bkgspec = json.load(open(bkgfile))
bkgonly = pyhf.Workspace(bkgspec)
patchset = pyhf.PatchSet(json.load(open(patchset_path)))

# Load the ONNX model
onnx_model = onnx.load(model_path)
sess = rt.InferenceSession(onnx_model.SerializeToString())
std_arr, mean_arr = None, None
for em in onnx_model.metadata_props:
    if em.key == 'standardization_std':
        std_arr = np.array(json.loads(em.value))
    elif em.key == 'standardization_mean':
        mean_arr = np.array(json.loads(em.value))

nll0_err = []
nll1_err = []
qtilde_err = []

POINTS = []
LIMIT_SPEY = []
LIMIT_NN = []
for ip, patch in enumerate(patchset.patches[:]):
    m1, m2 = parse_patch_name(patch.name)

    ws = jsonpatch.apply_patch(bkgonly, patch)
    model = ws.model()
    observed_data = ws.data(model)

    if ip == 0:
        # Get the list of channels
        bins_names = []
        for k, v in model.config.channel_nbins.items():
            for ii in range(v):
                bins_names.append(k + '-{}'.format(ii))
        logging.info(f"Channel bins: {bins_names}")
    
    logging.info(f"[{ip+1}/{len(patchset.patches)}] m1={m1} GeV m2={m2} GeV")

    stat_wrapper = spey.get_backend("pyhf")
    statistical_model = stat_wrapper(
        background_only_model=bkgspec,
        signal_patch=patch,
    )

    lSM0 = statistical_model.likelihood(poi_test=0.0, expected='observed')
    lSM1 = statistical_model.likelihood(poi_test=1.0, expected='observed')
    LIMIT_SPEY.append([m1, m2, 2*(lSM1-lSM0)])
    logging.info(f"spey: nLL_obs_mu0={lSM0:.2f} nLL_obs_mu1={lSM1:.2f} 2∆={2*(lSM1-lSM0):.2f}")

    poi_index = model.config.poi_index
    initial_parameters = model.config.suggested_init()
    initial_parameters[poi_index] = 1.0  # Set POI to 1
    fixed_parameters = model.config.suggested_fixed()
    expected_data_mu1 = model.expected_data(initial_parameters, fixed_parameters)
    POINTS.append([m1, m2] + list(expected_data_mu1[:Nchan]) + [lSM0, lSM1])

    # Normalize data
    x_test_norm = np.zeros(Nchan, dtype=np.float32)
    for cc in range(Nchan):
        x_test_norm[cc] = (expected_data_mu1[cc] - mean_arr[cc]) / std_arr[cc]
    x_test_norm = x_test_norm.reshape(1, len(x_test_norm))
    
    # Predict
    y_pred = sess.run(None, {'input_1': x_test_norm})[0]
    y_pred = np.array(y_pred[0])
    logging.info(f"NN: nLL_obs_mu0={y_pred[2]:.2f} nLL_obs_mu1={y_pred[3]:.2f} 2∆={2*(y_pred[3] - y_pred[2]):.2f}")

    nll0_err.append(y_pred[2] - lSM0)
    nll1_err.append(y_pred[3] - lSM1)
    qtilde_err.append(2 * (y_pred[3] - y_pred[2]) - 2 * (lSM1 - lSM0))
    LIMIT_NN.append([m1, m2, 2 * (y_pred[3] - y_pred[2])])

dat_to_save = np.array(POINTS).reshape(ip + 1, Nchan + 4)
logging.info(f"Data to save shape: {dat_to_save.shape}")
points_dset = pd.DataFrame(dat_to_save, columns=(['m1 GeV', 'm2 GeV'] + bins_names + ['nLL_obs_mu0', 'nLL_obs_mu1']))
points_dset.to_csv('datapoints.csv')

plt.clf()
fig, axs = plt.subplots(1, 3, figsize=(15, 5))
axs[0].hist(nll0_err, 30)
axs[0].set_ylabel('counts')
axs[0].set_xlabel('nLL_obs_mu0 error')

axs[1].hist(nll1_err, 30)
axs[1].set_ylabel('counts')
axs[1].set_xlabel('nLL_obs_mu1 error')

axs[2].hist(qtilde_err, 30)
axs[2].set_ylabel('counts')
axs[2].set_xlabel('qtilde error')
plt.tight_layout()
plt.show()

LIMIT_SPEY_ARR = np.array(LIMIT_SPEY).T
LIMIT_NN_ARR = np.array(LIMIT_NN).T

plt.clf()
fig, axs = plt.subplots(1, 2, figsize=(10, 5))
scatter1 = axs[0].scatter(LIMIT_SPEY_ARR[0], LIMIT_SPEY_ARR[1], c=np.sqrt(LIMIT_SPEY_ARR[2]))
fig.colorbar(scatter1, ax=axs[0])
scatter2 = axs[1].scatter(LIMIT_NN_ARR[0], LIMIT_NN_ARR[1], c=np.sqrt(LIMIT_NN_ARR[2]))
fig.colorbar(scatter2, ax=axs[1])
plt.tight_layout()
plt.show()
