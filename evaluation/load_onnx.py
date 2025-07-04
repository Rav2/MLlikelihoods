import onnxruntime as rt
import onnx
import pandas as pd
import numpy as np 
import json
import os, sys

# 

def load_model_and_normalize(model_path, x_test):
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

    x_test_norm = pd.DataFrame()
    if x_test is None:
        raise ValueError('[ERROR] x_test is None!')
    else:
        for cc, col in enumerate(x_test.columns):
            vals = x_test.loc[:, col].to_numpy()
            if std_arr[cc] != 0.0:
                x_test_norm.loc[:, col] = (vals - mean_arr[cc])/std_arr[cc]
            else:
                x_test_norm.loc[:, col] = (vals - mean_arr[cc])

    sess = rt.InferenceSession(onnx_model.SerializeToString())
    input_name = sess.get_inputs()[0].name
    input_shape = sess.get_inputs()[0].shape
    input_type = sess.get_inputs()[0].type
    output_name = sess.get_outputs()[0].name
    output_shape = sess.get_outputs()[0].shape
    output_type = sess.get_outputs()[0].type
    return sess, x_test_norm, mean_arr, std_arr, nLL_exp_mu0, nLL_obs_mu0, nLLA_exp_mu0, nLLA_obs_mu0
