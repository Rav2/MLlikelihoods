import os, sys
from os.path import join, dirname, basename, exists
import pandas as pd
import onnxruntime as rt
import onnx
import numpy as np
import json 
from time import process_time, sleep

def measure_time(model_path, size):
    json_path = join('../tables/', basename(dirname(model_path)), 'results-0.json')
    with open(json_path, 'r') as file:
        json_content = json.load(file)

    low_lim = json_content['lower_limits']
    up_lim = json_content['upper_limits']        
    data = np.random.uniform(low=low_lim, high=up_lim, size=(size, len(low_lim)))
    x_test = pd.DataFrame(data)
    t_start = process_time()  
    onnx_model = onnx.load(model_path)
    std_arr, mean_arr = None, None
    for em in onnx_model.metadata_props:
        if em.key == 'standardization_std':
            std_arr = np.array(json.loads(em.value))
        elif em.key == 'standardization_mean':
            mean_arr = np.array(json.loads(em.value))

    x_test_norm = pd.DataFrame()
    for cc, col in enumerate(x_test.columns):
        #print(col)
        vals = x_test.loc[:, col].to_numpy()
        x_test_norm.loc[:, col] = (vals - mean_arr[cc])/std_arr[cc]
    sess = rt.InferenceSession(onnx_model.SerializeToString())
    if False:
        input_name = sess.get_inputs()[0].name
        print("input name", input_name)
        input_shape = sess.get_inputs()[0].shape
        print("input shape", input_shape)
        input_type = sess.get_inputs()[0].type
        print("input type", input_type)

        output_name = sess.get_outputs()[0].name
        print("output name", output_name)
        output_shape = sess.get_outputs()[0].shape
        print("output shape", output_shape)
        output_type = sess.get_outputs()[0].type
        print("output type", output_type)

    x_test_norm = x_test_norm.to_numpy().astype(np.float32)

    y_pred = sess.run(None, {'input_1':x_test_norm})
    print('Zzzzzzzz')
    sleep(10)
    print('Good morning!')

    y_pred = np.array(y_pred[0])
    t_end = process_time()
    print('Total time: {} s.'.format(t_end-t_start))
    print(f'On average: {1000.*(t_end-t_start)/size} ms/point.')


if __name__ == '__main__':
    if len(sys.argv) < 3:
        print('python time_check.py <path_to_onnx> <N_points>')
        exit(1)

    measure_time(sys.argv[1], int(sys.argv[2]))
