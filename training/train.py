import tensorflow as tf

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except:
    print('\033[93m' + '[WARNING] Could not activate GPU acceleration!'+'\033[0m')

import warnings

warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="tensorflow.python.data.ops.structured_function"
)

import sys
import os

# Set up the logger
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'..', 'sampling'))
sys.path.insert(0, module_path)
from misc import *
logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s  (%(filename)s:%(lineno)d)")
fileHandler = logging.FileHandler("log.txt", mode='w')
fileHandler.setFormatter(logFormatter)
consoleHandler = logging.StreamHandler()
consoleFormatter = CustomFormatter()
consoleHandler.setFormatter(consoleFormatter)
log = logging.getLogger()  # root logger
for hdlr in log.handlers[:]:  # remove all old handlers
    log.removeHandler(hdlr)
log.addHandler(fileHandler)      # set the new handler
log.addHandler(consoleHandler)  
log.setLevel(logging.INFO)


import numpy as np
from tensorflow import keras
from NNmodel import MyModelNN
from BNNmodel import MyModelBNN, MyModelBNNSimple
import pandas as pd
import matplotlib as mpl
from matplotlib import pyplot as plt
import onnx, tf2onnx
from os.path import dirname, basename, join
# import onnxruntime as rt
import json
from sklearn.preprocessing import StandardScaler
import optuna
from optimize import *
import gc
import shutil
import platform
import time


def get_time_string(ns_t):
    s_t = ns_t//10**9
    m_t = s_t // 60
    h_t = m_t // 60
    left_s = s_t % 60
    left_m = m_t - 60*h_t
    return f"{h_t} hours {left_m} minutes {left_s} seconds"


def plot_weights(train_weights, val_weights, outpath):
    fig, axs = plt.subplots(2,1)
    axs[0].hist(train_weights, 100)
    axs[1].hist(val_weights, 100)
    for ax, lab in zip(axs, ['train', 'val']):
        ax.set_yscale('log')
        ax.set_xlabel(f'{lab} weights')
        ax.set_ylabel('Counts')
    plt.tight_layout()
    plt.savefig(join(outpath, 'weights_hist.pdf'))


def plot_deltas(df, outpath, normalised):
    fig, axs = plt.subplots(2,2)
    for ii in range(0, 4):
        axs[ii//2, ii%2].hist(df.iloc[:, -4+ii], 100)
        axs[ii//2, ii%2].set_yscale('log')
        axs[ii//2, ii%2].set_xlabel(df.columns[-4+ii])
        axs[ii//2, ii%2].set_ylabel('Counts')
        axs[ii//2, ii%2].title.set_text(f'mu={np.round(df.iloc[:, -4+ii].mean(),3)}, std={np.round(df.iloc[:, -4+ii].std(),3)}')
    plt.tight_layout()
    if normalised:
        plt.savefig(join(outpath, 'deltas_norm_hist.pdf'))
    else:
         plt.savefig(join(outpath, 'deltas_hist.pdf'))


def main(infile,  model_name='model'):

    start_time = time.process_time_ns()
    outfolder = basename(dirname(infile))
    outpath_model = os.path.abspath(os.path.join(os.path.dirname(__file__),'../models', outfolder))
    if not os.path.exists(outpath_model):
        os.mkdir(outpath_model)
    outpath_aux = os.path.abspath(os.path.join(os.path.dirname(__file__), '../auxiliary', outfolder+'-'+model_name))
    if not os.path.exists(outpath_aux):
        os.mkdir(outpath_aux)

    # handle the CSV data
    if basename(infile) == 'train.csv':
        train = pd.read_csv(infile)
        val = pd.read_csv(infile.replace('train','val'))
        test = pd.read_csv(infile.replace('train','test'))
    else:
        df = pd.read_csv(infile)
        train, val, test = np.split(df.sample(frac=1),[int(.8*len(df)), int(.9*len(df))])
        train.reset_index(drop=True, inplace=True)
        val.reset_index(drop=True, inplace=True)
        test.reset_index(drop=True, inplace=True)
        train.to_csv(join(outpath_aux, "train.csv"), index=False)
        val.to_csv(join(outpath_aux,"val.csv"), index=False)
        test.to_csv(join(outpath_aux,"test.csv"), index=False)
    
    logging.info(f'Loaded CSV data:')
    logging.info(f'train shape: {train.shape}')
    logging.info(f'val shape: {val.shape}')
    logging.info(f'test shape: {test.shape}')

    # calculate differences of nLL
    nLL_exp_mu0 = train.iloc[:, -8].mean()
    nLL_obs_mu0 = train.iloc[:, -6].mean()
    nLLA_exp_mu0 = train.iloc[:, -4].mean()
    nLLA_obs_mu0 = train.iloc[:, -2].mean()
    shift = [nLL_exp_mu0, nLL_obs_mu0, nLLA_exp_mu0, nLLA_obs_mu0]
    train.iloc[:, -7] =  train.iloc[:, -7] - train.iloc[:, -8]
    train.iloc[:, -5] =  train.iloc[:, -5] - train.iloc[:, -6]
    train.iloc[:, -3] =  train.iloc[:, -3] - train.iloc[:, -4]
    train.iloc[:, -1] =  train.iloc[:, -1] - train.iloc[:, -2]
    cols_to_drop = [ train.columns[cc] for cc in (-8, -6, -4, -2) ]
    train.drop(cols_to_drop, axis=1, inplace=True)

    test.iloc[:, -7] =  test.iloc[:, -7] - test.iloc[:, -8]
    test.iloc[:, -5] =  test.iloc[:, -5] - test.iloc[:, -6]
    test.iloc[:, -3] =  test.iloc[:, -3] - test.iloc[:, -4]
    test.iloc[:, -1] =  test.iloc[:, -1] - test.iloc[:, -2]
    test.drop(cols_to_drop, axis=1, inplace=True)
    
    val.iloc[:, -7] =  val.iloc[:, -7] - val.iloc[:, -8]
    val.iloc[:, -5] =  val.iloc[:, -5] - val.iloc[:, -6]
    val.iloc[:, -3] =  val.iloc[:, -3] - val.iloc[:, -4]
    val.iloc[:, -1] =  val.iloc[:, -1] - val.iloc[:, -2]
    val.drop(cols_to_drop, axis=1, inplace=True)

    if False:
        mask = (train.iloc[:, -4:] > 80.0).any(axis=1)
        train = train[~mask].copy()
        mask = (val.iloc[:, -4:] > 80.0).any(axis=1)
        val = val[~mask].copy()
        mask = (test.iloc[:, -4:] > 80.0).any(axis=1)
        test = test[~mask].copy()
        logging.warning('Removing rows with large Deltas!')

    logging.info(f'After data selection:')
    logging.info(f'train shape: {train.shape}')
    logging.info(f'val shape: {val.shape}')
    logging.info(f'test shape: {test.shape}')

    plot_deltas(train, outpath_aux, False)

    # assign weights based on nLL_obs_mu1
    if False:
        target_index = -3
        train_weights = np.exp(-0.5 * train.iloc[:, target_index]) 
        val_weights = np.exp(-0.5 * val.iloc[:, target_index])
        # normalize the weights
        train_weights /= np.max(train_weights)
        val_weights /= np.max(val_weights)
        plot_weights(train_weights, val_weights, outpath_aux)
        logging.info(f'train weights shape: {train_weights.shape}')
        logging.info(f'val weights shape: {val_weights.shape}')


    # scale all columns
    columns = train.columns
    logging.info(f'Train columns: {columns}')
    std_scaler = StandardScaler()
    train_scaled = std_scaler.fit_transform(train.to_numpy())
    mean_arr = std_scaler.mean_ 
    std_arr = np.sqrt(std_scaler.var_)
    train_scaled = pd.DataFrame(train_scaled, columns=columns)
    val_scaled = std_scaler.transform(val.to_numpy())
    val_scaled = pd.DataFrame(val_scaled, columns=columns)
    test_scaled = std_scaler.transform(test.to_numpy())
    test_scaled = pd.DataFrame(test_scaled, columns=columns)
    logging.info('Train after the normalisation:')
    for col in columns:
        logging.info(f'{col}: mean={train_scaled[col].mean()} std={train_scaled[col].std()}')
    
    logging.info('\nVal after the normalisation:')
    for col in columns:
        logging.info(f'{col}: mean={val_scaled[col].mean()} std={val_scaled[col].std()}')

    plot_deltas(train_scaled, outpath_aux, True)

    # optimize the parameters
    OPTIMIZE = False
    best_trail_path = os.path.abspath(os.path.join(os.path.dirname(__file__),'../auxiliary/', outfolder+'-'+model_name, 'best_trial.txt')) 
    if OPTIMIZE:
        if not os.path.exists(best_trail_path):
            parameters = optimize_params(train_scaled, val_scaled, best_trail_path, n_trials=100, n_jobs=5)
        else:
            logging.warning('Using existing parameters.')
            parameters = {}
            try:
                with open(best_trail_path, 'r') as fin:
                    parameters = json.load(fin)
            except:
                with open(best_trail_path, 'r') as fin:
                    lines = fin.readlines()
                for line in lines:
                    k,v = line.split(':')
                    if k == 'activation':
                        parameters[k] = v.strip()
                    elif '.' in v or 'e' in v or 'E' in v:
                        parameters[k] = float(v)
                    else:
                        parameters[k] = int(v)
    else: 
        parameters = {'neurons' :512, 
                     'blocks' : 4, 
                     'l2' : 1e-5,
                     'activation' : 'elu',
                     'batch_norm' : True,  
                    }
        with open(best_trail_path, 'w') as fpar:
            json.dump(parameters, fpar)

    gc.collect()

    #
    # Compile and train the best model
    #
    logging.info(f'NN parameters: {parameters}')
    # Learning Scheduler
    decay_steps = 4000*233
    initial_learning_rate = 0.
    warmup_steps = 40*233
    target_learning_rate = 0.001
    lr_warmup_decayed_fn = keras.optimizers.schedules.CosineDecay(
        initial_learning_rate, decay_steps, warmup_target=target_learning_rate,
        warmup_steps=warmup_steps
    )

    # Model creation
    themodel = MyModelNN(   input_shape = (len(columns)-4,),
                            neurons = parameters['neurons'], 
                            blocks = parameters['blocks'], 
                            l2 = parameters['l2'],
                            activation = parameters['activation'],
                            loss='huber',
                            output_size = 4,
                        )

    # Macs with M1/M2 should use legacy implementation
    if platform.system() == "Darwin" and platform.processor() == "arm":
        opt = tf.keras.optimizers.legacy.Adam(learning_rate=lr_warmup_decayed_fn)
    else:
        opt = tf.keras.optimizers.Adam(learning_rate=1e-3)

    # loss functions and metric
    themodel.compile(optimizer=opt, weighted_metrics=[])

    # callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(monitor='val_mse', patience=500, restore_best_weights=True)
    # model_checkpoint_path = join('../auxiliary/', outfolder, 'model_checkpoint')
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(model_checkpoint_path, monitor="val_loss", mode="min", save_best_only=True, verbose=0)

    # Fit the model
    history = themodel.fit(
        x=train_scaled.iloc[:, :-4],
        y=train_scaled.iloc[:, -4:],
        # sample_weight=train_weights,
        batch_size=1024,
        epochs=5000,
        validation_data=(
            val_scaled.iloc[:, :-4], 
            val_scaled.iloc[:, -4:], 
            # val_weights
        ),
        use_multiprocessing=True,
        callbacks=[]
    )    
    # load the best weights
    # try:
    #     if os.path.exists(model_checkpoint_path):
    #         themodel.load_weights(model_checkpoint_path)
    #         logging.info("Loaded best model weights from checkpoint.")
    # except Error as e:
    #     logging.error(e)
    
    end_time = time.process_time_ns() - start_time
    time_string = get_time_string(end_time)

    # add metadata and save
    onnx_model, _ = tf2onnx.convert.from_keras(themodel)
    m1 = onnx_model.metadata_props.add()
    m1.key = 'model_type'
    m1.value = json.dumps('Regressor')
    m2 = onnx_model.metadata_props.add()
    m2.key = 'standardization_mean'
    m2.value = json.dumps(mean_arr.tolist())
    m3 = onnx_model.metadata_props.add()
    m3.key = 'standardization_std'
    m3.value = json.dumps(std_arr.tolist())
    m4 = onnx_model.metadata_props.add()
    m4.key = 'model_author'
    m4.value = json.dumps('Rafal Maselek')
    m5 = onnx_model.metadata_props.add()
    m5.key = 'model_license'
    m5.value = json.dumps('CC BY 4.0')
    m6 = onnx_model.metadata_props.add()
    m6.key = 'nLL_exp_mu0'
    m6.value = json.dumps(nLL_exp_mu0)
    m7 = onnx_model.metadata_props.add()
    m7.key = 'nLL_obs_mu0'
    m7.value = json.dumps(nLL_obs_mu0)
    m8 = onnx_model.metadata_props.add()
    m8.key = 'nLLA_exp_mu0'
    m8.value = json.dumps(nLLA_exp_mu0)
    m9 = onnx_model.metadata_props.add()
    m9.key = 'nLLA_obs_mu0'
    m9.value = json.dumps(nLLA_obs_mu0)
    m10 = onnx_model.metadata_props.add()
    m10.key = 'folder_name'
    m10.value = json.dumps(outfolder)
    m11 = onnx_model.metadata_props.add()
    m11.key = 'model_parameters'
    m11.value = json.dumps(parameters)
    m12 = onnx_model.metadata_props.add()
    m12.key = 'model_name'
    m12.value = json.dumps(model_name)
    m13 = onnx_model.metadata_props.add()
    m13.key = 'training_duration'
    m13.value = json.dumps(time_string)
    m14 = onnx_model.metadata_props.add()
    m14.key = 'training_date'
    m14.value = json.dumps(time.strftime( "%Y-%m-%d %H:%M:%S", time.localtime() ))


    bkg_yields = None
    try:
        metafile = join(infile.replace('.csv','.json'))
        with open(metafile, 'r') as fmeta:
            metadata_dict = json.load(fmeta)
        for key, value in metadata_dict.items():
            metadata_entry = onnx_model.metadata_props.add()
            metadata_entry.key = key
            metadata_entry.value =  json.dumps(value)
            if key == 'bkg_yields':
                bkg_yields = value
        shutil.copyfile(metafile, join(outpath_aux, 'train.json'))

    except Exception as e:
        logging.error('Could not save custom metadata because of the following error: '+str(e))

    logging.info('Dummy prediction:')
    dummy_input = np.array( np.random.normal(loc=0, scale=1, size=(5, -4+len(columns))) )
    logging.info(themodel.predict(dummy_input))
    logging.info('Dummy prediction once more:')
    logging.info(themodel.predict(dummy_input))
    if bkg_yields is not None:
        logging.info('BKG yields:')
        # some channels might be removed
        removed_channels = metadata_dict['remove_channels']
        yields = []
        for c, y in bkg_yields:
            if c.split('-')[0] not in removed_channels:
                yields.append(y)
        logging.info(yields)
        yields = (yields - mean_arr[:-4])/std_arr[:-4]
        #yields.reshape(1, len(yields))
        logging.info('NLL prediction (mu=1) on bkg yields (S=0):')
        logging.info(themodel.predict(np.array([yields]))[0]*std_arr[-4:]+mean_arr[-4:]+shift)
        logging.info(f'NLL (mu=0) values: {shift}')

    onnx_path = join(outpath_model, model_name+".onnx")
    logging.info(f"saving the model to {onnx_path}")
    onnx.save(onnx_model, onnx_path)


    for var in history.history.keys():
        if 'val' in var:
            continue
        plt.clf()
        plt.plot(history.history[var], label=var)
        plt.plot(history.history['val_' + var], label='val_'+var, alpha=0.9)
        plt.yscale('log')
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.legend()
        plt.savefig(join(outpath_aux, f'{var}.pdf'))
        fig_data = np.array([history.history[var], history.history['val_' + var]]).T
        np.savetxt(join(outpath_aux, f'{var}.txt'), fig_data, header=f'{var}\t{"val_" + var}')
    


    # print('Loaded model')
    # onnx_model_loaded = onnx.load(join(outpath_model, model_name+".onnx"))
    # sess = rt.InferenceSession(onnx_model_loaded.SerializeToString())
    # logging.info('Dummy prediction:')
    # dummy_pred = sess.run(None, {'input_1': np.array(dummy_input, dtype=np.float32)})
    
    # logging.info(dummy_pred)
    # if bkg_yields is not None:
    #     logging.info('NLL prediction (mu=1) on bkg yields (S=0):')
    #     yields = np.array([ [em[1] for em in bkg_yields], ])
    #     yields = (yields - mean_arr[:-4])/std_arr[:-4]
    #     pred = sess.run(None, {'input_1': np.array(yields, dtype=np.float32)})
    #     logging.info(pred*std_arr[-4:]+mean_arr[-4:]+shift)
    #     logging.info(f'NLL (mu=0) values: {shift}')

    # logging.info(mean_arr)




if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('python train.py <input.csv>  <model_name>')
        exit(1)
    if len(sys.argv) == 2:
        main(sys.argv[1])
    elif len(sys.argv) > 2:
         main(sys.argv[1], sys.argv[2])
