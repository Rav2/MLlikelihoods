import tensorflow as tf
from tensorflow import keras
import os, sys 
import pandas as pd 
import optuna
from NNmodel import MyModelNN
import numpy as np
import gc

def objective(trial, X_train, y_train, X_val, y_val):
    try:
        param = {
            'neurons' : trial.suggest_categorical('neurons', [32, 64, 128, 256, 512, 1024, 2048,]),
            'blocks' : trial.suggest_int('blocks', 3, 6),
            'l2' : trial.suggest_float('l2', 1e-7, 1e-1, log=True),   
            'activation' : trial.suggest_categorical('activation', ['relu', 'elu', 'tanh', 'relu6', 'swish']),
            'width' : trial.suggest_categorical('width', ['equal', ]),
            'batch_norm' : trial.suggest_categorical('batch_norm', [True, False]),
            'use_residual' : trial.suggest_categorical('use_residual', [True, False]),
            'dropout_rate' : trial.suggest_float('dropout_rate', 0.0, 0.15)
        }

        model = MyModelNN(input_shape = (len(X_train.columns),),
                                neurons = param['neurons'], 
                                blocks = param['blocks'], 
                                l2 = param['l2'],
                                activation = param['activation'],
                                batch_norm = param['batch_norm'],
                                dropout_rate = param['dropout_rate'],
                                use_residual = param['use_residual'],
                                width = param['width'],  
                                )
        model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=1e-3, ))
        X_train_subset = X_train[::10].values.astype(np.float32)
        y_train_subset = y_train[::10].values.astype(np.float32)
        X_val_subset = X_val[::10].values.astype(np.float32)
        y_val_subset = y_val[::10].values.astype(np.float32)
        
        history = model.fit(
            X_train_subset, y_train_subset,
            batch_size=256,
            epochs=50,
            validation_data=(X_val_subset, y_val_subset),
            verbose=0,
        )       
        return history.history['val_mape'][-1]
    except (tf.errors.ResourceExhaustedError, tf.errors.InternalError) as e:
        print(f"[WARNING] Trial {trial.number} failed with GPU error: {type(e).__name__}")
        print(f"         Params: neurons={param['neurons']}, blocks={param['blocks']}, width={param['width']}")
        
        # Aggressive cleanup
        try:
            del model
        except:
            pass
        try:
            del history
        except:
            pass
        
        tf.keras.backend.clear_session()
        
        # Reset GPU state
        try:
            physical_devices = tf.config.list_physical_devices('GPU')
            if physical_devices:
                tf.config.experimental.set_memory_growth(physical_devices[0], True)
        except:
            pass
        
        gc.collect()
        
        # Return a high value to mark this trial as bad
        return float('inf')
        
    except Exception as e:
        print(f"[ERROR] Trial {trial.number} failed with unexpected error: {type(e).__name__}: {e}")
        
        # Cleanup
        try:
            del model
        except:
            pass
        try:
            del history
        except:
            pass
            
        tf.keras.backend.clear_session()
        gc.collect()
        
        # For unexpected errors, we might want to see them
        raise

def optimize_params(train_scaled, val_scaled, best_trail_path, n_trials=2, n_jobs=1):
    # we dont need nLL for mu=0 so we disable eager execution for better performance 
    physical_devices = tf.config.list_physical_devices('GPU')
    if physical_devices:
        gpu_details = tf.config.experimental.get_device_details(physical_devices[0])
    print(f"[INFO] GPU Device: {physical_devices[0].name}")

    tf.config.run_functions_eagerly(False)
    study = optuna.create_study(direction='minimize')
    study.optimize(lambda trial: objective(trial, 
                                        train_scaled.iloc[:,:-4],
                                        train_scaled.iloc[:, -4:],
                                        val_scaled.iloc[:,:-4], 
                                        val_scaled.iloc[:, -4:]), 
                                        n_trials=n_trials, n_jobs=n_jobs)
    print("[INFO] Number of finished trials: ", len(study.trials))
    print("[INFO] Best trial:")
    best_trial = study.best_trial
    print("  Value: {}".format(best_trial.value))
    print("  Params: ")
    for key, value in best_trial.params.items():
        print("    {}: {}".format(key, value))
    with open(best_trail_path, 'w') as fout:
        fout.write("{\n")
        for key, value in best_trial.params.items():
            fout.write('"{}":{},\n'.format(key, value))
        fout.write("}")
    parameters = {k:v for k,v in best_trial.params.items()}
    # we turn eager execution on again to get correct nLL for mu=0 when training the final model
    tf.config.run_functions_eagerly(True)
    return parameters
