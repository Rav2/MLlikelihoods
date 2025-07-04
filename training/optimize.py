import tensorflow as tf
from tensorflow import keras
import os, sys 
import pandas as pd 
import optuna
from NNmodel import MyModelNN


def objective(trial, X_train, y_train, X_val, y_val):
    param = {
        'neurons' : trial.suggest_categorical('neurons', [16, 32, 64, 128, 256, 512, 1024, 2048]),
        'blocks' : trial.suggest_int('blocks', 2, 8),
        'l2' : trial.suggest_float('l2', 1e-5, 1e-1, log=True),   
        'activation' : trial.suggest_categorical('activation', ['relu', 'elu', 'tanh']),
        'batch_norm' : trial.suggest_categorical('batch_norm', [True, False]),    
    }

    model = MyModelNN(input_shape = (len(X_train.columns),),
                            neurons = param['neurons'], 
                            blocks = param['blocks'], 
                            l2 = param['l2'],
                            activation = param['activation'],
                            batch_norm = param['batch_norm']  
                            )
    model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=5e-3))
    history = model.fit(
            X_train[::10], y_train[::10],
            batch_size=250, epochs=100, 
            validation_data=[X_val[::10], y_val[::10]],
            verbose=0,
                )
    #del model
    return history.history['val_mape'][-1]


def optimize_params(train_scaled, val_scaled, best_trail_path, n_trials=2, n_jobs=1):
    # we dont need nLL for mu=0 so we disable eager execution for better performance 
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
        for key, value in best_trial.params.items():
            fout.write("{}:{}\n".format(key, value))
    parameters = {k:v for k,v in best_trial.params.items()}
    # we turn eager execution on again to get correct nLL for mu=0 when training the final model
    tf.config.run_functions_eagerly(True)
    return parameters
