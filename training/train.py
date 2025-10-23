import tensorflow as tf
import warnings
import sys
import os
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import onnx
import tf2onnx
import onnxruntime as rt
import json
import time
import gc
import shutil
import platform
import argparse
import random
from os.path import dirname, basename, join
from tensorflow import keras
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm

from NNmodel import MyModelNN
from optimize import optimize_params

# Version
__version__ = "1.0.0"

# GPU Configuration
try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], True)
except Exception as e:
    print(f'\033[93m[WARNING] Could not activate GPU acceleration: {e}\033[0m')

# Warning filters
warnings.filterwarnings(
    "ignore",
    category=UserWarning,
    module="tensorflow.python.data.ops.structured_function"
)

# Logger setup
module_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'sampling'))
sys.path.insert(0, module_path)
from misc import *

logFormatter = logging.Formatter("%(asctime)s [%(levelname)s]  %(message)s  (%(filename)s:%(lineno)d)")
fileHandler = logging.FileHandler("log.txt", mode='w')
fileHandler.setFormatter(logFormatter)
consoleHandler = logging.StreamHandler()
consoleFormatter = CustomFormatter()
consoleHandler.setFormatter(consoleFormatter)
log = logging.getLogger()
for hdlr in log.handlers[:]:
    log.removeHandler(hdlr)
log.addHandler(fileHandler)
log.addHandler(consoleHandler)
log.setLevel(logging.INFO)


# ============================================================================
# Reproducibility
# ============================================================================

def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    np.random.seed(seed)
    random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'
    logging.info(f"Random seed set to {seed} for reproducibility")


# ============================================================================
# Utility Functions
# ============================================================================
class LRLogger(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        if epoch % 50 == 0:  # Every 50 epochs
            lr = self.model.optimizer.learning_rate
            if callable(lr):
                lr = lr(self.model.optimizer.iterations)
            logging.info(f"Epoch {epoch}: LR = {float(lr):.6e}")

def get_time_string(ns_t):
    """Convert nanoseconds to human-readable time string."""
    s_t = ns_t // 10**9
    m_t = s_t // 60
    h_t = m_t // 60
    left_s = s_t % 60
    left_m = m_t - 60 * h_t
    return f"{h_t} hours {left_m} minutes {left_s} seconds"


def plot_weights(train_weights, val_weights, outpath):
    """Plot histograms of training and validation weights."""
    try:
        fig, axs = plt.subplots(2, 1, figsize=(8, 8))
        axs[0].hist(train_weights, 100)
        axs[1].hist(val_weights, 100)
        for ax, lab in zip(axs, ['train', 'val']):
            ax.set_yscale('log')
            ax.set_xlabel(f'{lab} weights')
            ax.set_ylabel('Counts')
        plt.tight_layout()
        plt.savefig(join(outpath, 'weights_hist.pdf'))
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot weights: {e}")


def plot_deltas(df, outpath, normalised):
    """Plot histograms of the last 4 columns (deltas)."""
    try:
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        for ii in range(4):
            row, col = ii // 2, ii % 2
            data = df.iloc[:, -4 + ii]
            axs[row, col].hist(data, 100)
            axs[row, col].set_yscale('log')
            axs[row, col].set_xlabel(df.columns[-4 + ii])
            axs[row, col].set_ylabel('Counts')
            axs[row, col].set_title(f'mu={np.round(data.mean(), 3)}, std={np.round(data.std(), 3)}')
        plt.tight_layout()
        filename = 'deltas_norm_hist.pdf' if normalised else 'deltas_hist.pdf'
        plt.savefig(join(outpath, filename))
        plt.close()
    except Exception as e:
        logging.error(f"Failed to plot deltas: {e}")


def create_directories(infile, model_name):
    """Create output directories for models and auxiliary files."""
    try:
        outfolder = basename(dirname(infile))
        outpath_model = os.path.abspath(join(dirname(__file__), '../models', outfolder))
        outpath_aux = os.path.abspath(join(dirname(__file__), '../auxiliary', f'{outfolder}-{model_name}'))
        
        os.makedirs(outpath_model, exist_ok=True)
        os.makedirs(outpath_aux, exist_ok=True)
        
        return outfolder, outpath_model, outpath_aux
    except Exception as e:
        logging.error(f"Failed to create directories: {e}")
        raise


def validate_dataframe(df, name="DataFrame"):
    """Validate DataFrame for NaN, inf, and other issues.
    
    Args:
        df: pandas DataFrame to validate
        name: Name of the DataFrame for logging
        
    Raises:
        ValueError: If validation fails
    """
    # Check for NaN
    nan_count = df.isna().sum().sum()
    if nan_count > 0:
        nan_cols = df.columns[df.isna().any()].tolist()
        raise ValueError(f"{name} contains {nan_count} NaN values in columns: {nan_cols}")
    
    # Check for inf
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    inf_mask = np.isinf(df[numeric_cols]).any()
    if inf_mask.any():
        inf_cols = numeric_cols[inf_mask].tolist()
        raise ValueError(f"{name} contains inf values in columns: {inf_cols}")
    
    # Check if DataFrame is empty
    if len(df) == 0:
        raise ValueError(f"{name} is empty")
    
    logging.info(f"✓ {name} validation passed: {df.shape[0]} rows, {df.shape[1]} columns")


def load_and_split_data(infile, outpath_aux):
    """Load CSV data and split into train/val/test sets."""
    try:
        logging.info("Loading data...")
        if basename(infile) == 'train.csv':
            train = pd.read_csv(infile)
            val = pd.read_csv(infile.replace('train', 'val'))
            test = pd.read_csv(infile.replace('train', 'test'))
        else:
            df = pd.read_csv(infile)
            validate_dataframe(df, "Input data")
            
            train, val, test = np.split(df.sample(frac=1), [int(.8 * len(df)), int(.9 * len(df))])
            for dataset, name in zip([train, val, test], ['train', 'val', 'test']):
                dataset.reset_index(drop=True, inplace=True)
                dataset.to_csv(join(outpath_aux, f"{name}.csv"), index=False)
        
        # Validate all datasets
        validate_dataframe(train, "Train set")
        validate_dataframe(val, "Validation set")
        validate_dataframe(test, "Test set")
        
        logging.info(f'train shape: {train.shape}, val shape: {val.shape}, test shape: {test.shape}')
        
        return train, val, test
    except Exception as e:
        logging.error(f"Failed to load and split data: {e}")
        raise


def compute_deltas(df):
    """Compute delta values (differences of nLL columns).
    
    Args:
        df: pandas DataFrame with nLL columns
        
    Returns:
        list: Mean values of the base nLL columns (shift values)
    """
    try:
        
        shift = [df.iloc[:, col].mean() for col in [-8, -6, -4, -2]]
        
        df.iloc[:, -7] = df.iloc[:, -7] - df.iloc[:, -8]
        df.iloc[:, -5] = df.iloc[:, -5] - df.iloc[:, -6]
        df.iloc[:, -3] = df.iloc[:, -3] - df.iloc[:, -4]
        df.iloc[:, -1] = df.iloc[:, -1] - df.iloc[:, -2]
        
        cols_to_drop = [df.columns[col] for col in [-8, -6, -4, -2]]
        df.drop(cols_to_drop, axis=1, inplace=True)
        df.rename(columns={ 'nLL_exp_mu1' : 'Delta_nLL_exp',
                            'nLL_obs_mu1' : 'Delta_nLL_obs',
                            'nLLA_exp_mu1' : 'Delta_nLLA_exp',
                            'nLLA_obs_mu1' : 'Delta_nLLA_obs',
                                        }, inplace=True)
        return df, shift
    except Exception as e:
        logging.error(f"Failed to compute deltas: {e}")
        raise


def normalize_data(train, val, test):
    """Normalize all datasets using StandardScaler fitted on training data."""
    try:
        columns = train.columns
        logging.info(f'Train columns: {columns}')
        
        logging.info('Train before normalization:')
        for col in columns:
            logging.info(f'{col}: mean={train[col].mean():.6f} std={train[col].std():.6f}')

        std_scaler = StandardScaler()
        
        logging.info("Normalizing data...")
        train_scaled = pd.DataFrame(
            std_scaler.fit_transform(train), 
            columns=columns
        )
        val_scaled = pd.DataFrame(
            std_scaler.transform(val), 
            columns=columns
        )
        test_scaled = pd.DataFrame(
            std_scaler.transform(test), 
            columns=columns
        )
        
        mean_arr = std_scaler.mean_
        std_arr = np.sqrt(std_scaler.var_)
        
        # Validate normalized data
        validate_dataframe(train_scaled, "Normalized train set")
        validate_dataframe(val_scaled, "Normalized validation set")
        validate_dataframe(test_scaled, "Normalized test set")
        
        logging.info('Train after normalization:')
        for col in columns:
            logging.info(f'{col}: mean={train_scaled[col].mean():.6f} std={train_scaled[col].std():.6f}')
        
        return train_scaled, val_scaled, test_scaled, mean_arr, std_arr, columns
    except Exception as e:
        logging.error(f"Failed to normalize data: {e}")
        raise
    finally:
        # Clean up memory
        gc.collect()


def get_or_optimize_parameters(outfolder, model_name, train_scaled, val_scaled, optimize=False):
    """Load existing parameters or run optimization."""
    try:
        best_trial_path = join(dirname(__file__), '../auxiliary', f'{outfolder}-{model_name}', 'best_trial.txt')
        
        if optimize:
            if not os.path.exists(best_trial_path):
                parameters = optimize_params(train_scaled, val_scaled, best_trial_path, n_trials=100, n_jobs=5)
            else:
                logging.warning('Using existing parameters.')
                with open(best_trial_path, 'r') as fin:
                    parameters = json.load(fin)
        else:
            parameters = default_parameters
            with open(best_trial_path, 'w') as fpar:
                json.dump(parameters, fpar)
        
        return parameters
    except Exception as e:
        logging.error(f"Failed to get/optimize parameters: {e}")
        raise


def create_cosine_scheduler(initial_lr, peak_lr, final_lr, warmup_epochs, total_epochs, steps_per_epoch):
    """Create cosine learning rate scheduler with warmup and decay.
    
    Args:
        initial_lr: Starting learning rate (1e-5)
        peak_lr: Peak learning rate after warmup (1e-3)
        final_lr: Final learning rate after decay (1e-4)
        warmup_epochs: Number of epochs for warmup (100)
        total_epochs: Total training epochs (8100 = 100 warmup + 8000 decay)
        steps_per_epoch: Number of batches per epoch
        
    Returns:
        Learning rate schedule
    """
    warmup_steps = warmup_epochs * steps_per_epoch
    total_steps = total_epochs * steps_per_epoch
    
    return keras.optimizers.schedules.CosineDecay(
        initial_learning_rate=initial_lr,
        decay_steps=total_steps,
        warmup_target=peak_lr,
        warmup_steps=warmup_steps,
        alpha=final_lr / peak_lr
    )


def build_and_compile_model(parameters, columns, batch_size, train_size, loss_name = 'MSE'):
    """Build and compile the neural network model."""
    try:
        # Calculate steps per epoch
        steps_per_epoch = int(np.ceil(train_size / batch_size))
        
        # Create cosine scheduler: 1e-5 -> 1e-3 (100 epochs) -> 1e-4 (8000 epochs)
        lr_schedule = create_cosine_scheduler(
            initial_lr=1e-5,
            peak_lr=1e-3,
            final_lr=1e-5,
            warmup_epochs=100,
            total_epochs=9000,
            steps_per_epoch=steps_per_epoch
        )
        
        # Create model
        model = MyModelNN(
            input_shape=(len(columns) - 4,),
            neurons=parameters['neurons'],
            blocks=parameters['blocks'],
            l2=parameters['l2'],
            activation=parameters['activation'],
            loss=loss_name,
            output_size=4,
            dropout_rate = parameters['dropout_rate'],
            use_residual = parameters['use_residual'],
            width = parameters['width'],
        )
        
        # Create optimizer (handle Mac M1/M2 compatibility)
        if platform.system() == "Darwin" and platform.processor() == "arm":
            optimizer = tf.keras.optimizers.legacy.Adam(learning_rate=lr_schedule)
        else:
            optimizer = tf.keras.optimizers.Adam(learning_rate=lr_schedule)
        
        model.compile(optimizer=optimizer, weighted_metrics=[])
        
        logging.info(f"✓ Model compiled successfully")
        logging.info(f"  Optimizer: {optimizer.__class__.__name__}")
        logging.info(f"  Loss: {loss_name}")
        logging.info(f"  Steps per epoch: {steps_per_epoch}")
        
        return model, optimizer
    except Exception as e:
        logging.error(f"Failed to build and compile model: {e}")
        raise


def train_model(model, train_scaled, val_scaled, batch_size, epochs, use_early_stopping=True):
    """Train the model with optional early stopping."""
    try:
        callbacks = [LRLogger()]
        
        if use_early_stopping:
            early_stop = tf.keras.callbacks.EarlyStopping(
                monitor='loss',
                patience=500,
                restore_best_weights=True,
                verbose=1
            )
            callbacks.append(early_stop)
            logging.info("✓ Early stopping enabled (patience=500, monitor=val_loss)")
        
        logging.info(f"\nStarting training...")
        logging.info(f"  Epochs: {epochs}")
        logging.info(f"  Batch size: {batch_size}")
        logging.info(f"  Training samples: {len(train_scaled)}")
        logging.info(f"  Validation samples: {len(val_scaled)}\n")
        
        history = model.fit(
            x=train_scaled.iloc[:, :-4],
            y=train_scaled.iloc[:, -4:],
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(val_scaled.iloc[:, :-4], val_scaled.iloc[:, -4:]),
            use_multiprocessing=False,
            callbacks=callbacks,
            verbose=1
        )
        
        logging.info("\n✓ Training completed")
        if use_early_stopping and len(history.history['loss']) < epochs:
            logging.info(f"  Early stopping triggered at epoch {len(history.history['loss'])}")
            logging.info(f"  Best model restored from epoch {len(history.history['loss']) - 500}")
        
        return history
    except Exception as e:
        logging.error(f"Failed during training: {e}")
        raise
    finally:
        # Clean up memory
        gc.collect()


def add_metadata_to_onnx(onnx_model, metadata_dict):
    """Add metadata properties to ONNX model."""
    for key, value in metadata_dict.items():
        meta = onnx_model.metadata_props.add()
        meta.key = key
        meta.value = json.dumps(value)


def save_onnx_model(model, outpath_model, model_name, mean_arr, std_arr, shift, 
                    outfolder, parameters, time_string, infile, outpath_aux,
                    optimizer_name, loss_name, batch_size, early_stopping_used):
    """Convert model to ONNX and save with metadata."""
    try:
        logging.info("\nConverting model to ONNX format...")
        onnx_model, _ = tf2onnx.convert.from_keras(model)
        
        # Core metadata
        metadata = {
            'model_type': 'Regressor',
            'model_version': __version__,
            'standardization_mean': mean_arr.tolist(),
            'standardization_std': std_arr.tolist(),
            'model_author': 'Rafal Maselek',
            'model_license': 'CC BY 4.0',
            'nLL_exp_mu0': shift[0],
            'nLL_obs_mu0': shift[1],
            'nLLA_exp_mu0': shift[2],
            'nLLA_obs_mu0': shift[3],
            'folder_name': outfolder,
            'model_parameters': parameters,
            'model_name': model_name,
            'training_duration': time_string,
            'training_date': time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            'optimizer': optimizer_name,
            'loss_function': loss_name,
            'batch_size': batch_size,
            'early_stopping_used': early_stopping_used,
        }
        
        add_metadata_to_onnx(onnx_model, metadata)
        
        # Load and add custom metadata from JSON file
        bkg_yields = None
        try:
            metafile = infile.replace('.csv', '.json')
            with open(metafile, 'r') as fmeta:
                custom_metadata = json.load(fmeta)
            add_metadata_to_onnx(onnx_model, custom_metadata)
            bkg_yields = custom_metadata.get('bkg_yields')
            shutil.copyfile(metafile, join(outpath_aux, 'train.json'))
        except Exception as e:
            logging.warning(f'Could not load custom metadata: {str(e)}')
        
        # Save ONNX model
        onnx_path = join(outpath_model, f"{model_name}.onnx")
        onnx.save(onnx_model, onnx_path)
        logging.info(f"✓ Model saved to {onnx_path}")
        
        return bkg_yields, onnx_path
    except Exception as e:
        logging.error(f"Failed to save ONNX model: {e}")
        raise
    finally:
        # Clean up memory
        del onnx_model
        gc.collect()


def test_onnx_model(onnx_path, mean_arr, std_arr, columns):
    """Test loading and inference with the saved ONNX model."""
    try:
        logging.info("\n" + "="*60)
        logging.info("Testing ONNX Model")
        logging.info("="*60)
        
        # Load ONNX model
        onnx_model_loaded = onnx.load(onnx_path)
        sess = rt.InferenceSession(onnx_model_loaded.SerializeToString())
        
        # Create dummy input
        dummy_input = np.random.normal(loc=0, scale=1, size=(5, len(columns) - 4)).astype(np.float32)
        
        # Run inference
        input_name = sess.get_inputs()[0].name
        output_name = sess.get_outputs()[0].name
        predictions = sess.run([output_name], {input_name: dummy_input})[0]
        
        logging.info(f"✓ ONNX model loaded successfully from: {onnx_path}")
        logging.info(f"  Input shape: {dummy_input.shape}")
        logging.info(f"  Output shape: {predictions.shape}")
        logging.info(f"  Sample predictions:\n{predictions}")
        logging.info("="*60 + "\n")
        
        return predictions
    except Exception as e:
        logging.error(f"Failed to test ONNX model: {e}")
        raise
    finally:
        # Clean up memory
        del onnx_model_loaded
        del sess
        gc.collect()


def make_predictions(model, mean_arr, std_arr, columns, shift, bkg_yields, metadata_dict):
    """Make dummy predictions and predictions on background yields."""
    try:
        logging.info('\n' + '='*60)
        logging.info('Model Predictions')
        logging.info('='*60)
        
        # Dummy prediction
        dummy_input = np.random.normal(loc=0, scale=1, size=(5, len(columns) - 4))
        logging.info('Dummy prediction:')
        predictions = model.predict(dummy_input, verbose=0)
        logging.info(predictions)
        
        # Background yields prediction
        if bkg_yields is not None:
            logging.info('\nBKG yields:')
            removed_channels = metadata_dict.get('remove_channels', [])
            yields = [y for c, y in bkg_yields if c.split('-')[0] not in removed_channels]
            logging.info(yields)
            
            yields_normalized = (np.array(yields) - mean_arr[:-4]) / std_arr[:-4]
            prediction = model.predict(np.array([yields_normalized]), verbose=0)[0]
            prediction_denorm = prediction * std_arr[-4:] + mean_arr[-4:] + shift
            
            logging.info('NLL prediction (mu=1) on bkg yields (S=0):')
            logging.info(prediction_denorm)
            logging.info(f'NLL (mu=0) values: {shift}')
        
        logging.info('='*60 + '\n')
    except Exception as e:
        logging.error(f"Failed to make predictions: {e}")
        raise


def plot_training_history(history, outpath_aux):
    """Plot and save training history."""
    try:
        logging.info("Plotting training history...")
        for var in tqdm(list(history.history.keys()), desc="Saving plots"):
            if 'val' in var:
                continue
            
            plt.figure(figsize=(10, 6))
            plt.plot(history.history[var], label=var)
            plt.plot(history.history['val_' + var], label='val_' + var, alpha=0.9)
            plt.yscale('log')
            plt.xlabel('Epoch')
            plt.ylabel('Value')
            plt.legend()
            plt.title(f'Training History: {var}')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            plt.savefig(join(outpath_aux, f'{var}.pdf'))
            plt.close()
            
            # Save data
            fig_data = np.array([history.history[var], history.history['val_' + var]]).T
            np.savetxt(join(outpath_aux, f'{var}.txt'), fig_data, header=f'{var}\t{"val_" + var}')
        
        logging.info("✓ Training history plots saved")
    except Exception as e:
        logging.error(f"Failed to plot training history: {e}")
    finally:
        # Clean up memory
        plt.close('all')
        gc.collect()


# ============================================================================
# Main Training Pipeline
# ============================================================================

def main(args):
    """Main training pipeline."""
    start_time = time.process_time_ns()
    
    try:
        # Set seed for reproducibility
        set_seed(args.seed)
        
        # Setup
        outfolder, outpath_model, outpath_aux = create_directories(args.input, args.model_name)
        
        # Load data
        train, val, test = load_and_split_data(args.input, outpath_aux)
        
        # Compute deltas
        train, shift = compute_deltas(train)
        val, _ = compute_deltas(val)
        test, _ = compute_deltas(test)
        
        logging.info(f'After delta computation:')
        logging.info(f'train shape: {train.shape}, val shape: {val.shape}, test shape: {test.shape}')
        
        plot_deltas(train, outpath_aux, False)
        
        # Normalize data
        train_scaled, val_scaled, test_scaled, mean_arr, std_arr, columns = normalize_data(train, val, test)
        plot_deltas(train_scaled, outpath_aux, True)
        
        # Get/optimize parameters
        parameters = get_or_optimize_parameters(outfolder, args.model_name, train_scaled, val_scaled, 
                                               optimize=args.optimize)
        logging.info(f'NN parameters: {parameters}')
        
        gc.collect()
        
        loss_name = 'hybrid'

        # Build and compile model
        model, optimizer = build_and_compile_model(parameters, columns, args.batch_size, len(train_scaled), loss_name)
        
        # Train model
        history = train_model(model, train_scaled, val_scaled, args.batch_size, args.epochs,
                            use_early_stopping=args.early_stopping)
        
        # Calculate training time
        end_time = time.process_time_ns() - start_time
        time_string = get_time_string(end_time)
        logging.info(f'\n✓ Training completed in {time_string}')
        
        # Save ONNX model with metadata
        optimizer_name = optimizer.__class__.__name__
        
        bkg_yields, onnx_path = save_onnx_model(
            model, outpath_model, args.model_name, mean_arr, std_arr, shift,
            outfolder, parameters, time_string, args.input, outpath_aux,
            optimizer_name, loss_name, args.batch_size, args.early_stopping
        )
        
        # Make predictions
        try:
            metafile = args.input.replace('.csv', '.json')
            with open(metafile, 'r') as fmeta:
                metadata_dict = json.load(fmeta)
        except:
            metadata_dict = {}
        
        make_predictions(model, mean_arr, std_arr, columns, shift, bkg_yields, metadata_dict)
        
        # Test ONNX model
        test_onnx_model(onnx_path, mean_arr, std_arr, columns)
        
        # Plot training history
        plot_training_history(history, outpath_aux)
        
        logging.info('\n' + '='*60)
        logging.info('Training pipeline completed successfully!')
        logging.info('='*60)
        
    except Exception as e:
        logging.error(f"\n{'='*60}")
        logging.error(f"FATAL ERROR: Training pipeline failed")
        logging.error(f"{'='*60}")
        logging.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
    finally:
        # Final memory cleanup
        gc.collect()


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description='Train a neural network model for regression tasks.',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    parser.add_argument(
        'input',
        type=str,
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '--model-name',
        type=str,
        default='model',
        help='Name of the model to save'
    )
    
    parser.add_argument(
        '--batch-size',
        type=int,
        default=1024,
        help='Batch size for training'
    )
    
    parser.add_argument(
        '--epochs',
        type=int,
        default=10000,
        help='Maximum number of epochs to train'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=int(time.time()),
        help='Random seed for reproducibility'
    )
    
    parser.add_argument(
        '--optimize',
        action='store_true',
        help='Run hyperparameter optimization'
    )
    
    parser.add_argument(
        '--no-early-stopping',
        dest='early_stopping',
        action='store_false',
        help='Disable early stopping'
    )
    
    parser.add_argument(
        '--version',
        action='version',
        version=f'%(prog)s {__version__}'
    )
    
    parser.set_defaults(early_stopping=True)
    
    return parser.parse_args()

# ==========================
# DEFAULT PARAMETERS
# ==========================

default_parameters ={
                'neurons': 2048,
                'blocks': 5,
                'l2': 1e-6,
                'activation': 'elu',
                'batch_norm': True,
                'dropout_rate': 0.0,
                'use_residual' : False,
                'width' : 'equal',
            }

if __name__ == '__main__':
    args = parse_arguments()
    main(args)
