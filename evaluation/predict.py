"""
predict.py  –  Run ONNX model on a CSV dataset and write predictions to a new CSV.

Usage:
    python predict.py <path-to-onnx-model> <path-to-test.csv> [output.csv]

Output columns:
    <all input feature columns>  (channels, copied as-is)
    nLL_exp_mu0   – from model metadata (mu=0 baseline)
    nLL_exp_mu1   – mu0 + delta predicted by NN
    nLL_obs_mu0
    nLL_obs_mu1
    nLLA_exp_mu0
    nLLA_exp_mu1
    nLLA_obs_mu0
    nLLA_obs_mu1

NOTE: mu1 = mu0 + delta_NN.  If a factor of 2 is needed, change the
      line marked "# <-- factor" below.

Input CSV can be:
    - Inputs-only  (N_channel cols)            → channels used directly
    - Full training format (N_channel + 4 or 8 target cols) → deltas computed,
      last 4 cols treated as targets and stripped before inference
"""

import os
import sys
import logging
import numpy as np
import pandas as pd
from os.path import basename, dirname, join, exists

from load_onnx import load_model_and_normalize

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s]  %(message)s  (%(filename)s:%(lineno)d)",
    handlers=[
        logging.FileHandler("predict_log.txt", mode="w"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_csv(path: str) -> pd.DataFrame:
    """Load CSV, auto-detecting separator (comma or semicolon)."""
    with open(path, "r") as f:
        header = f.readline()
    sep = ";" if header.count(";") > header.count(",") else ","
    return pd.read_csv(path, sep=sep)


def preprocess_data(dataset: str):
    """
    Load CSV and return (df_channels, df_full_or_none).

    If the file has 8 extra columns (full training format), compute deltas
    and return the full df so load_model_and_normalize can use it.
    If it's inputs-only, return (df, df) — same object for both uses.
    """
    df = load_csv(dataset)
    n_cols = len(df.columns)

    # Heuristic: if column count matches a training file (channels + 8 raw
    # likelihood cols), compute deltas and strip them.
    # Otherwise treat all columns as channel inputs.
    has_targets = (n_cols % 1 == 0)  # placeholder; real check below

    # Try to detect target columns by name (contain 'nLL' or 'LL')
    target_mask = df.columns.str.contains(r'nLL|LL_', case=False, regex=True)
    n_targets = target_mask.sum()

    if n_targets == 8:
        log.info(f"Detected 8 likelihood target columns; computing deltas.")
        # Compute deltas for nLL values (mirrors evaluate.py)
        for i in range(4):
            col_idx = -8 + 2 * i
            df.iloc[:, col_idx + 1] -= df.iloc[:, col_idx]
            df.drop(df.columns[col_idx], axis=1, inplace=True)
        # Now last 4 cols are deltas; load_model_and_normalize expects full df
        return df.iloc[:, :-4], df
    elif n_targets == 4:
        log.info(f"Detected 4 delta target columns; using as-is.")
        return df.iloc[:, :-4], df
    else:
        log.info(f"No target columns detected; treating all {n_cols} columns as channel inputs.")
        return df, df


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(model_path: str, dataset: str, output_path: str | None = None) -> None:

    # --- resolve output path -------------------------------------------------
    if output_path is None:
        stem = basename(dataset).replace(".csv", "")
        output_path = join(dirname(dataset), f"{stem}_predictions.csv")

    out_dir = dirname(output_path) or "."
    if not exists(out_dir):
        os.makedirs(out_dir)

    # --- load & preprocess ---------------------------------------------------
    log.info(f"Loading dataset: {dataset}")
    df_channels, df_full = preprocess_data(dataset)

    log.info(f"Input shape: {df_channels.shape[0]:,} rows × {df_channels.shape[1]} channels")
    log.info(f"Loading model: {model_path}")

    sess, x_norm, mean_arr, std_arr, *mu0_nLLs = load_model_and_normalize(model_path, df_full)

    # If inputs-only, x_norm has no target columns to strip
    n_target_in_norm = x_norm.shape[1] - df_channels.shape[1]
    if n_target_in_norm > 0:
        x_input = x_norm.iloc[:, :-n_target_in_norm].to_numpy().astype(np.float32)
    else:
        x_input = x_norm.to_numpy().astype(np.float32)

    # --- run inference -------------------------------------------------------
    log.info("Running inference …")
    delta_pred_norm = sess.run(None, {"input_1": x_input})[0]          # (N, 4)
    delta_pred      = delta_pred_norm * std_arr[-4:] + mean_arr[-4:]   # denormalise

    # mu0 values come from model metadata via load_model_and_normalize
    mu0 = np.array(mu0_nLLs, dtype=np.float64)   # shape (4,)
    mu1 = mu0 + delta_pred                        # shape (N, 4)  # <-- factor

    # --- assemble output table -----------------------------------------------
    # Column order: nLL_exp, nLL_obs, nLLA_exp, nLLA_obs  (matches mu0_nLLs order)
    likelihood_cols = [
        "nLL_exp_mu0",  "nLL_exp_mu1",
        "nLL_obs_mu0",  "nLL_obs_mu1",
        "nLLA_exp_mu0", "nLLA_exp_mu1",
        "nLLA_obs_mu0", "nLLA_obs_mu1",
    ]

    n = len(df_channels)
    lik_data = np.empty((n, 8), dtype=np.float64)
    for i in range(4):
        lik_data[:, 2 * i]     = mu0[i]        # mu=0 (scalar broadcast)
        lik_data[:, 2 * i + 1] = mu1[:, i]     # mu=1 (per-event)

    df_out = pd.concat(
        [df_channels.reset_index(drop=True),
         pd.DataFrame(lik_data, columns=likelihood_cols)],
        axis=1,
    )

    # --- write CSV -----------------------------------------------------------
    df_out.to_csv(output_path, index=False)
    log.info(f"Wrote {len(df_out):,} rows × {len(df_out.columns)} columns → {output_path}")


# ---------------------------------------------------------------------------

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python predict.py <model.onnx> <test.csv> [output.csv]")
        sys.exit(1)

    model_path_ = sys.argv[1]
    dataset_    = sys.argv[2]
    out_        = sys.argv[3] if len(sys.argv) > 3 else None

    main(model_path_, dataset_, out_)