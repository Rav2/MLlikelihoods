#!/usr/bin/env python3
import os
import sys
import json
import argparse
import logging
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib as mpl
from matplotlib import pyplot as plt
from scipy.interpolate import griddata

# optional sklearn imports (handled lazily)
try:
    from sklearn.decomposition import PCA
    from sklearn.manifold import TSNE
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# Input validation
# ------------------------------------------------------------


def validate_inputs(file_path: str) -> str:
    """Validate the input file path and ensure required files exist."""
    logger.info(f"Validating inputs for file: {file_path}")

    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"CSV file not found: {file_path}")
    if not file_path.endswith(".csv"):
        raise ValueError("Input file must be a CSV file.")

    json_file = file_path.replace("csv", "json")
    if not os.path.isfile(json_file):
        raise FileNotFoundError(f"JSON metadata file not found: {json_file}")

    logger.info(f"Validation successful. JSON file: {json_file}")
    return json_file


# ------------------------------------------------------------
# Data loading
# ------------------------------------------------------------


def load_data(file_path: str):
    """Load dataframe and JSON metadata."""
    logger.info(f"Loading CSV file: {file_path}")
    try:
        df = pd.read_csv(file_path)
    except Exception:
        logger.exception("Failed to read CSV file.")
        raise

    json_path = file_path.replace("csv", "json")
    logger.info(f"Loading JSON metadata: {json_path}")
    try:
        with open(json_path, "r") as f:
            metadata = json.load(f)
    except json.JSONDecodeError:
        logger.exception("Malformed JSON metadata file.")
        raise
    except Exception:
        logger.exception("Failed to open JSON metadata file.")
        raise

    return df, metadata


# ------------------------------------------------------------
# Metadata consistency validation (Recommendation D)
# ------------------------------------------------------------


def validate_metadata(metadata: dict):
    """Validate metadata structure and determine which channels set is active.

    Returns:
        index (int): selected channels set index in metadata["channels"]
    """
    required_keys = ["channels", "lower_limits", "upper_limits", "obs_yields", "remove_channels"]
    for key in required_keys:
        if key not in metadata:
            raise ValueError(f"Metadata missing required key: {key}")

    # ---- PATCHSET SELECTION ----------------------------------------------------
    if "patchsets" not in metadata:
        raise ValueError("Metadata must contain 'patchsets' to determine active channel set.")

    patchsets = metadata["patchsets"]
    if not isinstance(patchsets, list):
        # normalize one-entry case
        patchsets = [patchsets]

    patchset_names = []
    patchset_active = []

    logger.info("Patchsets found:")
    for ie, entry in enumerate(patchsets):
        # entry may be: ["Name", 0/1] or "Name" (single)
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            name, applied = entry
            patchset_names.append(str(name))
            patchset_active.append(bool(applied))
            logger.info(f"    [{'X' if applied else ' '}] {name}")
        else:
            # treat as a single named patchset (mark as active)
            name = entry
            patchset_names.append(str(name))
            patchset_active.append(True)
            logger.info(f"    [X] {name}")

    n_active = int(np.sum(patchset_active))

    # Case 1 – exactly one active → automatically select it
    if n_active == 1:
        index = int(patchset_active.index(True))
        logger.info(f"Active patchset detected: {patchset_names[index]}")

    # Case 2 – none active but only one available → trivial
    elif n_active == 0 and len(patchsets) == 1:
        logger.info("No active patchset specified, but only one exists → selecting it.")
        index = 0

    # Case 3 – none active and multiple choices → ask user
    elif n_active == 0 and len(patchsets) > 1:
        logger.warning("No active patchset specified. User input required.")
        print("\nPlease select which patchset to plot:")
        for i, name in enumerate(patchset_names):
            print(f"  [{i}] {name}")

        try:
            index = int(input("Enter patchset number: "))
        except Exception:
            raise ValueError("Invalid input. Expected integer index.")

        if not (0 <= index < len(patchset_names)):
            raise ValueError(f"Selected patchset index {index} is out of range.")

        logger.info(f"User selected patchset: {patchset_names[index]}")

    # Case 4 – more than one active → ambiguous
    else:
        raise ValueError(
            "Metadata has more than one active patchset. Only one can be active at a time."
        )

    # Attach selected index to metadata for later use
    metadata["selected_patchset"] = index

    # ---- CONSISTENCY CHECKS ----------------------------------------------------
    try:
        channels_dict = metadata["channels"][index]
    except Exception:
        raise IndexError(f"Selected index {index} out of range for metadata['channels']")

    n_channels = len(channels_dict)
    ll = len(metadata["lower_limits"])
    ul = len(metadata["upper_limits"])
    oy = len(metadata["obs_yields"])

    logger.info(f"Using channel set: {patchset_names[index]}")
    logger.info(f"Channels: {n_channels}, lower={ll}, upper={ul}, observed={oy}")

    if not (ll == ul == oy == n_channels):
        raise ValueError(
            f"Metadata length mismatch for selected channel set.\n"
            f"channels={n_channels}, lower_limits={ll}, upper_limits={ul}, obs_yields={oy}"
        )

    logger.info("Metadata validation passed.")
    return index


# ------------------------------------------------------------
# Plotting functions with robust exception handling
# ------------------------------------------------------------


def create_violin_plots(df, labels, output_dir):
    logger.info("Creating violin plots...")
    for cols, label_name in labels:
        if not cols:
            logger.info(f"No columns for violin plot: {label_name}")
            continue

        logger.info(f"Generating violin plot for: {label_name}")

        try:
            plt.figure(figsize=(15, 6))
            sns.violinplot(data=df[cols])
            plt.ylabel("Yields")
            plt.xticks(rotation=30)
            plt.tight_layout()

            out_path = os.path.join(output_dir, f"violin_{label_name}.pdf")
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved violin plot: {out_path}")

        except Exception:
            logger.exception(f"Error creating violin plot for {label_name}")
            raise


def create_pairplots(df, lower_limits, upper_limits, observed, labels, output_dir):
    logger.info("Creating pair plots...")

    for cols, label_name in labels:
        if not cols:
            logger.info(f"No columns for pair plot: {label_name}")
            continue

        logger.info(f"Generating pair plot for: {label_name}")

        try:
            plt.close()
            nbins = 30
            pgrid = sns.pairplot(
                df[cols], corner=True, kind="hist",
                plot_kws={"bins": nbins},
                diag_kws={"bins": nbins}
            )

            column_indices = [df.columns.get_loc(c) for c in cols]

            for idx_row, col_idx in enumerate(column_indices):
                low_lim = lower_limits[col_idx]
                up_lim = upper_limits[col_idx]
                obs_val = observed[col_idx][1]

                # Vertical line on diagonal
                pgrid.axes[idx_row, idx_row].axvline(obs_val, color="orange", linestyle="--")

                for idx_col, col_idx_j in enumerate(column_indices):
                    if idx_col < idx_row:
                        continue

                    ax = pgrid.axes[idx_col, idx_row]
                    ax.axvline(up_lim, color="green", linestyle="--")
                    ax.axvline(low_lim, color="red", linestyle="--")

                    if idx_row != idx_col:
                        up_lim_j = upper_limits[col_idx_j]
                        low_lim_j = lower_limits[col_idx_j]

                        ax.axhline(up_lim_j, color="green", linestyle="--")
                        ax.axhline(low_lim_j, color="red", linestyle="--")

                        ax.scatter(obs_val, observed[col_idx_j][1], color="orange",
                                   marker="*", s=30)

            plt.tight_layout()
            out_path = os.path.join(output_dir, f"pairplot_{label_name}.pdf")
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved pair plot: {out_path}")

        except Exception:
            logger.exception(f"Error creating pair plot for {label_name}")
            raise


def create_lratio_histograms(df, num_yield_cols, output_dir):
    """Create likelihood-ratio histogram distributions.
    
    Args:
        df: DataFrame with yields followed by 8 -logL columns
        num_yield_cols: Number of yield columns (SR + CR channels)
        output_dir: Output directory for plots
    """
    logger.info("Creating likelihood-ratio histograms...")
    
    try:
        if df.shape[1] < num_yield_cols + 8:
            logger.warning(f"DataFrame has {df.shape[1]} columns but expected at least {num_yield_cols + 8}. Skipping LR histograms.")
            return
        
        # Define the 4 likelihood ratio pairs
        labels = [
            (r"$-2\Delta\log L$ (exp)", num_yield_cols, num_yield_cols + 1),
            (r"$-2\Delta\log L$ (obs)", num_yield_cols + 2, num_yield_cols + 3),
            (r"$-2\Delta\log L$ (exp, Asimov)", num_yield_cols + 4, num_yield_cols + 5),
            (r"$-2\Delta\log L$ (obs, Asimov)", num_yield_cols + 6, num_yield_cols + 7),
        ]
        
        nbins = 60
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        for i, (label_name, idx_mu0, idx_mu1) in enumerate(labels):
            try:
                # Get the two columns
                vals_mu0 = pd.to_numeric(df.iloc[:, idx_mu0], errors="coerce")
                vals_mu1 = pd.to_numeric(df.iloc[:, idx_mu1], errors="coerce")
                
                # Compute -2*Delta logL for each toy: 2*(L_mu1 - L_mu0)
                # Since we have -logL, this is 2*(-logL_mu1 - (-logL_mu0)) = 2*(logL_mu0 - logL_mu1)
                # Wait, let me reconsider: if columns are -logL, then L = exp(-col)
                # We want -2*log(L_mu1/L_mu0) = -2*(logL_mu1 - logL_mu0) = -2*(-col_mu1 - (-col_mu0)) = 2*(col_mu1 - col_mu0)
                # Actually simpler: we want 2*delta(-logL) = 2*((-logL_mu1) - (-logL_mu0))
                max_val = np.nanmin(vals_mu0)
                lr_vals = 2 * (vals_mu1 - max_val)
                
                # Plot histogram
                valid_vals = lr_vals.dropna()
                if len(valid_vals) == 0:
                    logger.warning(f"No valid values for {label_name}")
                    continue
                    
                axs[i].hist(valid_vals, bins=nbins, histtype="stepfilled", alpha=0.7, lw=2, label=label_name)
                axs[i].set_yscale("log")
                axs[i].set_xlabel(label_name)
                axs[i].set_ylabel("Counts")
                axs[i].legend()
                axs[i].grid(True, alpha=0.3)
                
                # Log statistics
                median = np.median(valid_vals)
                mean = np.mean(valid_vals)
                logger.info(f"{label_name}: mean={mean:.3f}, median={median:.3f}, std={np.std(valid_vals):.3f}")
                
            except Exception:
                logger.exception(f"Error processing {label_name}")
                continue
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, "L_ratio_histograms.pdf")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved likelihood-ratio histograms: {out_path}")
        
    except Exception:
        logger.exception("Error creating likelihood-ratio histograms.")
        raise

def create_nlls_histograms(df, num_yield_cols, output_dir):
    """Create nLLS (negative log-likelihood sum) histogram distributions.
    
    Args:
        df: DataFrame with yields followed by 8 -logL columns
        num_yield_cols: Number of yield columns (SR + CR channels)
        output_dir: Output directory for plots
    """
    logger.info("Creating nLLS histograms...")
    
    try:
        if df.shape[1] < num_yield_cols + 8:
            logger.warning(f"DataFrame has {df.shape[1]} columns but expected at least {num_yield_cols + 8}. Skipping nLLS histograms.")
            return
        
        # Define the 4 nLLS pairs (sum of both -logL values)
        labels = [
            (r"nLLS (exp)", num_yield_cols, num_yield_cols + 1),
            (r"nLLS (obs)", num_yield_cols + 2, num_yield_cols + 3),
            (r"nLLS (exp, Asimov)", num_yield_cols + 4, num_yield_cols + 5),
            (r"nLLS (obs, Asimov)", num_yield_cols + 6, num_yield_cols + 7),
        ]
        
        nbins = 60
        fig, axs = plt.subplots(2, 2, figsize=(12, 10))
        axs = axs.flatten()
        
        for i, (label_name, idx_mu0, idx_mu1) in enumerate(labels):
            try:
                # Get the two columns
                vals_mu1 = pd.to_numeric(df.iloc[:, idx_mu1], errors="coerce")
                
                nlls_vals = vals_mu1
                
                # Plot histogram
                valid_vals = nlls_vals.dropna()
                if len(valid_vals) == 0:
                    logger.warning(f"No valid values for {label_name}")
                    continue
                    
                axs[i].hist(valid_vals, bins=nbins, histtype="stepfilled", alpha=0.7, lw=2, label=label_name)
                axs[i].set_yscale("log")
                axs[i].set_xlabel(label_name)
                axs[i].set_ylabel("Counts")
                axs[i].legend()
                axs[i].grid(True, alpha=0.3)
                
                # Log statistics
                median = np.median(valid_vals)
                mean = np.mean(valid_vals)
                logger.info(f"{label_name}: mean={mean:.3f}, median={median:.3f}, std={np.std(valid_vals):.3f}")
                
            except Exception:
                logger.exception(f"Error processing {label_name}")
                continue
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, "nLLS_histograms.pdf")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved nLLS histograms: {out_path}")
        
    except Exception:
        logger.exception("Error creating nLLS histograms.")
        raise

# -----------------------------
# New diagnostic plots
# -----------------------------


def create_outlier_plots(df, selected_columns, output_dir, z_threshold=3.0):
    """Create outlier detection visualizations:
      - Z-score histogram (flattened across selected columns)
    """
    logger.info("Creating outlier detection plots...")
    if not selected_columns:
        logger.info("No columns selected for outlier detection.")
        return

    try:
        # Prepare data: drop columns not present
        cols = [c for c in selected_columns if c in df.columns]
        if not cols:
            logger.warning("None of the selected columns are present in the dataframe.")
            return

        # Z-scores and histogram
        flattened_z = []

        for c in cols:
            series = pd.to_numeric(df[c], errors="coerce").dropna()
            if series.empty:
                continue

            mean = series.mean()
            std = series.std(ddof=0)
            if std == 0:
                z = np.zeros_like(series)
            else:
                z = (series - mean) / std

            flattened_z.extend(z.tolist())

        # histogram of absolute z-scores (flattened)
        plt.figure(figsize=(8, 5))
        plt.hist(np.abs(flattened_z), bins=60)
        plt.yscale("log")
        plt.xlabel(f"|z| (flattened over {len(cols)} channels)")
        plt.ylabel("Counts (log scale)")
        plt.tight_layout()
        out_path = os.path.join(output_dir, "outliers_zscore_histogram.pdf")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved z-score histogram: {out_path}")

    except Exception:
        logger.exception("Error creating outlier diagnostics.")
        raise


def create_channel_importance_plot(df, SR_cols, CR_cols, lower_limits, upper_limits, observed, output_dir):
    """Create channel importance ranking based on contribution to sensitivity.
    
    Ranks channels by their relative deviation from expected, weighted by uncertainty.
    """
    logger.info("Creating channel importance ranking plot...")
    
    try:
        all_cols = SR_cols + CR_cols
        if not all_cols:
            logger.warning("No channels to rank.")
            return
        
        channel_names = []
        importance_scores = []
        channel_types = []
        
        for col in all_cols:
            # Get column index in dataframe
            col_idx = df.columns.get_loc(col)
            
            # Extract channel name (before the '-' if present)
            ch_name = col.split('-')[0] if '-' in col else col
            
            # Get limits and observed value
            lower = lower_limits[col_idx]
            upper = upper_limits[col_idx]
            obs = observed[col_idx][1]
            
            # Compute mean yield from toy data
            mean_yield = df[col].mean()
            
            # Compute importance: sensitivity-weighted deviation
            # Use relative deviation normalized by uncertainty range
            uncertainty = (upper - lower) / 2.0
            if uncertainty > 0:
                # Importance = |mean_toy - observed| / uncertainty
                importance = abs(mean_yield - obs) / uncertainty
            else:
                importance = 0.0
            
            channel_names.append(ch_name)
            importance_scores.append(importance)
            channel_types.append('SR' if col in SR_cols else 'CR')
        
        # Create DataFrame for sorting
        importance_df = pd.DataFrame({
            'channel': channel_names,
            'importance': importance_scores,
            'type': channel_types
        })
        
        # Sort by importance
        importance_df = importance_df.sort_values('importance', ascending=True)
        
        # Log top channels
        logger.info("Top 10 most important channels:")
        for idx, row in importance_df.tail(10).iterrows():
            logger.info(f"  {row['channel']} ({row['type']}): {row['importance']:.3f}")
        
        # Create horizontal bar plot
        fig, ax = plt.subplots(figsize=(10, max(6, len(importance_df) * 0.3)))
        
        colors = ['#1f77b4' if t == 'SR' else '#ff7f0e' for t in importance_df['type']]
        
        y_pos = np.arange(len(importance_df))
        ax.barh(y_pos, importance_df['importance'], color=colors, alpha=0.7)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(importance_df['channel'], fontsize=8)
        ax.set_xlabel('Importance Score (normalized deviation)')
        ax.set_title('Channel Importance Ranking')
        ax.grid(True, alpha=0.3, axis='x')
        
        # Add legend
        from matplotlib.patches import Patch
        legend_elements = [
            Patch(facecolor='#1f77b4', alpha=0.7, label='SR'),
            Patch(facecolor='#ff7f0e', alpha=0.7, label='CR')
        ]
        ax.legend(handles=legend_elements, loc='lower right')
        
        plt.tight_layout()
        out_path = os.path.join(output_dir, "channel_importance_ranking.pdf")
        plt.savefig(out_path, dpi=300)
        plt.close()
        logger.info(f"Saved channel importance ranking: {out_path}")
        
        # Save to CSV
        csv_path = os.path.join(output_dir, "channel_importance_ranking.csv")
        importance_df.sort_values('importance', ascending=False).to_csv(csv_path, index=False)
        logger.info(f"Saved channel importance CSV: {csv_path}")
        
    except Exception:
        logger.exception("Error creating channel importance plot.")
        raise


def create_dimreduction_plots(df, selected_columns, output_dir, n_components=2):
    """Create PCA plot for selected columns (SR/CR yields)."""
    logger.info("Creating PCA dimensionality-reduction plot...")
    if not selected_columns:
        logger.info("No columns selected for dimensionality reduction.")
        return

    cols = [c for c in selected_columns if c in df.columns]
    if not cols:
        logger.warning("None of the selected columns are present in the dataframe.")
        return

    data = df[cols].apply(pd.to_numeric, errors="coerce").dropna(axis=0, how="any")
    if data.shape[0] < 2:
        logger.warning("Not enough full rows to run PCA (need at least 2).")
        return

    # Standardize
    try:
        if SKLEARN_AVAILABLE:
            scaler = StandardScaler()
            X = scaler.fit_transform(data.values)
        else:
            # fallback simple scaling: mean 0, std 1 (numpy)
            X = (data.values - np.nanmean(data.values, axis=0)) / (np.nanstd(data.values, axis=0) + 1e-12)
    except Exception:
        logger.exception("Error while scaling data for dimensionality reduction.")
        raise

    # PCA
    try:
        pca = PCA(n_components=n_components) if SKLEARN_AVAILABLE else None
        if pca is not None:
            Xp = pca.fit_transform(X)
            plt.figure(figsize=(7, 6))
            plt.scatter(Xp[:, 0], Xp[:, 1], s=10, alpha=0.6)
            plt.xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}% var)")
            plt.ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}% var)")
            plt.title("PCA projection (first two components)")
            plt.tight_layout()
            out_path = os.path.join(output_dir, "pca_projection.pdf")
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved PCA projection: {out_path}")
            logger.info(f"PCA explained variance: PC1={pca.explained_variance_ratio_[0]:.3f}, PC2={pca.explained_variance_ratio_[1]:.3f}")
        else:
            logger.warning("sklearn PCA not available; skipping PCA plot.")
    except Exception:
        logger.exception("PCA failed.")
        raise

def create_yield_vs_nlldiff_2d_histograms(df, SR_cols, CR_cols, num_yield_cols, output_dir):
    """Create 2D histograms of yield vs nLL difference for each channel.
    
    For each yield feature (SR and CR), plots:
      - X-axis: 2(nLL_obs_mu1 - nLL_obs_mu0)
      - Y-axis: yield value
    
    Splits into 5x5 grids across multiple PDF files.
    
    Args:
        df: DataFrame with yields followed by 8 nLL columns
        SR_cols: List of signal region column names
        CR_cols: List of control region column names
        num_yield_cols: Number of yield columns
        output_dir: Output directory for plots
    """
    logger.info("Creating 2D yield vs nLL difference histograms...")
    
    try:
        all_cols = SR_cols + CR_cols
        if not all_cols:
            logger.warning("No channels to plot.")
            return
        
        # Extract nLL columns (observed case: columns num_yield_cols + 2 and num_yield_cols + 3)
        if df.shape[1] < num_yield_cols + 4:
            logger.warning(
                f"DataFrame has {df.shape[1]} columns but expected at least {num_yield_cols + 4}. "
                "Skipping 2D yield vs nLL histograms."
            )
            return
        
        idx_nll_obs_mu0 = num_yield_cols + 2
        idx_nll_obs_mu1 = num_yield_cols + 3
        
        # Compute nLL difference for all rows
        nll_obs_mu0 = pd.to_numeric(df.iloc[:, idx_nll_obs_mu0], errors="coerce")
        nll_obs_mu1 = pd.to_numeric(df.iloc[:, idx_nll_obs_mu1], errors="coerce")
        nll_diff = 2 * (nll_obs_mu1 - nll_obs_mu0)
        
        # Create plots in 5x5 grid, split across multiple files
        n_channels = len(all_cols)
        grid_size = 5
        plots_per_file = grid_size * grid_size
        n_files = int(np.ceil(n_channels / plots_per_file))
        
        logger.info(f"Creating {n_files} figure(s) with {grid_size}x{grid_size} grid ({plots_per_file} plots per file)")
        
        nbins = 50
        
        for file_idx in range(n_files):
            fig, axs = plt.subplots(grid_size, grid_size, figsize=(16, 16))
            axs = axs.flatten()
            
            start_idx = file_idx * plots_per_file
            end_idx = min(start_idx + plots_per_file, n_channels)
            
            for plot_idx, col_idx in enumerate(range(start_idx, end_idx)):
                col = all_cols[col_idx]
                ax = axs[plot_idx]
                
                # Get yield values
                yields = pd.to_numeric(df[col], errors="coerce")
                
                # Remove NaN values (align both arrays)
                valid_mask = ~(yields.isna() | nll_diff.isna())
                valid_yields = yields[valid_mask]
                valid_nll_diff = nll_diff[valid_mask]
                
                if len(valid_yields) == 0:
                    logger.warning(f"No valid data for channel {col}")
                    ax.text(0.5, 0.5, f"{col}\n(no data)", ha='center', va='center')
                    ax.set_title(col, fontsize=10)
                    continue
                
                # Create 2D histogram
                h = ax.hist2d(
                    valid_nll_diff, valid_yields,
                    bins=nbins,
                    cmap='viridis',
                    cmin=1  # Avoid showing empty bins at zero
                )
                
                # Format axes
                ax.set_xlabel(r"$2(\Delta$nLL$_{\text{obs,}\mu=1} - \Delta$nLL$_{\text{obs,}\mu=0})$", fontsize=8)
                ax.set_ylabel("Yield", fontsize=8)
                ax.set_title(col, fontsize=10, fontweight='bold')
                ax.tick_params(labelsize=7)
                
                # Add colorbar
                cbar = plt.colorbar(h[3], ax=ax)
                cbar.set_label("Counts", fontsize=8)
                cbar.ax.tick_params(labelsize=7)
                
                logger.info(f"Plotted {col} (file {file_idx + 1}/{n_files}, plot {plot_idx + 1}/{plots_per_file})")
            
            # Hide unused subplots
            for plot_idx in range(end_idx - start_idx, plots_per_file):
                axs[plot_idx].set_visible(False)
            
            plt.tight_layout()
            out_path = os.path.join(output_dir, f"yield_vs_nlldiff_2d_histograms_{file_idx + 1}.pdf")
            plt.savefig(out_path, dpi=300)
            plt.close()
            logger.info(f"Saved 2D histogram file {file_idx + 1}/{n_files}: {out_path}")
    
    except Exception:
        logger.exception("Error creating 2D yield vs nLL difference histograms.")
        raise

# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Plot scan visualization tool.")
    parser.add_argument("csv_file", help="Path to the CSV data file")
    parser.add_argument("--outdir", default=None,
                        help="Optional output directory for saving plots. If not provided, defaults to the CSV directory.")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    logger.info("Starting plot generation.")

    try:
        json_file = validate_inputs(args.csv_file)
        df, metadata = load_data(args.csv_file)
        index = validate_metadata(metadata)
    except Exception:
        logger.exception("Fatal error during input validation or loading.")
        sys.exit(1)

    # Use the correct channel set
    try:
        channels_dict = metadata["channels"][index]
    except Exception:
        logger.exception("Selected channel set index invalid.")
        sys.exit(1)

    # determine output dir
    output_dir = args.outdir if args.outdir else os.path.dirname(os.path.abspath(args.csv_file))
    if not os.path.isdir(output_dir):
        logger.info(f"Output directory does not exist. Creating: {output_dir}")
        try:
            os.makedirs(output_dir, exist_ok=True)
        except Exception:
            logger.exception(f"Failed to create output directory: {output_dir}")
            sys.exit(1)

    SR_names, SRs = [], []
    CR_names, CRs = [], []

    removed = metadata["remove_channels"]
    lower_limits = []
    upper_limits = []
    observed = []

    logger.info(f"Channels: {channels_dict}")
    logger.info(f"Removed channels: {removed}")

    for ii, (ch_name, ch_type) in enumerate(channels_dict.items()):
        if ch_name in removed:
            logger.info(f"Skipping removed channel: {ch_name}")
            continue

        if ch_type in ["SR", "CR"]:
            if ch_type == "SR":
                SR_names.append(ch_name)
            else:
                CR_names.append(ch_name)

            lower_limits.append(metadata["lower_limits"][ii])
            upper_limits.append(metadata["upper_limits"][ii])
            observed.append(metadata["obs_yields"][ii])

        elif ch_type == "VR":
            continue
        else:
            logger.error(f"Unknown channel type: {ch_type}")
            sys.exit(1)

    # Map DF columns to SR/CR
    for col in df.columns:
        prefix = col.split("-")[0]
        if prefix in SR_names:
            SRs.append(col)
        elif prefix in CR_names:
            CRs.append(col)

    num_yield_cols = len(SRs + CRs)
    logger.info(f"Total yield columns: {num_yield_cols} (SRs: {len(SRs)}, CRs: {len(CRs)})")
    logger.info(f"Total DataFrame columns: {df.shape[1]}")

    # -------------------------
    # Plotting
    # -------------------------
    try:
        # existing plots
        create_violin_plots(df, [(SRs, "SRs"), (CRs, "CRs")], output_dir)
        create_pairplots(df, lower_limits, upper_limits, observed, [(SRs, "SRs"), (CRs, "CRs")], output_dir)

        # Merged likelihood ratio histograms
        create_lratio_histograms(df, num_yield_cols, output_dir)
        # nLL histograms
        create_nlls_histograms(df, num_yield_cols, output_dir)
        # ---- Diagnostic plots ----
        # Outlier detection (z-score histogram only)
        selected_for_outliers = SRs + CRs
        create_outlier_plots(df, selected_for_outliers, output_dir, z_threshold=3.0)

        # Channel importance ranking
        create_channel_importance_plot(df, SRs, CRs, lower_limits, upper_limits, observed, output_dir)

        # PCA of SR/CR yields
        create_dimreduction_plots(df, selected_for_outliers, output_dir, n_components=2)

        # 2D histograms: yield vs nLL difference
        create_yield_vs_nlldiff_2d_histograms(df, SRs, CRs, num_yield_cols, output_dir)
    except Exception:
        logger.exception("Fatal plotting error.")
        sys.exit(1)

    logger.info("All plots generated successfully.")


if __name__ == "__main__":
    main()