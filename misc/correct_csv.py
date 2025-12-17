import argparse
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import json
from datetime import datetime


# Simple logging system
class Logger:
    LEVELS = {'DEBUG': 0, 'INFO': 1, 'WARNING': 2, 'ERROR': 3}

    def __init__(self, verbosity='INFO'):
        self.verbosity = self.LEVELS.get(verbosity.upper(), 1)

    def _log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.LEVELS[level] >= self.verbosity:
            print(f"[{timestamp}] [{level}] {message}")

    def debug(self, message): self._log('DEBUG', message)
    def info(self, message): self._log('INFO', message)
    def warning(self, message): self._log('WARNING', message)
    def error(self, message): self._log('ERROR', message)

logger = Logger()


def load_csv(file_path):
    logger.info(f"Loading CSV: {file_path}")
    try:
        df = pd.read_csv(file_path)
        logger.info(f"Loaded {df.shape[0]} rows x {df.shape[1]} columns")
        return df
    except Exception as e:
        logger.error(f"Failed to load CSV: {e}")
        sys.exit(1)


def identify_yield_cols(df):
    nll_patterns = ['nLL', 'nLLA']
    yield_cols = []
    for col in df.columns:
        if not any(p in col for p in nll_patterns):
            yield_cols.append(col)
        else:
            break
    return yield_cols


def identify_nll_cols(df):
    nll_patterns = ['nLL', 'nLLA']
    nll_cols = []
    found_first = False
    for col in df.columns:
        if any(p in col for p in nll_patterns) and '0' not in col:
            nll_cols.append(col)
            found_first = True
        elif found_first:
            break
    return nll_cols


def compute_deltas(df):
    """Compute all 4 deltas, multiply by 2, and return as dictionary"""
    delta_pairs = [
        ('nLL_exp_mu0', 'nLL_exp_mu1'),
        ('nLL_obs_mu0', 'nLL_obs_mu1'),
        ('nLLA_exp_mu0', 'nLLA_exp_mu1'),
        ('nLLA_obs_mu0', 'nLLA_obs_mu1')
    ]

    delta_dict = {}

    for a, b in delta_pairs:
        if a in df.columns and b in df.columns:
            delta_col = f"{a}_vs_{b}"
            delta_dict[delta_col] = 2 * (df[b] - df[a])

    return delta_dict


def zscore_filter(df, yield_cols, z_threshold):
    """Apply z-score filtering to yield columns"""
    zscores = pd.DataFrame(index=df.index)
    for col in yield_cols:
        mean = df[col].mean()
        std = df[col].std()
        if std > 0:
            zscores[col] = np.abs((df[col] - mean) / std)
        else:
            zscores[col] = 0.0
            logger.warning(f"Column {col} has zero standard deviation")

    mask = zscores.gt(z_threshold).any(axis=1)
    removal_counts = {}
    for col in yield_cols:
        count = (zscores[col] > z_threshold).sum()
        if count > 0:
            removal_counts[col] = count

    total_removed = mask.sum()
    percent_removed = 100 * total_removed / len(df)
    logger.info(f"Z-score filter ({z_threshold}σ) removal summary: removed {total_removed}/{len(df)} ({percent_removed:.2f}%)")
    
    if removal_counts:
        log_removal_counts(removal_counts, total_removed, len(df))

    df_filtered = df[~mask].copy()
    return df_filtered, zscores, removal_counts


def nll_filter(df, threshold):
    """Filter based on nLL values at mu=1 exceeding threshold"""
    nll_mu1_cols = [col for col in df.columns if 'nLL' in col and 'mu1' in col and '0' not in col]
    
    if not nll_mu1_cols:
        logger.warning("No nLL mu=1 columns found for filtering")
        return df, {}

    removal_counts = {}
    mask = pd.Series(False, index=df.index)

    for col in nll_mu1_cols:
        col_mask = df[col] > threshold
        removal_counts[col] = col_mask.sum()
        mask = mask | col_mask

    total_removed = mask.sum()
    percent_removed = 100 * total_removed / len(df)
    logger.info(f"nLL filter [{threshold}] removal summary: removed {total_removed}/{len(df)} ({percent_removed:.2f}%)")
    
    if removal_counts:
        log_removal_counts(removal_counts, total_removed, len(df))

    df_filtered = df[~mask].copy()
    return df_filtered, removal_counts


def delta_filter(df, delta_dict, threshold):
    """Filter based on delta values exceeding threshold"""
    mask = pd.Series(False, index=df.index)
    removal_counts = {}

    for delta_col, delta_vals in delta_dict.items():
        col_mask = delta_vals > threshold
        removal_counts[delta_col] = col_mask.sum()
        mask = mask | col_mask

    total_removed = mask.sum()
    percent_removed = 100 * total_removed / len(df)
    logger.info(f"Delta filter [{threshold}] removal summary: removed {total_removed}/{len(df)} ({percent_removed:.2f}%)")
    
    if removal_counts:
        log_removal_counts(removal_counts, total_removed, len(df))

    df_filtered = df[~mask].copy()
    return df_filtered, removal_counts


def log_removal_counts(removal_counts, total_removed, original_count):
    """Log removal counts with ASCII histogram, aligned and sorted by descending count"""
    # Sort by count descending
    sorted_counts = sorted(removal_counts.items(), key=lambda x: x[1], reverse=True)
    
    # Find max label length for alignment
    max_label_len = max(len(label) for label, _ in sorted_counts)
    
    # Calculate bar width based on max count
    max_count = sorted_counts[0][1] if sorted_counts else 1
    max_bar_width = 40
    
    for label, count in sorted_counts:
        if count > 0:
            bar_width = int(max_bar_width * count / max_count)
            bar = '█' * bar_width
            percent = 100 * count / original_count
            logger.info(f"  {label:<{max_label_len}} | {bar:<{max_bar_width}} | {count:7d} ({percent:6.2f}%)")


def plot_deltas(delta_dict_before, delta_dict_after, save_plot=False, output_dir='./'):
    """Plot delta distributions with before/after filtering overlaid"""
    delta_pairs = list(delta_dict_before.keys())
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    colors_before = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
    colors_after = ['#0055aa', '#dd6600', '#208000', '#aa0000']

    for i, delta_col in enumerate(delta_pairs):
        ax = axes[i]
        vals_before = delta_dict_before[delta_col]
        vals_after = delta_dict_after[delta_col]

        # Plot before filtering
        ax.hist(vals_before, bins=100, alpha=0.5, edgecolor='black', linewidth=0.5,
                color=colors_before[i], label=f'Before (n={len(vals_before)})')

        # Plot after filtering
        ax.hist(vals_after, bins=100, alpha=0.5, edgecolor='black', linewidth=0.5,
                color=colors_after[i], label=f'After (n={len(vals_after)})')

        # Statistics
        mean_before = vals_before.mean()
        mean_after = vals_after.mean()

        ax.axvline(mean_before, color=colors_before[i], linestyle='--', linewidth=2, label=f'Mean before: {mean_before:.2f}')
        ax.axvline(mean_after, color=colors_after[i], linestyle='--', linewidth=2, label=f'Mean after: {mean_after:.2f}')

        ax.set_title(delta_col, fontsize=12, fontweight='bold')
        ax.set_xlabel('Delta nLL')
        ax.set_ylabel('Counts')
        ax.set_yscale('log')
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=9)

    plt.tight_layout()
    if save_plot:
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        plot_file = output_path / 'delta_histograms.png'
        plt.savefig(plot_file, dpi=300, bbox_inches='tight')
        logger.info(f"Saved delta histograms to {plot_file}")
    plt.show()


def parse_arguments():
    parser = argparse.ArgumentParser(description='CSV filtering and analysis tool')
    parser.add_argument('input', help='Input CSV file')
    parser.add_argument('-o', '--output', help='Output CSV file', required=True)
    parser.add_argument('--z', type=float, default=None, help='Z-score threshold (e.g., 3.0)')
    parser.add_argument('--nll', type=float, default=None, help='nLL threshold for mu=1')
    parser.add_argument('--delta', type=float, default=None, help='Delta threshold (e.g., 150)')
    parser.add_argument('--dry-run', action='store_true', help='Dry run mode (no file written)')
    parser.add_argument('--save-plot', action='store_true', help='Save plots to file')
    parser.add_argument('--no-plot', action='store_true', help='Do not display plots')
    return parser.parse_args()


def main():
    args = parse_arguments()
    
    # Check if any filtering is requested
    if args.z is None and args.nll is None and args.delta is None:
        logger.warning("No filtering thresholds provided (--z, --nll, --delta). Data will not be filtered.")
    
    # Load data
    df = load_csv(args.input)
    original_count = len(df)
    yield_cols = identify_yield_cols(df)
    
    logger.info(f"Identified {len(yield_cols)} yield columns")
    
    # Compute deltas on original data (for plotting)
    delta_dict_original = compute_deltas(df)
    
    # Apply filters in sequence: Z-score -> nLL -> Delta
    
    # Z-score filter
    if args.z is not None:
        logger.info(f"Applying z-score filter (threshold={args.z})")
        df, zscores, removal_counts_z = zscore_filter(df, yield_cols, args.z)
        logger.info(f"Rows remaining: {len(df)}/{original_count}")
    
    # nLL filter (only for mu=1)
    if args.nll is not None:
        logger.info(f"Applying nLL filter for mu=1 (threshold={args.nll})")
        df, removal_counts_nll = nll_filter(df, args.nll)
        logger.info(f"Rows remaining: {len(df)}/{original_count}")
    
    # Compute deltas after all non-delta filters
    delta_dict_after_all = compute_deltas(df)
    
    # Delta filter
    if args.delta is not None:
        logger.info(f"Applying delta filter (threshold={args.delta})")
        df, removal_counts_delta = delta_filter(df, delta_dict_after_all, args.delta)
        logger.info(f"Rows remaining: {len(df)}/{original_count}")
        # Recompute deltas for final plot
        delta_dict_after_all = compute_deltas(df)
    
    # Plot deltas (before/after all filtering) - always make the plot unless --no-plot
    if not args.no_plot:
        output_dir = str(Path(args.output).parent) if Path(args.output).parent != Path('.') else './'
        plot_deltas(delta_dict_original, delta_dict_after_all, 
                   save_plot=args.save_plot, output_dir=output_dir)
    
    # Save CSV if not dry-run
    if not args.dry_run:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Saved cleaned CSV to {output_path}")
        total_removed = original_count - len(df)
        percent_removed = 100 * total_removed / original_count if original_count > 0 else 0
        logger.info(f"Total summary: removed {total_removed}/{original_count} ({percent_removed:.2f}%)")
        
        # Copy and update JSON file with metadata
        src_json = Path(args.input).with_suffix('.json')
        
        # Handle output file with or without extension
        if output_path.suffix == '.csv':
            dst_json = output_path.with_suffix('.json')
        else:
            # No extension provided, append .json
            dst_json = Path(str(output_path) + '.json')
        
        try:
            if src_json.exists():
                # Read the original JSON file
                with open(src_json, 'r') as f:
                    metadata = json.load(f)
                
                # Update metadata fields
                metadata['total_points'] = len(df)
                metadata['modified'] = True
                metadata['filtering_applied'] = {
                    'zscore': args.z if args.z is not None else False,
                    'nll': args.nll if args.nll is not None else False,
                    'delta': args.delta if args.delta is not None else False,
                    'original_points': original_count,
                    'removed_points': total_removed,
                    'removal_percentage': round(percent_removed, 2)
                }
                
                # Write updated metadata to destination
                dst_json.parent.mkdir(parents=True, exist_ok=True)
                with open(dst_json, 'w') as f:
                    json.dump(metadata, f, indent=2)
                
                logger.info(f"Metadata updated and saved to {dst_json} (total_points = {len(df)})")
            else:
                logger.warning(f"JSON file not found at {src_json}, skipping metadata update")
        except Exception as e:
            logger.warning(f"Could not process JSON file: {e}")
    else:
        logger.info(f"Dry-run mode: CSV not saved. Would have saved {len(df)} rows")

if __name__ == '__main__':
    main()