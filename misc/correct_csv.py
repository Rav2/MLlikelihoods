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
    """Simple logging system with different log levels."""
    
    LEVELS = {
        'DEBUG': 0,
        'INFO': 1,
        'WARNING': 2,
        'ERROR': 3
    }
    
    def __init__(self, verbosity='INFO'):
        self.verbosity = self.LEVELS.get(verbosity.upper(), 1)
    
    def _log(self, level, message):
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        if self.LEVELS[level] >= self.verbosity:
            print(f"[{timestamp}] [{level}] {message}")
    
    def debug(self, message):
        self._log('DEBUG', message)
    
    def info(self, message):
        self._log('INFO', message)
    
    def warning(self, message):
        self._log('WARNING', message)
    
    def error(self, message):
        self._log('ERROR', message)


logger = Logger()


def load_and_inspect_data(input_file):
    """
    Load CSV data and display basic information about the dataset.
    
    Args:
        input_file (str): Path to the input CSV file
        
    Returns:
        pd.DataFrame: Loaded dataframe
        
    Raises:
        FileNotFoundError: If the input file doesn't exist
        pd.errors.EmptyDataError: If the CSV file is empty
    """
    logger.info(f"Loading data from: {input_file}")
    
    try:
        if not input_file.lower().endswith('.csv'):
            logger.warning(f"Input file does not have .csv extension: {input_file}")
        
        df = pd.read_csv(input_file)
        logger.info(f"Data loaded successfully")
        logger.info(f"Dataset shape: {df.shape[0]} rows × {df.shape[1]} columns")
        logger.debug(f"Columns: {df.columns.tolist()}")
        logger.info(f"First 5 rows:\n{df.head()}")
        
        return df
    except FileNotFoundError:
        logger.error(f"Input file '{input_file}' not found")
        sys.exit(1)
    except pd.errors.EmptyDataError:
        logger.error(f"Input file '{input_file}' is empty")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        sys.exit(1)


def identify_yield_columns(df):
    """
    Identify yield columns (all columns before nLL/nLLA columns).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        list: List of yield column names
    """
    nll_patterns = ['nLL', 'nLLA']
    
    yield_cols = []
    for col in df.columns:
        if not any(pattern in col for pattern in nll_patterns):
            yield_cols.append(col)
        else:
            break
    
    logger.debug(f"Identified {len(yield_cols)} yield columns: {yield_cols}")
    return yield_cols


def validate_thresholds(low_threshold, up_threshold, zscore_threshold):
    """
    Validate threshold values.
    
    Args:
        low_threshold (float): Lower outlier threshold
        up_threshold (float): Upper outlier threshold
        zscore_threshold (float): Z-score threshold (or None if not set)
        
    Returns:
        bool: True if all thresholds are valid
    """
    logger.info("Validating threshold values")
    
    if low_threshold >= up_threshold:
        logger.error(f"Invalid thresholds: low_threshold ({low_threshold}) >= up_threshold ({up_threshold})")
        sys.exit(1)
    
    if zscore_threshold is not None and zscore_threshold <= 0:
        logger.error(f"Z-score threshold must be positive, got: {zscore_threshold}")
        sys.exit(1)
    
    logger.info(f"Threshold validation passed")
    logger.debug(f"  Low threshold: {low_threshold}")
    logger.debug(f"  Up threshold: {up_threshold}")
    if zscore_threshold is not None:
        logger.debug(f"  Z-score threshold: {zscore_threshold}")
    
    return True


def count_removals_per_column(df, yield_cols, mask_remove):
    """
    Count removed rows per column, assigning each removed row to the first column
    that caused its removal.
    
    Args:
        df (pd.DataFrame): Input dataframe
        yield_cols (list): List of yield column names
        mask_remove (pd.Series): Boolean mask of rows to remove
        
    Returns:
        dict: Dictionary with column names as keys and removal counts as values
    """
    removal_counts = {col: 0 for col in yield_cols}
    
    # For each row that needs to be removed, find the first column that caused it
    for idx in df.index[mask_remove]:
        for col in yield_cols:
            if idx in df.index and pd.notna(df.loc[idx, col]):
                # Check if this row violates the removal criteria for this column
                # (we'll pass the check function as parameter from caller)
                removal_counts[col] += 1
                break
    
    return removal_counts


def print_removal_summary(removal_counts, total_removed, initial_rows, method_name):
    """
    Print ASCII histogram and summary of removed rows per column.
    
    Args:
        removal_counts (dict): Dictionary with column names and removal counts
        total_removed (int): Total number of rows removed
        initial_rows (int): Initial number of rows in dataset
        method_name (str): Name of the removal method (e.g., "Range-based")
    """
    if total_removed == 0:
        logger.info(f"{method_name}: No rows removed")
        return
    
    logger.info(f"{method_name} removal summary:")
    logger.info(f"  Initial rows: {initial_rows}")
    logger.info(f"  Total removed: {total_removed} ({100*total_removed/initial_rows:.2f}%)")
    
    # Filter and sort by count (highest first)
    filtered_counts = {col: count for col, count in removal_counts.items() if count > 0}
    sorted_counts = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)
    
    if not sorted_counts:
        logger.info("No columns with removed rows")
        return
    
    # Find max count for scaling
    max_count = sorted_counts[0][1]
    max_bar_width = 40
    
    # Calculate terminal-friendly column width
    col_width = min(35, max(len(col) for col, _ in sorted_counts))
    
    logger.info("")
    logger.info("Rows removed per column (sorted by count):")
    logger.info("-" * (col_width + max_bar_width + 30))
    
    for col, count in sorted_counts:
        percentage = 100 * count / initial_rows
        bar_width = int((count / max_count) * max_bar_width) if max_count > 0 else 0
        bar = "█" * bar_width
        logger.info(f"  {col:{col_width}s} | {bar:{max_bar_width}s} | {count:7d} ({percentage:6.2f}%)")
    
    logger.info("-" * (col_width + max_bar_width + 30))
    logger.info("")


def clean_outliers_range(df, yield_cols, low_threshold, up_threshold, dry_run=False):
    """
    Remove rows with values outside specified range in yield columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        yield_cols (list): List of yield column names
        low_threshold (float): Lower threshold
        up_threshold (float): Upper threshold
        dry_run (bool): If True, only report what would be removed
        
    Returns:
        tuple: (cleaned dataframe or original, removal_counts dict)
    """
    logger.info(f"Cleaning outliers based on range thresholds")
    logger.debug(f"Lower threshold: {low_threshold}, Upper threshold: {up_threshold}")
    
    initial_shape = df.shape
    initial_rows = initial_shape[0]
    
    # Create mask for rows to remove and track which column caused it
    mask_remove = pd.Series([False] * len(df), index=df.index)
    removal_per_column = {col: pd.Series([False] * len(df), index=df.index) for col in yield_cols}
    
    for col in yield_cols:
        out_of_range = (df[col] < low_threshold) | (df[col] > up_threshold)
        removal_per_column[col] = out_of_range
        mask_remove |= out_of_range
    
    # Count removals per column (first column that caused removal)
    removal_counts = {col: 0 for col in yield_cols}
    for idx in df.index[mask_remove]:
        for col in yield_cols:
            if removal_per_column[col].loc[idx]:
                removal_counts[col] += 1
                break
    
    rows_to_remove = mask_remove.sum()
    
    # Print summary
    print_removal_summary(removal_counts, rows_to_remove, initial_rows, "Range-based")
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would remove {rows_to_remove} rows based on range thresholds")
        return df, removal_counts
    else:
        df_cleaned = df[~mask_remove]
        logger.info(f"Removed {rows_to_remove} rows based on range thresholds")
        logger.info(f"Dataset shape: {initial_shape} → {df_cleaned.shape}")
        return df_cleaned, removal_counts


def clean_outliers_zscore(df, yield_cols, zscore_threshold, dry_run=False):
    """
    Remove rows with z-score values exceeding threshold in yield columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        yield_cols (list): List of yield column names
        zscore_threshold (float): Z-score threshold (standard deviations from mean)
        dry_run (bool): If True, only report what would be removed
        
    Returns:
        tuple: (cleaned dataframe or original, zscore dataframe, removal_counts dict)
    """
    logger.info(f"Applying z-score based outlier removal (threshold: {zscore_threshold}σ)")
    
    initial_rows = df.shape[0]
    
    # Calculate z-scores for all yield columns
    zscores = pd.DataFrame(index=df.index)
    for col in yield_cols:
        mean_val = df[col].mean()
        std_val = df[col].std()
        zscores[col] = np.abs((df[col] - mean_val) / std_val)
        logger.debug(f"  Column '{col}': μ={mean_val:.6f}, σ={std_val:.6f}")
    
    # Create mask for rows to remove (where any column exceeds threshold)
    exceeds_threshold = zscores > zscore_threshold
    mask_remove = exceeds_threshold.any(axis=1)
    
    # Count removals per column (first column that caused removal)
    removal_counts = {col: 0 for col in yield_cols}
    for idx in df.index[mask_remove]:
        for col in yield_cols:
            if exceeds_threshold.loc[idx, col]:
                removal_counts[col] += 1
                break
    
    rows_to_remove = mask_remove.sum()
    
    # Print summary
    print_removal_summary(removal_counts, rows_to_remove, initial_rows, "Z-score based")
    
    if dry_run:
        logger.info(f"[DRY-RUN] Would remove {rows_to_remove} rows based on z-score threshold")
        return df, zscores, removal_counts
    else:
        logger.info(f"Removed {rows_to_remove} rows based on z-score threshold")
        logger.info(f"Dataset shape: {df.shape} → {df[~mask_remove].shape}")
        return df[~mask_remove], zscores, removal_counts


def create_histogram_plots(df, yield_cols, save_plot=False, output_dir="./"):
    """
    Create and display/save histograms of test statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        yield_cols (list): List of yield column names
        save_plot (bool): Whether to save the plot to file
        output_dir (str): Directory to save the plot
    """
    logger.info(f"Creating test statistics histograms for {len(yield_cols)} yield columns")
    
    # Define nLL/nLLA test statistics
    test_stats = [
        {
            'cols': ['nLL_exp_mu0', 'nLL_exp_mu1'],
            'name': 'ΔnLL (Expected)',
            'title': 'Expected μ₁ - μ₀'
        },
        {
            'cols': ['nLL_obs_mu0', 'nLL_obs_mu1'],
            'name': 'ΔnLL (Observed)',
            'title': 'Observed μ₁ - μ₀'
        },
        {
            'cols': ['nLLA_exp_mu0', 'nLLA_exp_mu1'],
            'name': 'ΔnLLA (Asimov Expected)',
            'title': 'Asimov Expected μ₁ - μ₀'
        },
        {
            'cols': ['nLLA_obs_mu0', 'nLLA_obs_mu1'],
            'name': 'ΔnLLA (Asimov Observed)',
            'title': 'Asimov Observed μ₁ - μ₀'
        }
    ]
    
    # Check which test statistics can be calculated
    available_stats = []
    for stat in test_stats:
        if all(col in df.columns for col in stat['cols']):
            available_stats.append(stat)
        else:
            missing = [col for col in stat['cols'] if col not in df.columns]
            logger.debug(f"Cannot plot {stat['name']}. Missing columns: {missing}")
    
    if not available_stats:
        logger.warning("Cannot create nLL/nLLA histograms. Required columns are missing.")
    else:
        # Create subplots for nLL/nLLA statistics
        n_plots = len(available_stats)
        n_cols = min(2, n_plots)
        n_rows = (n_plots + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
        fig.suptitle('Distribution of Test Statistics (nLL/nLLA)', 
                     fontsize=14, fontweight='bold')
        
        if n_plots == 1:
            axes = [axes]
        else:
            axes = axes.flatten()
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728']
        
        for i, stat in enumerate(available_stats):
            try:
                test_stat = df[stat['cols'][1]] - df[stat['cols'][0]]
                
                ax = axes[i]
                n, bins, patches = ax.hist(test_stat, bins=100, alpha=0.7, 
                                         edgecolor='black', linewidth=0.5, 
                                         color=colors[i])
                
                ax.set_xlabel(stat['name'])
                ax.set_ylabel('Counts')
                ax.set_yscale('log')
                ax.set_title(stat['title'])
                ax.grid(True, alpha=0.3)
                
                mean_val = test_stat.mean()
                std_val = test_stat.std()
                median_val = test_stat.median()
                
                ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                          label=f'Mean: {mean_val:.2f}')
                ax.axvline(median_val, color='orange', linestyle='-.', alpha=0.8, 
                          label=f'Median: {median_val:.2f}')
                
                stats_text = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nN = {len(test_stat):,}'
                ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
                       verticalalignment='top', bbox=dict(boxstyle='round', 
                       facecolor='wheat', alpha=0.8))
                
                ax.legend(loc='upper right')
                
            except Exception as e:
                logger.warning(f"Error creating histogram for {stat['name']}: {e}")
                continue
        
        # Hide unused subplots
        for j in range(len(available_stats), len(axes)):
            axes[j].set_visible(False)
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = Path(output_dir) / "test_statistics_histograms.png"
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            logger.info(f"Test statistics histograms saved to: {plot_path}")
        
        plt.show()


def create_zscore_plots(zscores, yield_cols, save_plot=False, output_dir="./"):
    """
    Create and display/save z-score histograms for all yield columns.
    
    Args:
        zscores (pd.DataFrame): Z-scores for yield columns
        yield_cols (list): List of yield column names
        save_plot (bool): Whether to save the plot to file
        output_dir (str): Directory to save the plot
    """
    logger.info(f"Creating z-score histograms for {len(yield_cols)} yield columns")
    
    n_cols = len(yield_cols)
    n_plot_cols = min(3, n_cols)  # Maximum 3 columns per row
    n_plot_rows = (n_cols + n_plot_cols - 1) // n_plot_cols
    
    fig, axes = plt.subplots(n_plot_rows, n_plot_cols, 
                             figsize=(5 * n_plot_cols, 4 * n_plot_rows))
    fig.suptitle('Distribution of Z-Scores (Yield Columns)', 
                 fontsize=14, fontweight='bold')
    
    if n_cols == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    colors = plt.cm.tab20(np.linspace(0, 1, n_cols))
    
    for i, col in enumerate(yield_cols):
        try:
            ax = axes[i]
            z_vals = zscores[col]
            
            n, bins, patches = ax.hist(z_vals, bins=100, alpha=0.7, 
                                      edgecolor='black', linewidth=0.5, 
                                      color=colors[i])
            
            ax.set_xlabel('|Z-Score|')
            ax.set_ylabel('Counts')
            ax.set_yscale('log')
            ax.set_title(f'{col}')
            ax.grid(True, alpha=0.3)
            
            mean_val = z_vals.mean()
            std_val = z_vals.std()
            median_val = z_vals.median()
            max_val = z_vals.max()
            
            ax.axvline(mean_val, color='red', linestyle='--', alpha=0.8, 
                      label=f'Mean: {mean_val:.2f}')
            ax.axvline(median_val, color='orange', linestyle='-.', alpha=0.8, 
                      label=f'Median: {median_val:.2f}')
            
            stats_text = f'μ = {mean_val:.2f}\nσ = {std_val:.2f}\nMax = {max_val:.2f}\nN = {len(z_vals):,}'
            ax.text(0.98, 0.97, stats_text, transform=ax.transAxes, 
                   verticalalignment='top', horizontalalignment='right',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            ax.legend(loc='upper left')
            
        except Exception as e:
            logger.warning(f"Error creating z-score histogram for {col}: {e}")
            continue
    
    # Hide unused subplots
    for j in range(len(yield_cols), len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    
    if save_plot:
        plot_path = Path(output_dir) / "zscore_histograms.png"
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        logger.info(f"Z-score histograms saved to: {plot_path}")
    
    plt.show()


def save_cleaned_data(df, output_file):
    """
    Save the cleaned dataframe to a CSV file.
    
    Args:
        df (pd.DataFrame): Cleaned dataframe to save
        output_file (str): Path to the output CSV file
    """
    logger.info(f"Saving cleaned data to: {output_file}")
    
    try:
        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        df.to_csv(output_file, index=False)
        logger.info(f"Cleaned data saved successfully ({len(df)} rows)")
    except Exception as e:
        logger.error(f"Error saving data: {e}")
        sys.exit(1)


def update_json_metadata(csv_file, output_file, df_cleaned):
    """
    Update JSON metadata file with cleaning information.
    
    Args:
        csv_file (str): Original CSV file path
        output_file (str): Output CSV file path
        df_cleaned (pd.DataFrame): Cleaned dataframe
    """
    src = csv_file.replace('.csv', '.json')
    output_path = Path(output_file)
    
    if output_path.suffix == '.csv':
        dst = str(output_path.with_suffix('.json'))
    else:
        dst = str(output_path) + '.json'
    
    logger.info(f"Updating metadata in: {dst}")
    
    try:
        if Path(src).exists():
            with open(src, 'r') as f:
                metadata = json.load(f)
            
            metadata['total_points'] = len(df_cleaned)
            metadata['modified'] = True
            metadata['modification_time'] = datetime.now().isoformat()
            
            with open(dst, 'w') as f:
                json.dump(metadata, f, indent=2)
            
            logger.debug(f"Metadata updated: total_points = {len(df_cleaned)}")
        else:
            logger.debug(f"JSON file not found at {src}, skipping metadata update")
    except Exception as e:
        logger.warning(f"Could not process JSON file: {e}")


def parse_arguments():
    """
    Parse command line arguments.
    
    Returns:
        argparse.Namespace: Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Process CSV data by cleaning outliers based on range and z-score thresholds, and generating visualizations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python correct_csv.py data.csv
    python correct_csv.py data.csv -o cleaned_data.csv
    python correct_csv.py data.csv --save-plot
    python correct_csv.py data.csv --up-threshold 50000 --low-threshold 0.001
    python correct_csv.py data.csv --zscore-threshold 3 --save-plot
    python correct_csv.py data.csv --dry-run --zscore-threshold 3
        """
    )
    
    parser.add_argument(
        'input',
        help='Path to input CSV file'
    )
    
    parser.add_argument(
        '-o', '--output',
        help='Path to output CSV file (default: output.csv in script directory)'
    )
    
    parser.add_argument(
        '--no-plot',
        action='store_true',
        help='Do not display plots'
    )
    
    parser.add_argument(
        '--save-plot',
        action='store_true',
        help='Save plots to files'
    )
    
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save cleaned data to output file'
    )
    
    parser.add_argument(
        '--up-threshold',
        type=float,
        default=1e5,
        help='Upper range threshold for outlier removal (default: 100000)'
    )
    
    parser.add_argument(
        '--low-threshold',
        type=float,
        default=1e-5,
        help='Lower range threshold for outlier removal (default: 0.00001)'
    )
    
    parser.add_argument(
        '--zscore-threshold',
        type=float,
        default=None,
        help='Z-score threshold (in standard deviations) for outlier removal. If not specified, no z-score filtering is applied'
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview what would be removed without actually removing or plotting'
    )
    
    return parser.parse_args()


def main():
    """
    Main function to orchestrate the data processing workflow.
    """
    args = parse_arguments()
    
    # Determine output file path
    if args.output:
        output_file = args.output
    else:
        script_dir = Path(__file__).parent
        output_file = script_dir / "output.csv"
    
    logger.info("=" * 70)
    logger.info("CSV Data Processing Script")
    logger.info("=" * 70)
    
    # Load and inspect data
    df = load_and_inspect_data(args.input)
    initial_rows = df.shape[0]
    
    # Identify yield columns
    yield_cols = identify_yield_columns(df)
    if not yield_cols:
        logger.error("No yield columns found in the dataset")
        sys.exit(1)
    
    logger.info(f"Initial dataset: {initial_rows} rows")
    
    # Validate thresholds
    validate_thresholds(args.low_threshold, args.up_threshold, args.zscore_threshold)
    
    # Clean outliers based on range
    df_cleaned, removal_counts_range = clean_outliers_range(df, yield_cols, args.low_threshold, 
                                                             args.up_threshold, args.dry_run)
    
    # Clean outliers based on z-score if specified
    zscores = None
    removal_counts_zscore = None
    if args.zscore_threshold is not None:
        df_cleaned, zscores, removal_counts_zscore = clean_outliers_zscore(df_cleaned, yield_cols, 
                                                                             args.zscore_threshold, args.dry_run)
    else:
        # Still calculate z-scores for plotting even if no threshold is applied
        if not args.no_plot:
            logger.debug("Calculating z-scores for visualization purposes")
            zscores = pd.DataFrame(index=df_cleaned.index)
            for col in yield_cols:
                mean_val = df_cleaned[col].mean()
                std_val = df_cleaned[col].std()
                zscores[col] = np.abs((df_cleaned[col] - mean_val) / std_val)
    
    # Print total removal summary
    total_removed = initial_rows - df_cleaned.shape[0]
    logger.info("")
    logger.info("=" * 70)
    logger.info(f"TOTAL ROWS REMOVED: {total_removed} out of {initial_rows} ({100*total_removed/initial_rows:.2f}%)")
    logger.info(f"REMAINING ROWS: {df_cleaned.shape[0]}")
    logger.info("=" * 70)
    logger.info("")
    
    # If dry-run mode, exit here
    if args.dry_run:
        logger.info("DRY-RUN MODE: No files were modified")
        logger.info("=" * 70)
        return
    
    # Create plots
    if not args.no_plot:
        output_dir = Path(output_file).parent
        
        # Plot nLL/nLLA statistics
        create_histogram_plots(df_cleaned, yield_cols, args.save_plot, output_dir)
        
        # Plot z-scores
        if zscores is not None:
            create_zscore_plots(zscores, yield_cols, args.save_plot, output_dir)
    
    # Save cleaned data
    if not args.no_save:
        save_cleaned_data(df_cleaned, output_file)
        update_json_metadata(args.input, output_file, df_cleaned)
    else:
        logger.info("Skipping data save (--no-save flag used)")
    
    logger.info("=" * 70)
    logger.info("Processing completed successfully!")
    logger.info("=" * 70)


if __name__ == "__main__":
    main()